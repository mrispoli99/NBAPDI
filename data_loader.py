import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
import io
import numpy as np

def _all_tables_with_ids(soup):
    """Return [(id, table_element), ...] for every table on the page, whether it
    lives directly in the DOM or is hidden inside an HTML comment (see _find_table)."""
    found = []
    for t in soup.find_all('table'):
        if t.get('id'):
            found.append((t.get('id'), t))

    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for c in comments:
        comment_soup = BeautifulSoup(c, 'html.parser')
        for t in comment_soup.find_all('table'):
            if t.get('id'):
                found.append((t.get('id'), t))
    return found

def _find_table(soup, keyword):
    """
    Basketball-Reference (and the wider Sports-Reference family) frequently ships tables
    wrapped inside an HTML comment: <!-- <table id="...">...</table> -->. A real browser
    runs client-side JS that un-comments them for display, but a plain BeautifulSoup
    .find('table', id=...) will not see inside a comment node, so it returns None even
    though the data is sitting right there in the raw markup.

    On top of that, the site's August-2024 "Upgraded" table redesign has been rolling out
    table-by-table and can change/namespace table ids (and split a page into separate
    Regular Season / Playoffs tables), so matching on an *exact* historical id like
    'per_game' is fragile. Instead we match any table whose id *contains* the keyword,
    skip anything that looks like a playoffs table, and prefer the largest match
    (the full-season regular-season table) when more than one candidate remains.
    """
    candidates = [
        (tid, table) for tid, table in _all_tables_with_ids(soup)
        if keyword in tid.lower() and 'playoff' not in tid.lower() and not tid.lower().endswith('_po')
    ]
    if not candidates:
        return None, []
    # Prefer the table with the most rows -- the full regular-season table, not a small widget
    candidates.sort(key=lambda pair: len(pair[1].find_all('tr')), reverse=True)
    return candidates[0][1], [tid for tid, _ in candidates]

def fetch_real_season_data(year):
    """
    Scrapes both Advanced and Per-Game stats to build a comprehensive historical snapshot.
    Defensively handles eras before 3PM (1980) and before Minutes Played tracking (1952).
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    adv_url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
    pg_url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
    
    res_adv = requests.get(adv_url, headers=headers)
    res_pg = requests.get(pg_url, headers=headers)
    
    if res_adv.status_code != 200 or res_pg.status_code != 200:
        print(f"[HTTP adv={res_adv.status_code} pg={res_pg.status_code}] ", end="", flush=True)
        return pd.DataFrame()
        
    soup_adv = BeautifulSoup(res_adv.content, 'html.parser')
    table_adv, adv_candidates = _find_table(soup_adv, 'advanced')
    
    soup_pg = BeautifulSoup(res_pg.content, 'html.parser')
    table_pg, pg_candidates = _find_table(soup_pg, 'per_game')
    
    if table_adv is None or table_pg is None:
        if table_adv is None:
            all_ids = sorted(set(tid for tid, _ in _all_tables_with_ids(soup_adv)))
            print(f"[no 'advanced'-like table; ids on page: {all_ids}] ", end="", flush=True)
        if table_pg is None:
            all_ids = sorted(set(tid for tid, _ in _all_tables_with_ids(soup_pg)))
            print(f"[no 'per_game'-like table; ids on page: {all_ids}] ", end="", flush=True)
        return pd.DataFrame()

    df_adv = pd.read_html(io.StringIO(str(table_adv)))[0]
    df_pg = pd.read_html(io.StringIO(str(table_pg)))[0]
    
    # Flatten column multi-indexes
    for df in [df_adv, df_pg]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[-1] if isinstance(col, tuple) else col for col in df.columns]
        df.columns = [str(c).strip() for c in df.columns]
        
    df_adv = df_adv[df_adv['Player'] != 'Player'].copy()
    df_pg = df_pg[df_pg['Player'] != 'Player'].copy()

    # --- ROBUST TEAM COLUMN DETECTION ---
    # Basketball-Reference has used both 'Tm' and 'Team' as the header for this column
    # depending on the table/era. Never assume it lives at a fixed index -- find it by name.
    for df, label in [(df_adv, 'advanced'), (df_pg, 'per_game')]:
        if 'Tm' not in df.columns:
            if 'Team' in df.columns:
                df.rename(columns={'Team': 'Tm'}, inplace=True)
            else:
                raise ValueError(
                    f"Could not locate a team column in the {label} table for {year}. "
                    f"Columns were: {list(df.columns)}"
                )

    # --- COLLAPSE TRADED-PLAYER ROWS ---
    # A player traded mid-season gets one row per team PLUS a 'TOT' (total) row.
    # Keep only the TOT row when present so each player appears once per season,
    # and so 'TOT' never gets treated as a real team downstream.
    for df in [df_adv, df_pg]:
        has_tot = df.groupby('Player')['Tm'].transform(lambda s: (s == 'TOT').any())
        df.drop(df[has_tot & (df['Tm'] != 'TOT')].index, inplace=True)
    
    # Convert types on Per-Game safely
    df_pg['PTS'] = pd.to_numeric(df_pg['PTS'], errors='coerce').fillna(0)
    df_pg['G'] = pd.to_numeric(df_pg['G'], errors='coerce').fillna(0)
    
    if '3P%' in df_pg.columns:
        df_pg['3P%'] = pd.to_numeric(df_pg['3P%'], errors='coerce').fillna(0)
    else:
        df_pg['3P%'] = 0.0  
    
    elite_scorers = df_pg[df_pg['PTS'] >= 20.0]['Player'].nunique()

    # --- LEAGUE-WIDE FOUL RATE (whistle tightness as a physicality proxy) ---
    # Fewer fouls called per game historically often meant defenders could play with more
    # contact before getting whistled -- a real difficulty factor independent of spacing.
    # We use the league-wide average of individual PF (personal fouls per game) as a simple,
    # real-data proxy for how tightly a season was officiated. Note this is a per-player,
    # per-game figure (not pace-adjusted), so it's a rough signal, not a precise one.
    if 'PF' in df_pg.columns:
        df_pg['PF'] = pd.to_numeric(df_pg['PF'], errors='coerce').fillna(0)
        # Only count players with meaningful playing time so garbage-time scrubs don't skew it
        qualified = df_pg[df_pg['G'] >= df_pg['G'].quantile(0.25)] if len(df_pg) > 0 else df_pg
        league_fouls_per_game = qualified['PF'].mean() if len(qualified) > 0 else np.nan
    else:
        league_fouls_per_game = np.nan

    # --- LEAGUE-WIDE 3-POINT ACCURACY (not any individual player's) ---
    # How physically/spatially demanding a season was is a property of the LEAGUE, not of
    # whichever player we happen to be scoring -- a non-shooting center shouldn't be treated
    # as if he were playing in a different era just because of his own shot profile.
    # Basketball-Reference's '3P%' column is already each player's season-long makes/attempts
    # ratio, so we recover totals via 3P% * 3PA (with 3PA scaled from per-game to season).
    if '3P' in df_pg.columns and '3PA' in df_pg.columns:
        df_pg['3P'] = pd.to_numeric(df_pg['3P'], errors='coerce').fillna(0)
        df_pg['3PA'] = pd.to_numeric(df_pg['3PA'], errors='coerce').fillna(0)
        total_3pm = (df_pg['3P'] * df_pg['G']).sum()
        total_3pa = (df_pg['3PA'] * df_pg['G']).sum()
        league_3pt_pct = (total_3pm / total_3pa) if total_3pa > 0 else 0.0
    else:
        league_3pt_pct = 0.0
    
    # Convert types on Advanced stats safely
    df_adv['WS'] = pd.to_numeric(df_adv['WS'], errors='coerce').fillna(0)
    df_adv['PER'] = pd.to_numeric(df_adv['PER'], errors='coerce').fillna(0)

    # USG% (Usage Percentage): estimate of the share of team possessions a player used
    # while on the floor. This is what lets us credit a player who carried a heavy
    # offensive load even when the team's overall Win Shares were low because the
    # roster around him was weak.
    if 'USG%' in df_adv.columns:
        df_adv['USG%'] = pd.to_numeric(df_adv['USG%'], errors='coerce').fillna(0)
    else:
        df_adv['USG%'] = 0.0
    
    # --- DEFENSIVE MINUTES CHECK (Pre-1952 Fix) ---
    if 'MP' in df_adv.columns:
        df_adv['MP'] = pd.to_numeric(df_adv['MP'], errors='coerce').fillna(0)
        # If the column exists but is entirely 0 (like 1950/1951 data sheets)
        if df_adv['MP'].sum() == 0:
            df_adv['MP'] = pd.to_numeric(df_adv['G'], errors='coerce').fillna(0)
    else:
        df_adv['MP'] = pd.to_numeric(df_adv['G'], errors='coerce').fillna(0)
    
    # --- CALC TEAMMATE INFLATION ADJUSTER ---
    # 'TOT' is not a real roster -- exclude those rows from team grouping so traded
    # players don't get bucketed together as if they shared a locker room.
    real_team_mask = df_adv['Tm'] != 'TOT'
    df_adv['Rotation_Rank'] = np.nan
    df_adv.loc[real_team_mask, 'Rotation_Rank'] = (
        df_adv[real_team_mask].groupby('Tm')['MP'].rank(ascending=False, method='first')
    )

    # Core rotation = top 5 players by minutes on a team. For each player IN that top 5,
    # their own PER must be excluded from their own "how good was my supporting cast" score --
    # otherwise a star who happens to rank 2nd-5th in team minutes gets partial credit for
    # his own production when measuring the quality of his help.
    def _core_support_excluding_self(group):
        top5 = group[group['Rotation_Rank'] <= 5]
        out = pd.Series(index=group.index, dtype=float)
        for idx in group.index:
            others = top5.drop(index=idx, errors='ignore')
            out[idx] = others['PER'].mean() if len(others) > 0 else np.nan
        return out

    df_adv['Core_Support_PER'] = np.nan
    df_adv.loc[real_team_mask, 'Core_Support_PER'] = (
        df_adv[real_team_mask].groupby('Tm', group_keys=False).apply(_core_support_excluding_self)
    )
    # Traded (TOT) players and anyone outside a resolvable team context fall back to league-average 15.0
    df_adv['Core_Support_PER'] = df_adv['Core_Support_PER'].fillna(15.0)
    
    df_pg_clean = df_pg[['Player', 'PTS', '3P%']].drop_duplicates(subset=['Player'])
    merged = df_adv.merge(df_pg_clean, on='Player', how='left')
    
    merged['League_20_PPG_Count'] = elite_scorers
    merged['League_3PT_Pct'] = league_3pt_pct
    merged['League_Fouls_Per_Game'] = league_fouls_per_game
    merged['Season'] = str(year)
    
    return merged[['Player', 'Pos', 'Age', 'Tm', 'MP', 'WS', 'PER', 'Core_Support_PER', 'PTS', '3P%',
                    'USG%', 'League_3PT_Pct', 'League_Fouls_Per_Game', 'League_20_PPG_Count', 'Season']]
def get_era_baselines(year):
    year = int(year)
    if year <= 1969:
        return {"league_baseline": 0.45, "kinematic_baseline": 0.60, "historical_teams": 9}
    elif 1970 <= year <= 1989:
        return {"league_baseline": 0.75, "kinematic_baseline": 0.85, "historical_teams": 22}
    elif 1990 <= year <= 2009:
        return {"league_baseline": 0.95, "kinematic_baseline": 1.05, "historical_teams": 30}
    else:
        return {"league_baseline": 1.00, "kinematic_baseline": 1.20, "historical_teams": 30}