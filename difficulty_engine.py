import pandas as pd
import numpy as np
from data_loader import get_era_baselines

def calculate_real_pdi(df, foul_baseline=None):
    """
    Applies multi-variable adjustments (Pace, Scale, Competition Concentration, 
    Positional Perimeter Stress, and Continuous Teammate Burden) to a seasonal group.

    foul_baseline: league-wide average fouls-per-game across ALL seasons in the archive,
    used as the reference point for the Foul Era Modifier below. Passed in from
    load_all_real_seasons() so a single season's calculation can be judged against the
    full historical range rather than an arbitrary hardcoded number.
    """
    if df.empty:
        return df
        
    year = int(df['Season'].iloc[0])
    baselines = get_era_baselines(year)
    
    n_teams = baselines['historical_teams']
    league_base = baselines['league_baseline']
    kinematic_base = baselines['kinematic_baseline']
    
    # 1. Base Performance Layer: Seasonal Win Share Z-Score
    df['dominance_z'] = (df['WS'] - df['WS'].mean()) / df['WS'].std()
    
    pdi_scores = []
    rsd_list = []
    psp_list = []
    scb_list = []
    burden_list = []
    foul_friction_list = []
    
    for idx, row in df.iterrows():
        # A. Strength of Competition (Concentration of 20+ PPG Buckets across the league)
        buckets_per_team = row['League_20_PPG_Count'] / n_teams
        competition_factor = 1.0 + (buckets_per_team * 0.15)
        
        # B. Spacing & Positional Perimeter Stress Engine
        # Handles the paradox of physical clog down low vs modern hunting/switching on the perimeter.
        # Driven by the SEASON'S ACTUAL LEAGUE-WIDE 3P% (computed in data_loader.py from real
        # league-wide makes/attempts) -- never any individual player's own shooting. A higher
        # league 3P% means defenses are spread thinner and more efficiently punished, which is
        # a property of the league that season, not of whichever player we're scoring right now.
        player_pos = str(row['Pos']).upper()
        if 'C' in player_pos or 'F' in player_pos:
            position_stress_multiplier = 1.10  # 10% premium for defensive perimeter range
        else:
            position_stress_multiplier = 1.02  # Small guard chasing premium

        if year < 1980:
            # PRE-3PT LINE / EARLY ERA: no 3-point data exists, so we credit the very real
            # physical congestion of the paint-bound game instead. Previously this was a flat
            # 1.15 with no positional differentiation at all, which systematically undervalued
            # this era relative to the post-3pt game (which gets both a league-wide spacing
            # multiplier AND a positional premium). Using a baseline comparable to a typical
            # modern season's spacing stress, plus the same positional premium, keeps this era
            # in the same ballpark rather than being structurally discounted.
            era_spacing_stress = 1.25
        else:
            era_spacing_stress = 1.0 + row.get('League_3PT_Pct', 0.0)

        spacing_friction = era_spacing_stress * position_stress_multiplier

        # B2. Foul Era ("Whistle Tightness") Modifier
        # A season with a lower league-wide foul rate than the historical norm implies looser
        # officiating -- more physical contact allowed, which is a real difficulty factor on
        # top of spacing. Self-calibrated against the archive's own historical average rather
        # than a guessed constant, and kept as a modest modifier since it's a rough, non-pace-
        # adjusted proxy (fewer fouls can also just mean a slower-paced game).
        league_fouls = row.get('League_Fouls_Per_Game', np.nan)
        if foul_baseline and foul_baseline > 0 and pd.notna(league_fouls):
            foul_friction = 1.0 + ((foul_baseline - league_fouls) * 0.03)
            foul_friction = np.clip(foul_friction, 0.92, 1.15)
        else:
            foul_friction = 1.0

        spacing_friction *= foul_friction
        
        # Combine macro filters into the Regular Season Baseline
        era_multiplier = league_base * kinematic_base * (np.log(n_teams) / np.log(30)) * competition_factor * spacing_friction
        rsd = row['dominance_z'] * era_multiplier
        
        # C. Playoff Bracket Format Friction
        # Smooth ramp instead of a hard on/off cliff -- a player a few dozen minutes under
        # the old MP>2400 threshold used to fall straight to zero bonus while a player just
        # over it got the full bonus, which could swing the final score by more than the
        # entire skill-based (Win Share) portion of the formula for two otherwise-similar
        # players. Now MP and WS each ramp linearly up to "fully qualified" rather than
        # snapping on/off, so a near-miss still gets a near-full (not zero) bonus.
        rounds = 4.0 if year >= 2003 else (3.5 if year >= 1984 else (2.5 if year >= 1970 else 1.5))
        max_psp = 0.6 * (rounds * (np.log(n_teams) / np.log(8)))
        mp_factor = np.clip((row['MP'] - 1800) / (2400 - 1800), 0.0, 1.0)
        ws_factor = np.clip((row['WS'] - 5.0) / (8.0 - 5.0), 0.0, 1.0)
        psp = max_psp * mp_factor * ws_factor
            
        # D. Continuous Teammate Efficiency Gradient (The Support Fix)
        # Centered exactly at a league-average 15.0 PER. No sudden cliffs.
        # Every point below 15 rewards a carrying job; every point above applies a luxury deflation.
        scb = 1.0 + ((15.0 - row['Core_Support_PER']) * 0.04)
        scb = np.clip(scb, 0.75, 1.30)  # Capped at a maximum 30% modifier swinging either way

        # E. Offensive Burden Modifier
        # Win Shares can understate a player who shouldered a massive scoring/offensive load
        # on a bad team (high USG%, low team success) -- the "30 PPG while the roster around
        # him was terrible" case. USG% (share of team possessions used while on the floor) is
        # the standard measure of that burden. We give a modest credit for usage above the
        # league-average ~20%, and amplify that credit specifically when Core_Support_PER shows
        # the supporting cast was weak -- that combination is exactly the "carried a bad team"
        # signal that a low team-wide WS would otherwise hide.
        usage = row.get('USG%', 0.0)
        usage = usage if usage > 0 else 20.0  # missing data falls back to a neutral load
        usage_burden = 1.0 + ((usage - 20.0) * 0.015)
        if row['Core_Support_PER'] < 15.0:
            teammate_weakness = (15.0 - row['Core_Support_PER']) * 0.02
            usage_burden *= (1.0 + teammate_weakness)
        usage_burden = np.clip(usage_burden, 0.85, 1.50)
        
        # Combine all structural layers
        raw_pdi = (rsd + psp) * scb * usage_burden
        pdi_scores.append(max(0, raw_pdi))
        rsd_list.append(rsd)
        psp_list.append(psp)
        scb_list.append(scb)
        burden_list.append(usage_burden)
        foul_friction_list.append(foul_friction)
        
    df['raw_pdi'] = pdi_scores
    df['Regular_Season_Difficulty_Base'] = rsd_list
    df['Playoff_Format_Friction'] = psp_list
    df['Supporting_Cast_Modifier'] = scb_list
    df['Offensive_Burden_Modifier'] = burden_list
    df['Foul_Era_Modifier'] = foul_friction_list
    df['Total_League_Teams'] = n_teams
    
    return df.sort_values(by='WS', ascending=False).head(50)

def load_all_real_seasons():
    """
    Loads the full pre-scraped 1950-2026 archive from the local CSV,
    processes seasonal baselines, and applies a true global max 0-100 normalization.
    """
    try:
        raw_df = pd.read_csv("nba_raw_archive.csv")
    except FileNotFoundError:
        print("Error: 'nba_raw_archive.csv' not found. Please run build_database.py first.")
        return pd.DataFrame()

    # Self-calibrate the Foul Era Modifier against this archive's own historical average
    # fouls-per-game, rather than a hardcoded guess about what "normal" looks like.
    if 'League_Fouls_Per_Game' in raw_df.columns:
        per_season_fouls = raw_df.groupby('Season')['League_Fouls_Per_Game'].first()
        foul_baseline = per_season_fouls.mean() if per_season_fouls.notna().any() else None
    else:
        foul_baseline = None
        
    processed_seasons = []
    for season, group in raw_df.groupby('Season'):
        processed_group = calculate_real_pdi(group, foul_baseline=foul_baseline)
        processed_seasons.append(processed_group)
        
    full_dataset = pd.concat(processed_seasons, ignore_index=True)
    
    # Global Max Normalization (Standardized Anchor across history)
    global_max = full_dataset['raw_pdi'].max()
    full_dataset['pdi_final'] = (full_dataset['raw_pdi'] / global_max) * 100
    
    return full_dataset