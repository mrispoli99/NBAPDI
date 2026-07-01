import streamlit as st
import pandas as pd
import plotly.express as px
from difficulty_engine import load_all_real_seasons

st.set_page_config(layout="wide", page_title="NBA Player Difficulty Index")

st.title("🏀 NBA Player Difficulty Index (PDI) Studio")
st.markdown("Evaluating real dominance against historical context, athletic evolution, and teammate quality.")

with st.container(border=True):
    st.markdown("#### ℹ️ What is the PDI?")
    st.markdown(
        "The **Player Difficulty Index (PDI)** estimates how hard it was for a player to produce "
        "the season they had, given the era they played in, the quality of the teammates around them, "
        "how physical and spaced-out the game was, and how much of the offensive load they personally carried. "
        "It's scored 0–100, normalized against the single hardest season in NBA history, so 100 represents "
        "the toughest individual season ever recorded in this dataset -- not a perfect or 'ideal' player."
    )
    range_col1, range_col2, range_col3, range_col4 = st.columns(4)
    with range_col1:
        st.markdown("**80–100**")
        st.caption("All-time, historically brutal seasons")
    with range_col2:
        st.markdown("**55–79**")
        st.caption("Elite, top-of-era difficulty")
    with range_col3:
        st.markdown("**30–54**")
        st.caption("Strong, above-average difficulty")
    with range_col4:
        st.markdown("**0–29**")
        st.caption("Solid season, more favorable circumstances")

@st.cache_data
def get_cached_data():
    return load_all_real_seasons()

df = get_cached_data()
all_players = sorted(df['Player'].unique())

st.sidebar.header("Compare Options")
mode = st.sidebar.radio("Analysis Mode", ["Head-to-Head Comparison", "Player Career Timeline", "Player Leaderboard"])

if mode == "Head-to-Head Comparison":
    st.subheader("Compare Individual Seasonal Mountains")
    
    col1, col2 = st.columns(2)
    with col1:
        p1 = st.selectbox("Player 1", all_players, index=all_players.index("LeBron James") if "LeBron James" in all_players else 0)
        p1_seasons = ["All Years (Career Average)"] + sorted(df[df['Player'] == p1]['Season'].unique().tolist())
        s1 = st.selectbox("Season / Framework 1", p1_seasons)
        
    with col2:
        p2 = st.selectbox("Player 2", all_players, index=all_players.index("Michael Jordan") if "Michael Jordan" in all_players else 0)
        p2_seasons = ["All Years (Career Average)"] + sorted(df[df['Player'] == p2]['Season'].unique().tolist())
        s2 = st.selectbox("Season / Framework 2", p2_seasons)
        
    def resolve_metrics(player, choice):
        if choice == "All Years (Career Average)":
            p_data = df[df['Player'] == player]
            return {
                "score": p_data['pdi_final'].mean(), "ws": p_data['WS'].max(), "per": p_data['PER'].max(),
                "support_per": p_data['Core_Support_PER'].max(), "teams": p_data['Total_League_Teams'].max(),
                "p3": p_data['3P%'].max(), "league_p3": p_data['League_3PT_Pct'].max(),
                "scorers": p_data['League_20_PPG_Count'].max(),
                "modifier": p_data['Supporting_Cast_Modifier'].max(),
                "mp": p_data['MP'].max(), "usg": p_data['USG%'].max(),
                "burden_modifier": p_data['Offensive_Burden_Modifier'].max(),
                "playoff_friction": p_data['Playoff_Format_Friction'].max(),
                "label": f"{player} (Career Average)"
            }
        else:
            row = df[(df['Player'] == player) & (df['Season'] == choice)].iloc[0]
            return {
                "score": row['pdi_final'], "ws": row['WS'], "per": row['PER'],
                "support_per": row['Core_Support_PER'], "teams": row['Total_League_Teams'],
                "p3": row['3P%'], "league_p3": row['League_3PT_Pct'],
                "scorers": row['League_20_PPG_Count'],
                "modifier": row['Supporting_Cast_Modifier'],
                "mp": row['MP'], "usg": row['USG%'],
                "burden_modifier": row['Offensive_Burden_Modifier'],
                "playoff_friction": row['Playoff_Format_Friction'],
                "label": f"{player} ({choice})"
            }

    m1_res = resolve_metrics(p1, s1)
    m2_res = resolve_metrics(p2, s2)
    
    m1, m2 = st.columns(2)
    m1.metric(label=f"{m1_res['label']} PDI Score", value=f"{m1_res['score']:.1f} / 100")
    m2.metric(label=f"{m2_res['label']} PDI Score", value=f"{m2_res['score']:.1f} / 100")
    
    st.markdown("### 📊 Structural Context Comparison")
    context_data = pd.DataFrame([
        {"Variable / Metric": "Total Teams in League", m1_res['label']: f"{m1_res['teams']:.0f} Teams", m2_res['label']: f"{m2_res['teams']:.0f} Teams"},
        {"Variable / Metric": "League-Wide 20+ PPG Scorers", m1_res['label']: f"{m1_res['scorers']:.0f} Players", m2_res['label']: f"{m2_res['scorers']:.0f} Players"},
        {"Variable / Metric": "League-Wide 3-Point Accuracy (drives spacing difficulty)", m1_res['label']: f"{m1_res['league_p3']*100:.1f}%", m2_res['label']: f"{m2_res['league_p3']*100:.1f}%"},
        {"Variable / Metric": "Player's Own 3-Point Accuracy (context only, not used in score)", m1_res['label']: f"{m1_res['p3']*100:.1f}%", m2_res['label']: f"{m2_res['p3']*100:.1f}%"},
        {"Variable / Metric": "Teammate Core Rotation PER (Lower = Less Help)", m1_res['label']: f"{m1_res['support_per']:.1f} PER", m2_res['label']: f"{m2_res['support_per']:.1f} PER"},
        {"Variable / Metric": "Resulting Supporting Cast Weight Modifier", m1_res['label']: f"{m1_res['modifier']:.2f}x", m2_res['label']: f"{m2_res['modifier']:.2f}x"},
        {"Variable / Metric": "Player Individual PER", m1_res['label']: f"{m1_res['per']:.1f} PER", m2_res['label']: f"{m2_res['per']:.1f} PER"},
        {"Variable / Metric": "Win Shares (primary score driver)", m1_res['label']: f"{m1_res['ws']:.1f} WS", m2_res['label']: f"{m2_res['ws']:.1f} WS"},
        {"Variable / Metric": "Minutes Played (affects Playoff Format Friction eligibility)", m1_res['label']: f"{m1_res['mp']:.0f} MP", m2_res['label']: f"{m2_res['mp']:.0f} MP"},
        {"Variable / Metric": "Usage Rate (USG%)", m1_res['label']: f"{m1_res['usg']:.1f}%", m2_res['label']: f"{m2_res['usg']:.1f}%"},
        {"Variable / Metric": "Resulting Offensive Burden Modifier", m1_res['label']: f"{m1_res['burden_modifier']:.2f}x", m2_res['label']: f"{m2_res['burden_modifier']:.2f}x"},
        {"Variable / Metric": "Playoff Format Friction Bonus", m1_res['label']: f"{m1_res['playoff_friction']:.2f}", m2_res['label']: f"{m2_res['playoff_friction']:.2f}"},
    ])
    st.table(context_data.set_index("Variable / Metric"))

elif mode == "Player Career Timeline":
    st.subheader("Player Career Progression Through Eras")

    tcol1, tcol2 = st.columns(2)
    with tcol1:
        player_a = st.selectbox(
            "Player A", all_players,
            index=all_players.index("LeBron James") if "LeBron James" in all_players else 0,
            key="timeline_player_a"
        )
    with tcol2:
        player_b_options = ["(None -- show one player only)"] + all_players
        default_b = "Michael Jordan" if "Michael Jordan" in all_players else "(None -- show one player only)"
        player_b = st.selectbox(
            "Player B (optional, for comparison)", player_b_options,
            index=player_b_options.index(default_b),
            key="timeline_player_b"
        )

    selected_players = [player_a] if player_b == "(None -- show one player only)" else [player_a, player_b]
    timeline_df = df[df['Player'].isin(selected_players)].sort_values("Season").copy()

    # Career Year = 1st, 2nd, 3rd... season for THAT player, regardless of the actual
    # calendar year -- this is what lets two players from completely different eras
    # (e.g. a 1960s player vs. a 2020s player) line up on the same axis for comparison.
    timeline_df['Career_Year'] = timeline_df.groupby('Player')['Season'].rank(method='first').astype(int)

    chart_title = (
        f"{player_a}'s Career PDI Path" if len(selected_players) == 1
        else f"{player_a} vs {player_b}: Career PDI Path (aligned by career year)"
    )
    fig = px.line(
        timeline_df, x="Career_Year", y="pdi_final", color="Player", markers=True,
        hover_data=["Season"], title=chart_title,
        labels={"Career_Year": "Career Year", "pdi_final": "PDI Score"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(timeline_df[['Player', 'Career_Year', 'Season', 'pdi_final', 'Core_Support_PER', 'League_20_PPG_Count',
                               '3P%', 'League_3PT_Pct', 'WS', 'USG%', 'Offensive_Burden_Modifier',
                               'Playoff_Format_Friction']])

else:
    st.subheader("🏆 All-Time Player Leaderboard")
    st.markdown("Every player in the archive, ranked by Player Difficulty Index across their career.")

    leaderboard_df = df.groupby('Player').agg(
        Average_PDI=('pdi_final', 'mean'),
        Max_PDI=('pdi_final', 'max'),
        Seasons_Played=('Season', 'nunique')
    ).reset_index()

    # Look up which season each player's Max PDI actually happened in
    max_idx = df.groupby('Player')['pdi_final'].idxmax()
    max_year_lookup = df.loc[max_idx, ['Player', 'Season']].rename(columns={'Season': 'Max_PDI_Year'})
    leaderboard_df = leaderboard_df.merge(max_year_lookup, on='Player', how='left')

    leaderboard_df['Average_PDI'] = leaderboard_df['Average_PDI'].round(1)
    leaderboard_df['Max_PDI'] = leaderboard_df['Max_PDI'].round(1)
    leaderboard_df = leaderboard_df[['Player', 'Average_PDI', 'Max_PDI', 'Max_PDI_Year', 'Seasons_Played']]
    leaderboard_df = leaderboard_df.rename(columns={
        'Average_PDI': 'Average PDI', 'Max_PDI': 'Max PDI',
        'Max_PDI_Year': 'Max PDI Year', 'Seasons_Played': 'Seasons Played'
    })

    sort_col1, sort_col2 = st.columns([2, 1])
    with sort_col1:
        sort_by = st.selectbox("Sort by", ["Average PDI", "Max PDI", "Max PDI Year", "Seasons Played", "Player"])
    with sort_col2:
        sort_dir = st.radio("Order", ["Descending", "Ascending"], horizontal=True)

    leaderboard_df = leaderboard_df.sort_values(by=sort_by, ascending=(sort_dir == "Ascending"))

    st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)