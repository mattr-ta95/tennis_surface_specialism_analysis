import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('/Users/matthewrussell/Documents/Github/tennis_surface_impact/data/atp_matches_combined.csv')

print("=" * 80)
print("EXTENDED ANALYSIS: SURFACE PERFORMANCE TRAJECTORIES")
print("=" * 80)

df['year'] = df['tourney_date'].astype(str).str[:4].astype(int)

# Create match records from both winner and loser perspectives
def create_player_match_records(df):
    winners = df[['winner_id', 'winner_name', 'winner_age', 'surface', 'year', 'tourney_date']].copy()
    winners.columns = ['player_id', 'player_name', 'age', 'surface', 'year', 'tourney_date']
    winners['won'] = 1

    losers = df[['loser_id', 'loser_name', 'loser_age', 'surface', 'year', 'tourney_date']].copy()
    losers.columns = ['player_id', 'player_name', 'age', 'surface', 'year', 'tourney_date']
    losers['won'] = 0

    all_matches = pd.concat([winners, losers], ignore_index=True)
    all_matches = all_matches.dropna(subset=['age'])
    return all_matches

player_matches = create_player_match_records(df)

# ============================================================================
# Analysis of rising stars with full trajectories in dataset
# ============================================================================
print("\n" + "=" * 80)
print("RISING STARS: SURFACE PERFORMANCE EVOLUTION")
print("=" * 80)

rising_stars = ['Jannik Sinner', 'Carlos Alcaraz', 'Holger Rune', 'Ben Shelton']

for player_name in rising_stars:
    player_data = player_matches[player_matches['player_name'] == player_name]

    if len(player_data) == 0:
        continue

    print(f"\n{'='*60}")
    print(f"{player_name.upper()}")
    print(f"{'='*60}")
    print(f"Total matches: {len(player_data)}")
    print(f"Age range: {player_data['age'].min():.1f} to {player_data['age'].max():.1f}")

    # Year by year progression
    print(f"\nYear-by-year surface performance:")
    yearly_surface = player_data.groupby(['year', 'surface']).agg(
        matches=('won', 'count'),
        wins=('won', 'sum'),
        win_rate=('won', 'mean')
    ).round(3)

    yearly_pivot = yearly_surface.reset_index().pivot(index='year', columns='surface', values='win_rate')
    matches_pivot = yearly_surface.reset_index().pivot(index='year', columns='surface', values='matches')

    print("\nWin rates by year and surface:")
    print(yearly_pivot.round(3).to_string())
    print("\nMatch counts:")
    print(matches_pivot.fillna(0).astype(int).to_string())

    # Calculate year-over-year changes
    print("\nYear-over-year win rate changes by surface:")
    for surface in ['Hard', 'Clay', 'Grass']:
        if surface in yearly_pivot.columns:
            changes = yearly_pivot[surface].diff()
            print(f"\n{surface}:")
            for year, change in changes.dropna().items():
                direction = "+" if change > 0 else ""
                print(f"  {int(year-1)} -> {int(year)}: {direction}{change:.3f}")

# ============================================================================
# Cohort Analysis: Young vs Old players
# ============================================================================
print("\n" + "=" * 80)
print("COHORT ANALYSIS: YOUNG VS VETERAN PERFORMANCE BY SURFACE")
print("=" * 80)

# Define cohorts
player_matches['cohort'] = pd.cut(
    player_matches['age'],
    bins=[0, 22, 26, 30, 100],
    labels=['Young (Under 23)', 'Prime (23-26)', 'Experienced (27-30)', 'Veteran (30+)']
)

cohort_surface = player_matches.groupby(['cohort', 'surface']).agg(
    matches=('won', 'count'),
    wins=('won', 'sum'),
    win_rate=('won', 'mean')
).round(3)

print("\nWin rates by cohort and surface:")
cohort_pivot = cohort_surface.reset_index().pivot(index='cohort', columns='surface', values='win_rate')
print(cohort_pivot.round(3))

print("\nMatch counts:")
cohort_matches = cohort_surface.reset_index().pivot(index='cohort', columns='surface', values='matches')
print(cohort_matches.fillna(0).astype(int))

# Surface where each cohort performs BEST relative to their overall win rate
print("\n" + "=" * 80)
print("RELATIVE SURFACE STRENGTH BY COHORT")
print("=" * 80)

for cohort in cohort_pivot.index:
    cohort_row = cohort_pivot.loc[cohort]
    overall_avg = cohort_row.mean()
    print(f"\n{cohort} (avg win rate: {overall_avg:.3f}):")
    for surface in cohort_row.index:
        diff = cohort_row[surface] - overall_avg
        print(f"  {surface}: {cohort_row[surface]:.3f} ({diff:+.3f} vs avg)")

# ============================================================================
# Year-over-Year Surface Performance Changes
# ============================================================================
print("\n" + "=" * 80)
print("SURFACE PERFORMANCE TRENDS BY YEAR")
print("=" * 80)

yearly_stats = player_matches.groupby(['year', 'surface']).agg(
    total_matches=('won', 'count'),
    avg_winner_age=('age', 'mean')
).reset_index()

# Win rate by year (this is always 0.5 due to data structure)
# Instead, let's look at winner's age trends
print("\nAverage age of match participants by year and surface:")
age_pivot = yearly_stats.pivot(index='year', columns='surface', values='avg_winner_age')
print(age_pivot.round(1))

# ============================================================================
# Analysis of players who improved most on specific surfaces
# ============================================================================
print("\n" + "=" * 80)
print("PLAYERS WITH BIGGEST SURFACE IMPROVEMENT OVER TIME")
print("=" * 80)

def calculate_player_improvement(player_data, min_matches_per_period=15):
    """Calculate improvement from first half to second half of career in dataset"""
    player_data = player_data.sort_values('tourney_date')
    mid_point = len(player_data) // 2

    if mid_point < min_matches_per_period:
        return None

    first_half = player_data.iloc[:mid_point]
    second_half = player_data.iloc[mid_point:]

    results = {}
    for surface in ['Hard', 'Clay', 'Grass']:
        first_surface = first_half[first_half['surface'] == surface]
        second_surface = second_half[second_half['surface'] == surface]

        if len(first_surface) >= 5 and len(second_surface) >= 5:
            first_wr = first_surface['won'].mean()
            second_wr = second_surface['won'].mean()
            results[surface] = {
                'first_half': first_wr,
                'second_half': second_wr,
                'improvement': second_wr - first_wr,
                'first_matches': len(first_surface),
                'second_matches': len(second_surface)
            }

    return results if results else None

# Find players with enough matches
player_counts = player_matches.groupby('player_id').size()
active_players = player_counts[player_counts >= 100].index

improvements = []
for player_id in active_players:
    player_data = player_matches[player_matches['player_id'] == player_id]
    player_name = player_data['player_name'].iloc[0]
    imp = calculate_player_improvement(player_data)

    if imp:
        for surface, stats in imp.items():
            improvements.append({
                'player_name': player_name,
                'surface': surface,
                **stats
            })

imp_df = pd.DataFrame(improvements)

print("\nTop 10 biggest improvers by surface:")
print("-" * 60)

for surface in ['Hard', 'Clay', 'Grass']:
    surface_imp = imp_df[imp_df['surface'] == surface].nlargest(5, 'improvement')
    print(f"\n{surface}:")
    for _, row in surface_imp.iterrows():
        print(f"  {row['player_name']}: {row['first_half']:.3f} -> {row['second_half']:.3f} ({row['improvement']:+.3f})")
        print(f"    (first {row['first_matches']} matches vs last {row['second_matches']} matches)")

print("\nTop 10 biggest decliners by surface:")
print("-" * 60)

for surface in ['Hard', 'Clay', 'Grass']:
    surface_dec = imp_df[imp_df['surface'] == surface].nsmallest(5, 'improvement')
    print(f"\n{surface}:")
    for _, row in surface_dec.iterrows():
        print(f"  {row['player_name']}: {row['first_half']:.3f} -> {row['second_half']:.3f} ({row['improvement']:+.3f})")
        print(f"    (first {row['first_matches']} matches vs last {row['second_matches']} matches)")

# ============================================================================
# Surface Specialists Analysis
# ============================================================================
print("\n" + "=" * 80)
print("SURFACE SPECIALISTS: PLAYERS WITH EXTREME SURFACE PREFERENCES")
print("=" * 80)

def identify_specialists(player_matches, min_matches=50):
    """Identify players with strong surface preferences"""
    specialists = []

    for player_id in player_matches['player_id'].unique():
        player_data = player_matches[player_matches['player_id'] == player_id]

        if len(player_data) < min_matches:
            continue

        player_name = player_data['player_name'].iloc[0]
        surface_stats = player_data.groupby('surface').agg(
            matches=('won', 'count'),
            win_rate=('won', 'mean')
        )

        # Only consider surfaces with at least 10 matches
        valid = surface_stats[surface_stats['matches'] >= 10]

        if len(valid) >= 2:
            best_surface = valid['win_rate'].idxmax()
            worst_surface = valid['win_rate'].idxmin()
            spread = valid['win_rate'].max() - valid['win_rate'].min()

            specialists.append({
                'player_name': player_name,
                'total_matches': len(player_data),
                'best_surface': best_surface,
                'best_wr': valid.loc[best_surface, 'win_rate'],
                'worst_surface': worst_surface,
                'worst_wr': valid.loc[worst_surface, 'win_rate'],
                'spread': spread
            })

    return pd.DataFrame(specialists)

specialists_df = identify_specialists(player_matches)

print("\nMost surface-specialized players (highest spread between best and worst surface):")
top_specialists = specialists_df.nlargest(15, 'spread')
for _, row in top_specialists.iterrows():
    print(f"\n{row['player_name']} ({row['total_matches']} matches):")
    print(f"  Best: {row['best_surface']} ({row['best_wr']:.3f})")
    print(f"  Worst: {row['worst_surface']} ({row['worst_wr']:.3f})")
    print(f"  Spread: {row['spread']:.3f}")

print("\n\nMost consistent players across surfaces (lowest spread):")
consistent = specialists_df[specialists_df['total_matches'] >= 100].nsmallest(10, 'spread')
for _, row in consistent.iterrows():
    print(f"\n{row['player_name']} ({row['total_matches']} matches):")
    print(f"  Best: {row['best_surface']} ({row['best_wr']:.3f})")
    print(f"  Worst: {row['worst_surface']} ({row['worst_wr']:.3f})")
    print(f"  Spread: {row['spread']:.3f}")

# ============================================================================
# Age at which specialists develop their preferences
# ============================================================================
print("\n" + "=" * 80)
print("WHEN DO SURFACE PREFERENCES DEVELOP?")
print("=" * 80)

def analyze_preference_development(player_matches):
    """Analyze when players develop their surface preferences"""
    results = []

    for player_id in player_matches['player_id'].unique():
        player_data = player_matches[player_matches['player_id'] == player_id].sort_values('tourney_date')

        if len(player_data) < 100:
            continue

        player_name = player_data['player_name'].iloc[0]

        # Look at first 50 matches vs later matches
        early = player_data.head(50)
        later = player_data.iloc[50:]

        if len(later) < 30:
            continue

        # Calculate spreads
        for period, data, label in [(early, early, 'early'), (later, later, 'later')]:
            surface_stats = data.groupby('surface').agg(
                matches=('won', 'count'),
                win_rate=('won', 'mean')
            )
            valid = surface_stats[surface_stats['matches'] >= 5]
            if len(valid) >= 2:
                spread = valid['win_rate'].max() - valid['win_rate'].min()
                results.append({
                    'player_name': player_name,
                    'period': label,
                    'avg_age': data['age'].mean(),
                    'spread': spread,
                    'best_surface': valid['win_rate'].idxmax()
                })

    return pd.DataFrame(results)

pref_dev = analyze_preference_development(player_matches)

if not pref_dev.empty:
    # Compare early vs later spreads
    early_avg = pref_dev[pref_dev['period'] == 'early']['spread'].mean()
    later_avg = pref_dev[pref_dev['period'] == 'later']['spread'].mean()

    print(f"\nAverage surface spread in first 50 matches: {early_avg:.3f}")
    print(f"Average surface spread in later career: {later_avg:.3f}")

    # How often does best surface change?
    pivot = pref_dev.pivot(index='player_name', columns='period', values='best_surface')
    pivot = pivot.dropna()
    changed = (pivot['early'] != pivot['later']).sum()
    total = len(pivot)
    print(f"\nPlayers whose best surface changed from early to later career: {changed}/{total} ({100*changed/total:.1f}%)")

    # Show examples
    print("\nExamples of players whose best surface changed:")
    changed_players = pivot[pivot['early'] != pivot['later']]
    for player in changed_players.head(10).index:
        print(f"  {player}: {pivot.loc[player, 'early']} -> {pivot.loc[player, 'later']}")

# ============================================================================
# Detailed analysis of the Big 3
# ============================================================================
print("\n" + "=" * 80)
print("THE BIG 3: SURFACE PERFORMANCE IN LATE CAREER (2020-2024)")
print("=" * 80)

big3 = ['Novak Djokovic', 'Rafael Nadal', 'Roger Federer']

for player_name in big3:
    player_data = player_matches[player_matches['player_name'] == player_name]

    if len(player_data) == 0:
        continue

    print(f"\n{'-'*60}")
    print(f"{player_name}")
    print(f"{'-'*60}")

    # Head to head by surface against top opponents
    player_wins = df[df['winner_name'] == player_name]
    player_losses = df[df['loser_name'] == player_name]

    print(f"\nWins by surface:")
    print(player_wins['surface'].value_counts())
    print(f"\nLosses by surface:")
    print(player_losses['surface'].value_counts())

    # Tournament level performance by surface
    print(f"\nGrand Slam matches:")
    gs_wins = player_wins[player_wins['tourney_level'] == 'G']
    gs_losses = player_losses[player_losses['tourney_level'] == 'G']
    print(f"  Wins: {len(gs_wins)}")
    if len(gs_wins) > 0:
        print(f"  By surface: {dict(gs_wins['surface'].value_counts())}")
    print(f"  Losses: {len(gs_losses)}")
    if len(gs_losses) > 0:
        print(f"  By surface: {dict(gs_losses['surface'].value_counts())}")

print("\n" + "=" * 80)
print("END OF EXTENDED ANALYSIS")
print("=" * 80)
