import pandas as pd
import numpy as np
from collections import defaultdict

# Load the data
df = pd.read_csv('/Users/matthewrussell/Documents/Github/tennis_surface_impact/data/atp_matches_combined.csv')

print("=" * 80)
print("TENNIS SURFACE PERFORMANCE OVER TIME ANALYSIS")
print("=" * 80)

# Basic data overview
print("\n1. DATA OVERVIEW")
print("-" * 40)
print(f"Total matches: {len(df):,}")
print(f"Date range: {df['tourney_date'].min()} to {df['tourney_date'].max()}")
print(f"\nSurfaces in dataset:")
print(df['surface'].value_counts())

# Convert tourney_date to datetime
df['year'] = df['tourney_date'].astype(str).str[:4].astype(int)

print(f"\nYear range: {df['year'].min()} to {df['year'].max()}")

# Create match records from both winner and loser perspectives
def create_player_match_records(df):
    """Create a dataframe with one row per player per match"""

    # Winner records
    winners = df[['winner_id', 'winner_name', 'winner_age', 'surface', 'year', 'tourney_date']].copy()
    winners.columns = ['player_id', 'player_name', 'age', 'surface', 'year', 'tourney_date']
    winners['won'] = 1

    # Loser records
    losers = df[['loser_id', 'loser_name', 'loser_age', 'surface', 'year', 'tourney_date']].copy()
    losers.columns = ['player_id', 'player_name', 'age', 'surface', 'year', 'tourney_date']
    losers['won'] = 0

    # Combine
    all_matches = pd.concat([winners, losers], ignore_index=True)

    # Remove rows with missing age
    all_matches = all_matches.dropna(subset=['age'])

    return all_matches

player_matches = create_player_match_records(df)
print(f"\nTotal player-match records: {len(player_matches):,}")
print(f"Unique players: {player_matches['player_id'].nunique():,}")

# Create age brackets
def age_bracket(age):
    if age < 20:
        return "Under 20"
    elif age < 23:
        return "20-22"
    elif age < 26:
        return "23-25"
    elif age < 29:
        return "26-28"
    elif age < 32:
        return "29-31"
    elif age < 35:
        return "32-34"
    else:
        return "35+"

player_matches['age_bracket'] = player_matches['age'].apply(age_bracket)

# ============================================================================
# QUESTION 1: Do players' win rates on specific surfaces change as they age?
# ============================================================================
print("\n" + "=" * 80)
print("QUESTION 1: WIN RATES BY SURFACE AND AGE")
print("=" * 80)

# Overall win rates by surface and age bracket
age_surface_stats = player_matches.groupby(['age_bracket', 'surface']).agg(
    matches=('won', 'count'),
    wins=('won', 'sum'),
    win_rate=('won', 'mean')
).round(3)

# Reorder age brackets
age_order = ["Under 20", "20-22", "23-25", "26-28", "29-31", "32-34", "35+"]
age_surface_stats = age_surface_stats.reset_index()
age_surface_stats['age_bracket'] = pd.Categorical(age_surface_stats['age_bracket'], categories=age_order, ordered=True)
age_surface_stats = age_surface_stats.sort_values(['age_bracket', 'surface'])

print("\nWin rates by age bracket and surface (all players):")
print("-" * 60)

# Pivot for better readability
pivot_table = age_surface_stats.pivot(index='age_bracket', columns='surface', values='win_rate')
print(pivot_table.round(3))

# Count table
count_pivot = age_surface_stats.pivot(index='age_bracket', columns='surface', values='matches')
print("\nMatch counts by age bracket and surface:")
print(count_pivot.astype(int))

# ============================================================================
# QUESTION 2: Surface specialization over time
# ============================================================================
print("\n" + "=" * 80)
print("QUESTION 2: SURFACE SPECIALIZATION ANALYSIS")
print("=" * 80)

def calculate_specialization(player_data):
    """
    Calculate surface specialization as the standard deviation of win rates across surfaces.
    Higher std = more specialized (big differences between surfaces)
    """
    surface_stats = player_data.groupby('surface').agg(
        matches=('won', 'count'),
        win_rate=('won', 'mean')
    )

    # Need at least some matches on multiple surfaces
    surfaces_with_enough = surface_stats[surface_stats['matches'] >= 10]

    if len(surfaces_with_enough) >= 2:
        return {
            'specialization_std': surfaces_with_enough['win_rate'].std(),
            'specialization_range': surfaces_with_enough['win_rate'].max() - surfaces_with_enough['win_rate'].min(),
            'n_surfaces': len(surfaces_with_enough),
            'best_surface': surfaces_with_enough['win_rate'].idxmax(),
            'best_win_rate': surfaces_with_enough['win_rate'].max(),
            'worst_surface': surfaces_with_enough['win_rate'].idxmin(),
            'worst_win_rate': surfaces_with_enough['win_rate'].min()
        }
    return None

# Calculate specialization by age bracket
print("\nSurface Specialization by Age Bracket")
print("(Specialization = std dev of win rates across surfaces)")
print("-" * 60)

spec_by_age = []
for age_bracket in age_order:
    age_data = player_matches[player_matches['age_bracket'] == age_bracket]

    # Get specialization for each player in this age bracket
    player_specs = []
    for player_id in age_data['player_id'].unique():
        player_age_data = age_data[age_data['player_id'] == player_id]
        spec = calculate_specialization(player_age_data)
        if spec:
            player_specs.append(spec)

    if player_specs:
        avg_spec_std = np.mean([s['specialization_std'] for s in player_specs])
        avg_spec_range = np.mean([s['specialization_range'] for s in player_specs])
        n_players = len(player_specs)
        spec_by_age.append({
            'age_bracket': age_bracket,
            'avg_specialization_std': avg_spec_std,
            'avg_specialization_range': avg_spec_range,
            'n_players': n_players
        })

spec_df = pd.DataFrame(spec_by_age)
print(spec_df.to_string(index=False))

# ============================================================================
# QUESTION 3: Peak ages by surface
# ============================================================================
print("\n" + "=" * 80)
print("QUESTION 3: PEAK PERFORMANCE AGES BY SURFACE")
print("=" * 80)

# Find the age bracket with highest win rate for each surface
print("\nBest performing age bracket by surface:")
print("-" * 60)

for surface in ['Hard', 'Clay', 'Grass']:
    surface_data = age_surface_stats[age_surface_stats['surface'] == surface]
    if not surface_data.empty:
        best_age = surface_data.loc[surface_data['win_rate'].idxmax()]
        print(f"{surface}: {best_age['age_bracket']} (win rate: {best_age['win_rate']:.3f}, n={int(best_age['matches'])})")

# More granular age analysis
print("\nDetailed win rates by specific age (minimum 500 matches per age):")
print("-" * 60)

player_matches['age_int'] = player_matches['age'].astype(int)
age_surface_detailed = player_matches.groupby(['age_int', 'surface']).agg(
    matches=('won', 'count'),
    win_rate=('won', 'mean')
).reset_index()

# Only ages with enough matches
for surface in ['Hard', 'Clay', 'Grass']:
    print(f"\n{surface}:")
    surface_ages = age_surface_detailed[
        (age_surface_detailed['surface'] == surface) &
        (age_surface_detailed['matches'] >= 500)
    ].sort_values('win_rate', ascending=False).head(5)
    for _, row in surface_ages.iterrows():
        print(f"  Age {int(row['age_int'])}: {row['win_rate']:.3f} (n={int(row['matches'])})")

# ============================================================================
# QUESTION 4: Individual Player Trajectories
# ============================================================================
print("\n" + "=" * 80)
print("QUESTION 4: INDIVIDUAL PLAYER TRAJECTORIES")
print("=" * 80)

# Identify notable players (most matches in dataset)
player_match_counts = player_matches.groupby(['player_id', 'player_name']).size().reset_index(name='total_matches')
top_players = player_match_counts.nlargest(20, 'total_matches')

print("\nTop 20 players by total matches:")
print(top_players.to_string(index=False))

# Analyze specific notable players
notable_players = ['Roger Federer', 'Rafael Nadal', 'Novak Djokovic', 'Andy Murray', 'Stan Wawrinka']

print("\n" + "=" * 80)
print("DETAILED ANALYSIS OF NOTABLE PLAYERS")
print("=" * 80)

for player_name in notable_players:
    player_data = player_matches[player_matches['player_name'] == player_name]

    if len(player_data) == 0:
        print(f"\n{player_name}: Not found in dataset")
        continue

    print(f"\n{'='*60}")
    print(f"{player_name.upper()}")
    print(f"{'='*60}")
    print(f"Total matches: {len(player_data)}")
    print(f"Age range: {player_data['age'].min():.1f} to {player_data['age'].max():.1f}")
    print(f"Years active (in data): {player_data['year'].min()} to {player_data['year'].max()}")

    # Overall win rates by surface
    print(f"\nOverall win rates by surface:")
    surface_overall = player_data.groupby('surface').agg(
        matches=('won', 'count'),
        wins=('won', 'sum'),
        win_rate=('won', 'mean')
    ).round(3)
    print(surface_overall)

    # Win rates by surface and age bracket
    print(f"\nWin rates by age bracket and surface:")
    player_age_surface = player_data.groupby(['age_bracket', 'surface']).agg(
        matches=('won', 'count'),
        win_rate=('won', 'mean')
    ).round(3)

    player_pivot = player_age_surface.reset_index()
    player_pivot['age_bracket'] = pd.Categorical(player_pivot['age_bracket'], categories=age_order, ordered=True)
    player_pivot = player_pivot.sort_values(['age_bracket', 'surface'])

    # Create pivot table
    win_rate_pivot = player_pivot.pivot(index='age_bracket', columns='surface', values='win_rate')
    matches_pivot = player_pivot.pivot(index='age_bracket', columns='surface', values='matches')

    print("\nWin rates:")
    print(win_rate_pivot.round(3).to_string())
    print("\nMatch counts:")
    print(matches_pivot.fillna(0).astype(int).to_string())

    # Specialization trend
    print(f"\nSurface specialization trend (std dev of win rates):")
    for age in age_order:
        age_data = player_data[player_data['age_bracket'] == age]
        if len(age_data) >= 20:
            spec = calculate_specialization(age_data)
            if spec:
                print(f"  {age}: {spec['specialization_std']:.3f} (best: {spec['best_surface']} {spec['best_win_rate']:.3f}, worst: {spec['worst_surface']} {spec['worst_win_rate']:.3f})")

# ============================================================================
# Additional Analysis: Surface preference changes over career
# ============================================================================
print("\n" + "=" * 80)
print("ADDITIONAL ANALYSIS: BEST SURFACE CHANGES OVER CAREER")
print("=" * 80)

# For players with long careers, see if their best surface changed
def analyze_career_surface_shifts(player_data, min_matches=30):
    """Analyze how a player's best surface changed over their career"""
    results = []

    for age_bracket in age_order:
        bracket_data = player_data[player_data['age_bracket'] == age_bracket]
        if len(bracket_data) >= min_matches:
            surface_rates = bracket_data.groupby('surface').agg(
                matches=('won', 'count'),
                win_rate=('won', 'mean')
            )
            # Only consider surfaces with enough matches
            valid_surfaces = surface_rates[surface_rates['matches'] >= 10]
            if len(valid_surfaces) > 0:
                best = valid_surfaces['win_rate'].idxmax()
                results.append({
                    'age_bracket': age_bracket,
                    'best_surface': best,
                    'win_rate': valid_surfaces.loc[best, 'win_rate'],
                    'total_matches': len(bracket_data)
                })

    return results

print("\nCareer surface preference evolution for notable players:")
print("-" * 60)

for player_name in notable_players:
    player_data = player_matches[player_matches['player_name'] == player_name]
    if len(player_data) > 0:
        career_shifts = analyze_career_surface_shifts(player_data)
        if career_shifts:
            print(f"\n{player_name}:")
            for period in career_shifts:
                print(f"  {period['age_bracket']}: Best on {period['best_surface']} ({period['win_rate']:.3f}, n={period['total_matches']})")

# ============================================================================
# Statistical summary
# ============================================================================
print("\n" + "=" * 80)
print("STATISTICAL SUMMARY")
print("=" * 80)

# Calculate correlation between age and win rate by surface
print("\nCorrelation between age and win rate by surface:")
print("-" * 60)

for surface in ['Hard', 'Clay', 'Grass']:
    surface_data = player_matches[player_matches['surface'] == surface].copy()

    # Group by age (integer) for cleaner analysis
    age_win_rates = surface_data.groupby('age_int').agg(
        win_rate=('won', 'mean'),
        matches=('won', 'count')
    )

    # Only use ages with sufficient data
    age_win_rates = age_win_rates[age_win_rates['matches'] >= 100]

    if len(age_win_rates) > 5:
        correlation = age_win_rates.index.to_series().corr(age_win_rates['win_rate'])
        print(f"{surface}: {correlation:.3f}")

# Calculate which surface shows the steepest decline with age
print("\nWin rate change from peak age bracket (26-28) to 35+:")
print("-" * 60)

for surface in ['Hard', 'Clay', 'Grass']:
    surface_stats = age_surface_stats[age_surface_stats['surface'] == surface]

    peak = surface_stats[surface_stats['age_bracket'] == '26-28']['win_rate'].values
    older = surface_stats[surface_stats['age_bracket'] == '35+']['win_rate'].values

    if len(peak) > 0 and len(older) > 0:
        change = older[0] - peak[0]
        print(f"{surface}: {change:+.3f} ({peak[0]:.3f} -> {older[0]:.3f})")

# Young player analysis
print("\nWin rate change from Under 20 to peak (26-28):")
print("-" * 60)

for surface in ['Hard', 'Clay', 'Grass']:
    surface_stats = age_surface_stats[age_surface_stats['surface'] == surface]

    young = surface_stats[surface_stats['age_bracket'] == 'Under 20']['win_rate'].values
    peak = surface_stats[surface_stats['age_bracket'] == '26-28']['win_rate'].values

    if len(young) > 0 and len(peak) > 0:
        change = peak[0] - young[0]
        print(f"{surface}: {change:+.3f} ({young[0]:.3f} -> {peak[0]:.3f})")

print("\n" + "=" * 80)
print("END OF ANALYSIS")
print("=" * 80)
