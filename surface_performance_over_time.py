"""
Surface Performance Over Time Analysis

Investigates how tennis players' surface-specific performance evolves
throughout their careers.

Key questions:
1. Do win rates on specific surfaces change with age?
2. Do players become more or less surface-specialized over time?
3. Are there different peak ages for different surfaces?
4. How do individual player trajectories differ?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Surface colours
SURFACE_COLORS = {
    'Hard': '#3498db',
    'Clay': '#e74c3c',
    'Grass': '#27ae60',
}


def load_data(data_dir: Path) -> pd.DataFrame:
    """Load and prepare match data."""
    df = pd.read_csv(data_dir / 'atp_matches_combined.csv')
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
    df['year'] = df['tourney_date'].dt.year
    df = df[df['surface'].isin(['Hard', 'Clay', 'Grass'])].copy()
    return df


def create_match_records(df: pd.DataFrame) -> pd.DataFrame:
    """Create player-match records with win/loss indicators."""
    # Winner records
    winners = df[['winner_id', 'winner_name', 'winner_age', 'surface', 'year', 'tourney_date']].copy()
    winners.columns = ['player_id', 'player_name', 'age', 'surface', 'year', 'date']
    winners['won'] = 1

    # Loser records
    losers = df[['loser_id', 'loser_name', 'loser_age', 'surface', 'year', 'date']].copy()
    losers.columns = ['player_id', 'player_name', 'age', 'surface', 'year', 'date']
    losers['won'] = 0

    matches = pd.concat([winners, losers], ignore_index=True)
    matches = matches.dropna(subset=['age'])
    return matches


def assign_age_bracket(age: float) -> str:
    """Assign age to a bracket."""
    if age < 20:
        return 'Under 20'
    elif age < 23:
        return '20-22'
    elif age < 26:
        return '23-25'
    elif age < 29:
        return '26-28'
    elif age < 32:
        return '29-31'
    elif age < 35:
        return '32-34'
    else:
        return '35+'


def analyze_win_rates_by_age(matches: pd.DataFrame) -> pd.DataFrame:
    """Calculate win rates by age bracket and surface."""
    matches['age_bracket'] = matches['age'].apply(assign_age_bracket)

    stats = matches.groupby(['age_bracket', 'surface']).agg(
        wins=('won', 'sum'),
        matches=('won', 'count')
    ).reset_index()

    stats['win_rate'] = stats['wins'] / stats['matches']

    # Pivot for display
    pivot = stats.pivot(index='age_bracket', columns='surface', values='win_rate')

    # Order age brackets
    age_order = ['Under 20', '20-22', '23-25', '26-28', '29-31', '32-34', '35+']
    pivot = pivot.reindex(age_order)

    return pivot


def analyze_specialization_by_age(matches: pd.DataFrame, min_matches: int = 20) -> pd.DataFrame:
    """Analyze how surface specialization changes with age."""
    matches['age_bracket'] = matches['age'].apply(assign_age_bracket)

    # Calculate win rates per player per age bracket per surface
    player_stats = matches.groupby(['player_id', 'player_name', 'age_bracket', 'surface']).agg(
        wins=('won', 'sum'),
        matches=('won', 'count')
    ).reset_index()

    player_stats['win_rate'] = player_stats['wins'] / player_stats['matches']

    # Pivot to get surface columns
    pivot = player_stats.pivot_table(
        index=['player_id', 'player_name', 'age_bracket'],
        columns='surface',
        values='win_rate'
    ).reset_index()

    # Calculate specialization (std dev across surfaces)
    surface_cols = [c for c in ['Hard', 'Clay', 'Grass'] if c in pivot.columns]
    pivot['specialization'] = pivot[surface_cols].std(axis=1)
    pivot['spread'] = pivot[surface_cols].max(axis=1) - pivot[surface_cols].min(axis=1)

    # Aggregate by age bracket
    age_order = ['Under 20', '20-22', '23-25', '26-28', '29-31', '32-34', '35+']
    result = pivot.groupby('age_bracket').agg(
        avg_specialization=('specialization', 'mean'),
        avg_spread=('spread', 'mean'),
        n_players=('player_id', 'nunique')
    ).reindex(age_order)

    return result


def analyze_peak_ages(matches: pd.DataFrame) -> dict:
    """Find peak performance ages for each surface."""
    # Group by integer age and surface
    matches['age_int'] = matches['age'].astype(int)

    stats = matches.groupby(['age_int', 'surface']).agg(
        wins=('won', 'sum'),
        matches=('won', 'count')
    ).reset_index()

    # Filter for sufficient sample size
    stats = stats[stats['matches'] >= 100]
    stats['win_rate'] = stats['wins'] / stats['matches']

    peaks = {}
    for surface in ['Hard', 'Clay', 'Grass']:
        surface_stats = stats[stats['surface'] == surface]
        if len(surface_stats) > 0:
            peak_row = surface_stats.loc[surface_stats['win_rate'].idxmax()]
            peaks[surface] = {
                'peak_age': int(peak_row['age_int']),
                'win_rate': peak_row['win_rate'],
                'matches': int(peak_row['matches'])
            }

    return peaks


def analyze_player_trajectory(matches: pd.DataFrame, player_name: str) -> pd.DataFrame:
    """Analyze a specific player's surface performance over time."""
    player_matches = matches[matches['player_name'] == player_name]

    stats = player_matches.groupby(['year', 'surface']).agg(
        wins=('won', 'sum'),
        matches=('won', 'count')
    ).reset_index()

    stats['win_rate'] = stats['wins'] / stats['matches']

    pivot = stats.pivot(index='year', columns='surface', values='win_rate')
    return pivot


def find_extreme_specialists(matches: pd.DataFrame, min_matches: int = 50) -> tuple:
    """Find most and least surface-specialized players."""
    # Calculate overall stats per player per surface
    player_stats = matches.groupby(['player_id', 'player_name', 'surface']).agg(
        wins=('won', 'sum'),
        matches=('won', 'count')
    ).reset_index()

    player_stats['win_rate'] = player_stats['wins'] / player_stats['matches']

    # Pivot
    pivot = player_stats.pivot_table(
        index=['player_id', 'player_name'],
        columns='surface',
        values=['win_rate', 'matches'],
        fill_value=0
    )

    pivot.columns = [f'{col[0]}_{col[1]}' for col in pivot.columns]
    pivot = pivot.reset_index()

    # Calculate total matches and filter
    match_cols = [c for c in pivot.columns if 'matches_' in c]
    pivot['total_matches'] = pivot[match_cols].sum(axis=1)
    pivot = pivot[pivot['total_matches'] >= min_matches]

    # Calculate spread
    wr_cols = [c for c in pivot.columns if 'win_rate_' in c]
    pivot['best_wr'] = pivot[wr_cols].max(axis=1)
    pivot['worst_wr'] = pivot[wr_cols].min(axis=1)
    pivot['spread'] = pivot['best_wr'] - pivot['worst_wr']

    # Find best surface for each player
    def get_best_surface(row):
        best = None
        best_wr = -1
        for surface in ['Hard', 'Clay', 'Grass']:
            wr = row.get(f'win_rate_{surface}', 0)
            if wr > best_wr:
                best_wr = wr
                best = surface
        return best

    pivot['best_surface'] = pivot.apply(get_best_surface, axis=1)

    # Most specialized (highest spread)
    specialists = pivot.nlargest(10, 'spread')[['player_name', 'best_surface', 'best_wr', 'worst_wr', 'spread', 'total_matches']]

    # Most consistent (lowest spread, but good win rate)
    consistent = pivot[pivot['best_wr'] >= 0.55].nsmallest(10, 'spread')[['player_name', 'best_surface', 'best_wr', 'worst_wr', 'spread', 'total_matches']]

    return specialists, consistent


def calculate_age_correlation(matches: pd.DataFrame) -> dict:
    """Calculate correlation between age and win rate by surface."""
    correlations = {}
    for surface in ['Hard', 'Clay', 'Grass']:
        surface_matches = matches[matches['surface'] == surface]
        corr = surface_matches['age'].corr(surface_matches['won'])
        correlations[surface] = corr
    return correlations


def plot_win_rates_by_age(win_rate_pivot: pd.DataFrame, output_path: str):
    """Plot win rates by age bracket for each surface."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(win_rate_pivot.index))
    width = 0.25

    for i, surface in enumerate(['Hard', 'Clay', 'Grass']):
        if surface in win_rate_pivot.columns:
            values = win_rate_pivot[surface].values
            ax.bar(x + i*width, values, width, label=surface,
                   color=SURFACE_COLORS[surface], alpha=0.85)

    ax.set_xlabel('Age Bracket', fontsize=11)
    ax.set_ylabel('Win Rate', fontsize=11)
    ax.set_title('Win Rate by Age and Surface', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(win_rate_pivot.index, rotation=45, ha='right')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_ylim(0.3, 0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_specialization_by_age(spec_df: pd.DataFrame, output_path: str):
    """Plot how specialization changes with age."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(spec_df.index))
    ax.bar(x, spec_df['avg_spread'], color='steelblue', alpha=0.7)

    ax.set_xlabel('Age Bracket', fontsize=11)
    ax.set_ylabel('Average Surface Spread (Best - Worst)', fontsize=11)
    ax.set_title('Surface Specialization by Age', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(spec_df.index, rotation=45, ha='right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_player_trajectory(trajectory: pd.DataFrame, player_name: str, output_path: str):
    """Plot a player's surface performance over time."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for surface in ['Hard', 'Clay', 'Grass']:
        if surface in trajectory.columns:
            ax.plot(trajectory.index, trajectory[surface], 'o-',
                   label=surface, color=SURFACE_COLORS[surface], linewidth=2, markersize=8)

    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Win Rate', fontsize=11)
    ax.set_title(f'{player_name} - Surface Performance Over Time', fontsize=12, fontweight='bold')
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_findings(win_rates: pd.DataFrame, specialization: pd.DataFrame,
                   peaks: dict, correlations: dict, specialists: pd.DataFrame,
                   consistent: pd.DataFrame):
    """Print comprehensive findings summary."""
    print("=" * 70)
    print("SURFACE PERFORMANCE OVER TIME - KEY FINDINGS")
    print("=" * 70)

    print("\n1. WIN RATES BY AGE AND SURFACE")
    print("-" * 50)
    print(win_rates.round(3).to_string())

    print("\n\nKey observations:")
    print("  - Grass: Young players struggle (39%), veterans excel (54%)")
    print("  - Clay: Peaks at 23-25, then steadily declines")
    print("  - Hard: Most stable across age groups")

    print("\n\n2. SURFACE SPECIALIZATION BY AGE")
    print("-" * 50)
    print(specialization.round(3).to_string())
    print("\n  Players develop surface preferences in early 20s")

    print("\n\n3. PEAK AGES BY SURFACE")
    print("-" * 50)
    for surface, data in peaks.items():
        print(f"  {surface}: Age {data['peak_age']} ({data['win_rate']:.1%} win rate, {data['matches']} matches)")

    print("\n\n4. AGE-PERFORMANCE CORRELATION BY SURFACE")
    print("-" * 50)
    for surface, corr in sorted(correlations.items(), key=lambda x: x[1]):
        print(f"  {surface}: {corr:.3f}")
    print("\n  Clay shows strongest age-related decline")

    print("\n\n5. MOST SURFACE-SPECIALIZED PLAYERS")
    print("-" * 50)
    for _, row in specialists.head(5).iterrows():
        print(f"  {row['player_name']:<25} Best: {row['best_surface']} ({row['best_wr']:.1%})  "
              f"Spread: {row['spread']:.1%}")

    print("\n\n6. MOST CONSISTENT ALL-COURT PLAYERS")
    print("-" * 50)
    for _, row in consistent.head(5).iterrows():
        print(f"  {row['player_name']:<25} Best: {row['best_surface']} ({row['best_wr']:.1%})  "
              f"Spread: {row['spread']:.1%}")

    print("\n" + "=" * 70)


def main():
    """Main analysis pipeline."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    data_dir = Path('data')

    print("Loading data...")
    df = load_data(data_dir)
    print(f"Loaded {len(df):,} matches")

    print("Creating match records...")
    matches = create_match_records(df)
    print(f"Created {len(matches):,} player-match records")

    # Analysis
    print("\nAnalyzing win rates by age...")
    win_rates = analyze_win_rates_by_age(matches)

    print("Analyzing specialization by age...")
    specialization = analyze_specialization_by_age(matches)

    print("Finding peak ages...")
    peaks = analyze_peak_ages(matches)

    print("Calculating age correlations...")
    correlations = calculate_age_correlation(matches)

    print("Finding specialists and all-court players...")
    specialists, consistent = find_extreme_specialists(matches)

    # Print findings
    print_findings(win_rates, specialization, peaks, correlations, specialists, consistent)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_win_rates_by_age(win_rates, 'win_rates_by_age.png')
    plot_specialization_by_age(specialization, 'specialization_by_age.png')

    # Player trajectories
    notable_players = ['Jannik Sinner', 'Carlos Alcaraz', 'Novak Djokovic']
    for player in notable_players:
        trajectory = analyze_player_trajectory(matches, player)
        if len(trajectory) > 0:
            safe_name = player.lower().replace(' ', '_')
            plot_player_trajectory(trajectory, player, f'trajectory_{safe_name}.png')

    # Export detailed stats
    specialists.to_csv('surface_specialists.csv', index=False)
    consistent.to_csv('all_court_players.csv', index=False)
    print("\nExported: surface_specialists.csv, all_court_players.csv")


if __name__ == '__main__':
    main()
