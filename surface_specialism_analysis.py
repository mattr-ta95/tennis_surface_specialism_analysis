"""
Quantifying Surface Specialism in Men's Tennis

Author: Matt (Portfolio Project)
Date: January 2026
Data Source: Jeff Sackmann's tennis_atp (https://github.com/JeffSackmann/tennis_atp)

This analysis introduces a Surface Dependency Index (SDI) to quantify how much
a player's performance varies across surfaces.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Surface colours (consistent throughout)
SURFACE_COLORS = {
    'Hard': '#3498db',   # Blue
    'Clay': '#e74c3c',   # Orange/Red
    'Grass': '#27ae60',  # Green
}


def load_data(data_dir: Path) -> pd.DataFrame:
    """Load ATP match data from available sources."""
    if (data_dir / 'atp_matches_combined.csv').exists():
        df = pd.read_csv(data_dir / 'atp_matches_combined.csv')
        print("Loaded combined ATP matches dataset")
    elif (data_dir / 'sample_atp_matches.csv').exists():
        df = pd.read_csv(data_dir / 'sample_atp_matches.csv')
        print("Loaded sample dataset (run download_data.py for full data)")
    else:
        # Load individual year files
        dfs = []
        for year in range(2020, 2026):
            fp = data_dir / f'atp_matches_{year}.csv'
            if fp.exists():
                dfs.append(pd.read_csv(fp))
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            print(f"Loaded {len(dfs)} year files")
        else:
            raise FileNotFoundError("No data found! Run download_data.py first.")

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess match data: convert dates and filter surfaces."""
    df = df.copy()
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
    df['year'] = df['tourney_date'].dt.year

    # Filter to main surfaces only (exclude carpet, which is rare)
    main_surfaces = ['Hard', 'Clay', 'Grass']
    df = df[df['surface'].isin(main_surfaces)].copy()

    return df


def calculate_player_surface_stats(df: pd.DataFrame, min_matches: int = 30) -> pd.DataFrame:
    """
    Calculate win rates by surface for each player.

    Args:
        df: Match dataframe with winner_id, loser_id, surface columns
        min_matches: Minimum total matches required for inclusion

    Returns:
        DataFrame with player surface stats
    """
    # Count wins and losses per player per surface
    wins = df.groupby(['winner_id', 'winner_name', 'surface']).size().reset_index(name='wins')
    wins.columns = ['player_id', 'player_name', 'surface', 'wins']

    losses = df.groupby(['loser_id', 'loser_name', 'surface']).size().reset_index(name='losses')
    losses.columns = ['player_id', 'player_name', 'surface', 'losses']

    # Merge wins and losses
    stats = pd.merge(wins, losses, on=['player_id', 'player_name', 'surface'], how='outer')
    stats = stats.fillna(0)

    # Calculate totals and win rate
    stats['matches'] = stats['wins'] + stats['losses']
    stats['win_rate'] = stats['wins'] / stats['matches']

    # Pivot to get surface columns
    pivot = stats.pivot_table(
        index=['player_id', 'player_name'],
        columns='surface',
        values=['win_rate', 'matches'],
        fill_value=0
    )

    # Flatten column names
    pivot.columns = [f'{col[0]}_{col[1]}' for col in pivot.columns]
    pivot = pivot.reset_index()

    # Calculate total matches
    pivot['total_matches'] = pivot[[c for c in pivot.columns if 'matches_' in c]].sum(axis=1)

    # Filter by minimum matches
    pivot = pivot[pivot['total_matches'] >= min_matches].copy()

    return pivot


def calculate_sdi(row: pd.Series, min_surface_matches: int = 5) -> float:
    """
    Calculate Surface Dependency Index for a player.

    Only considers surfaces where player has minimum required matches.
    Returns NaN if player has data on fewer than 2 surfaces.
    """
    surfaces = ['Hard', 'Clay', 'Grass']
    win_rates = []

    for surface in surfaces:
        matches = row.get(f'matches_{surface}', 0)
        if matches >= min_surface_matches:
            win_rates.append(row.get(f'win_rate_{surface}', np.nan))

    if len(win_rates) < 2:
        return np.nan

    return np.std(win_rates)


def identify_best_surface(row: pd.Series) -> str:
    """Identify player's best surface based on win rate."""
    surfaces = ['Hard', 'Clay', 'Grass']
    win_rates = {s: row.get(f'win_rate_{s}', 0) for s in surfaces}
    best = max(win_rates, key=win_rates.get)
    return best


def calculate_surface_advantage(row: pd.Series) -> float:
    """Calculate how much better player is on best surface vs worst."""
    surfaces = ['Hard', 'Clay', 'Grass']
    win_rates = [row.get(f'win_rate_{s}', 0) for s in surfaces if row.get(f'matches_{s}', 0) >= 5]
    if len(win_rates) < 2:
        return 0
    return max(win_rates) - min(win_rates)


def calculate_yearly_sdi(df: pd.DataFrame, year: int, min_matches: int = 15) -> pd.DataFrame:
    """Calculate SDI for players in a specific year."""
    year_df = df[df['year'] == year]
    stats = calculate_player_surface_stats(year_df, min_matches=min_matches)
    stats['SDI'] = stats.apply(lambda r: calculate_sdi(r, min_surface_matches=3), axis=1)
    stats['year'] = year
    return stats.dropna(subset=['SDI'])


def plot_surface_comparison(ax, data: pd.DataFrame, title: str):
    """Plot grouped bar chart comparing win rates across surfaces."""
    players = data['player_name'].values
    x = np.arange(len(players))
    width = 0.25

    for i, surface in enumerate(['Hard', 'Clay', 'Grass']):
        values = data[f'win_rate_{surface}'].values
        ax.bar(x + i*width, values, width,
               label=surface, color=SURFACE_COLORS[surface], alpha=0.85)

    ax.set_xlabel('')
    ax.set_ylabel('Win Rate', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([p.split()[-1] for p in players], rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))


def create_specialist_comparison_plot(specialists: pd.DataFrame, all_court: pd.DataFrame, output_path: str):
    """Create side-by-side comparison of specialists vs all-court players."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_surface_comparison(axes[0], specialists, 'Surface Specialists (High SDI)')
    plot_surface_comparison(axes[1], all_court, 'All-Court Players (Low SDI)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_sdi_analysis_plot(player_stats: pd.DataFrame, output_path: str):
    """Create SDI distribution and scatter plot analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # SDI Distribution
    ax1 = axes[0]
    ax1.hist(player_stats['SDI'], bins=25, edgecolor='white', alpha=0.7, color='steelblue')
    ax1.axvline(player_stats['SDI'].median(), color='red', linestyle='--',
                label=f'Median: {player_stats["SDI"].median():.3f}')
    ax1.axvline(player_stats['SDI'].mean(), color='orange', linestyle='--',
                label=f'Mean: {player_stats["SDI"].mean():.3f}')
    ax1.set_xlabel('Surface Dependency Index (SDI)', fontsize=11)
    ax1.set_ylabel('Number of Players', fontsize=11)
    ax1.set_title('Distribution of Surface Dependency', fontsize=12, fontweight='bold')
    ax1.legend()

    # SDI vs Overall Win Rate
    ax2 = axes[1]
    scatter = ax2.scatter(
        player_stats['overall_win_rate'],
        player_stats['SDI'],
        c=player_stats['total_matches'],
        cmap='viridis',
        alpha=0.6,
        s=50
    )
    ax2.set_xlabel('Overall Win Rate', fontsize=11)
    ax2.set_ylabel('Surface Dependency Index (SDI)', fontsize=11)
    ax2.set_title('SDI vs Overall Performance', fontsize=12, fontweight='bold')
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Total Matches', fontsize=10)

    # Annotate notable players
    top_players = player_stats[player_stats['overall_win_rate'] > 0.7]
    for _, row in top_players.iterrows():
        ax2.annotate(row['player_name'].split()[-1],
                    (row['overall_win_rate'], row['SDI']),
                    fontsize=8, alpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_surface_preferences_plot(player_stats: pd.DataFrame, surface_counts: pd.Series, output_path: str):
    """Create surface preference distribution plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart of surface preferences
    colors = [SURFACE_COLORS[s] for s in surface_counts.index]
    axes[0].pie(surface_counts.values, labels=surface_counts.index, colors=colors,
                autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Distribution of Best Surfaces', fontsize=12, fontweight='bold')

    # Surface advantage distribution
    for surface in ['Hard', 'Clay', 'Grass']:
        subset = player_stats[player_stats['best_surface'] == surface]
        axes[1].hist(subset['surface_advantage'], bins=15, alpha=0.5,
                     label=f'{surface} ({len(subset)})', color=SURFACE_COLORS[surface])

    axes[1].set_xlabel('Surface Advantage (Best - Worst Win Rate)', fontsize=11)
    axes[1].set_ylabel('Number of Players', fontsize=11)
    axes[1].set_title('Surface Advantage by Best Surface', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_trend_plot(df: pd.DataFrame, output_path: str) -> pd.DataFrame: # | None:
    """Create temporal trend analysis plot."""
    years = sorted(df['year'].dropna().unique())
    yearly_stats = []

    for year in years:
        try:
            stats = calculate_yearly_sdi(df, int(year), min_matches=15)
            if len(stats) > 10:
                yearly_stats.append({
                    'year': int(year),
                    'median_sdi': stats['SDI'].median(),
                    'mean_sdi': stats['SDI'].mean(),
                    'std_sdi': stats['SDI'].std(),
                    'n_players': len(stats)
                })
        except Exception:
            continue

    if not yearly_stats:
        print("Not enough yearly data for trend analysis")
        return None

    trend_df = pd.DataFrame(yearly_stats)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(trend_df['year'], trend_df['median_sdi'], 'o-',
            color='steelblue', linewidth=2, markersize=8, label='Median SDI')
    ax.fill_between(trend_df['year'],
                    trend_df['median_sdi'] - trend_df['std_sdi'],
                    trend_df['median_sdi'] + trend_df['std_sdi'],
                    alpha=0.2, color='steelblue')

    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Surface Dependency Index', fontsize=11)
    ax.set_title('Surface Specialism Over Time', fontsize=12, fontweight='bold')

    # Add trend line
    z = np.polyfit(trend_df['year'], trend_df['median_sdi'], 1)
    p = np.poly1d(z)
    ax.plot(trend_df['year'], p(trend_df['year']), '--', color='red', alpha=0.7,
           label=f'Trend (slope: {z[0]:.4f})')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    return trend_df


def print_findings(df: pd.DataFrame, player_stats: pd.DataFrame, surface_counts: pd.Series):
    """Print key findings summary."""
    print("=" * 70)
    print("SURFACE SPECIALISM ANALYSIS - KEY FINDINGS")
    print("=" * 70)

    print(f"\nDataset: {len(df):,} matches, {player_stats['player_name'].nunique()} players analysed")

    print(f"\nSDI Statistics:")
    print(f"   Mean SDI:   {player_stats['SDI'].mean():.3f}")
    print(f"   Median SDI: {player_stats['SDI'].median():.3f}")
    print(f"   Std Dev:    {player_stats['SDI'].std():.3f}")

    print(f"\nMost Surface-Dependent Player:")
    top = player_stats.nlargest(1, 'SDI').iloc[0]
    print(f"   {top['player_name']} (SDI: {top['SDI']:.3f})")
    print(f"   Hard: {top['win_rate_Hard']:.1%} | Clay: {top['win_rate_Clay']:.1%} | Grass: {top['win_rate_Grass']:.1%}")

    print(f"\nMost Consistent All-Court Player:")
    consistent = player_stats[player_stats['overall_win_rate'] >= 0.6].nsmallest(1, 'SDI')
    if len(consistent) > 0:
        c = consistent.iloc[0]
        print(f"   {c['player_name']} (SDI: {c['SDI']:.3f})")
        print(f"   Hard: {c['win_rate_Hard']:.1%} | Clay: {c['win_rate_Clay']:.1%} | Grass: {c['win_rate_Grass']:.1%}")

    print(f"\nSurface Preference Distribution:")
    for surface, count in surface_counts.items():
        pct = count / len(player_stats) * 100
        print(f"   {surface}: {count} players ({pct:.1f}%)")

    print("\n" + "=" * 70)


def export_results(player_stats: pd.DataFrame, output_path: str):
    """Export player analysis results to CSV."""
    output_cols = [
        'player_id', 'player_name', 'total_matches',
        'win_rate_Hard', 'win_rate_Clay', 'win_rate_Grass',
        'matches_Hard', 'matches_Clay', 'matches_Grass',
        'overall_win_rate', 'SDI', 'best_surface', 'surface_advantage'
    ]

    export_df = player_stats[output_cols].sort_values('SDI', ascending=False)
    export_df.to_csv(output_path, index=False)
    print(f"Exported player analysis to: {output_path}")
    print(f"Total players: {len(export_df)}")


def main():
    """Main analysis pipeline."""
    # Setup
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    data_dir = Path('data')

    # Load and preprocess data
    print("Loading data...")
    df = load_data(data_dir)
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['tourney_date'].min()} to {df['tourney_date'].max()}")
    print(f"\nSurface distribution:")
    print(df['surface'].value_counts())

    df = preprocess_data(df)
    print(f"\nMatches after filtering: {len(df):,}")
    print(f"Unique winners: {df['winner_name'].nunique():,}")

    # Calculate player stats
    print("\nCalculating player surface statistics...")
    player_stats = calculate_player_surface_stats(df, min_matches=30)
    print(f"Players with 30+ matches: {len(player_stats)}")

    # Calculate SDI
    player_stats['SDI'] = player_stats.apply(calculate_sdi, axis=1)
    player_stats['overall_win_rate'] = (
        player_stats['win_rate_Hard'] * player_stats['matches_Hard'] +
        player_stats['win_rate_Clay'] * player_stats['matches_Clay'] +
        player_stats['win_rate_Grass'] * player_stats['matches_Grass']
    ) / player_stats['total_matches']

    player_stats_valid = player_stats.dropna(subset=['SDI']).copy()
    print(f"Players with valid SDI: {len(player_stats_valid)}")

    # Identify specialists and all-court players
    specialists = player_stats_valid.nlargest(10, 'SDI').copy()
    all_court = player_stats_valid[player_stats_valid['overall_win_rate'] >= 0.55].nsmallest(10, 'SDI').copy()

    print("\n" + "=" * 60)
    print("TOP 10 SURFACE SPECIALISTS (Highest SDI)")
    print("=" * 60)
    for _, row in specialists.iterrows():
        print(f"{row['player_name']:<25} SDI: {row['SDI']:.3f}  "
              f"Hard: {row['win_rate_Hard']:.1%}  "
              f"Clay: {row['win_rate_Clay']:.1%}  "
              f"Grass: {row['win_rate_Grass']:.1%}")

    print("\n" + "=" * 60)
    print("TOP 10 ALL-COURT PLAYERS (Lowest SDI, 55%+ overall)")
    print("=" * 60)
    for _, row in all_court.iterrows():
        print(f"{row['player_name']:<25} SDI: {row['SDI']:.3f}  "
              f"Hard: {row['win_rate_Hard']:.1%}  "
              f"Clay: {row['win_rate_Clay']:.1%}  "
              f"Grass: {row['win_rate_Grass']:.1%}")

    # Calculate best surface and surface advantage
    player_stats_valid['best_surface'] = player_stats_valid.apply(identify_best_surface, axis=1)
    player_stats_valid['surface_advantage'] = player_stats_valid.apply(calculate_surface_advantage, axis=1)
    surface_counts = player_stats_valid['best_surface'].value_counts()

    # Create visualizations
    print("\nGenerating visualizations...")
    create_specialist_comparison_plot(specialists, all_court, 'surface_comparison.png')
    create_sdi_analysis_plot(player_stats_valid, 'sdi_analysis.png')
    create_surface_preferences_plot(player_stats_valid, surface_counts, 'surface_preferences.png')
    trend_df = create_trend_plot(df, 'sdi_trend.png')

    if trend_df is not None:
        print("\nYearly Summary:")
        print(trend_df.to_string(index=False))

    # Print findings
    print_findings(df, player_stats_valid, surface_counts)

    # Export results
    export_results(player_stats_valid, 'player_surface_analysis.csv')


if __name__ == '__main__':
    main()
