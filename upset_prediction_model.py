"""
Predicting Upsets in Men's Tennis

Author: Matt (Portfolio Project)
Date: January 2026
Data Source: Jeff Sackmann's tennis_atp

This model predicts upset probability (lower-ranked player defeating higher-ranked
opponent) using ranking, surface-specific performance, and recent form features.

IMPORTANT: This version fixes data leakage issues present in the original notebook:
- Uses temporal train-test split (train on 2020-2023, test on 2024)
- Computes surface win rates using only historical data (before each match)
- Adds rolling form features as promised in documentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple

from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report
)

SURFACE_COLORS = {
    'Hard': '#3498db',
    'Clay': '#e74c3c',
    'Grass': '#27ae60',
}


def load_and_prepare_data(data_dir: Path) -> pd.DataFrame:
    """Load and prepare match data."""
    if (data_dir / 'atp_matches_combined.csv').exists():
        df = pd.read_csv(data_dir / 'atp_matches_combined.csv')
    else:
        raise FileNotFoundError("No data found! Run download_data.py first.")

    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
    df['year'] = df['tourney_date'].dt.year

    # Filter to main surfaces and valid rankings
    df = df[df['surface'].isin(['Hard', 'Clay', 'Grass'])].copy()
    df = df.dropna(subset=['winner_rank', 'loser_rank', 'tourney_date'])

    # Sort by date for temporal calculations
    df = df.sort_values('tourney_date').reset_index(drop=True)

    return df


def calculate_historical_stats(df: pd.DataFrame) -> Tuple[dict, dict]:
    """
    Calculate cumulative surface win rates and rolling form for each player.

    CRITICAL: Only uses matches BEFORE each date to avoid data leakage.

    Returns:
        surface_wr_lookup: {(player_id, surface, date): win_rate}
        form_lookup: {(player_id, date): rolling_win_rate}
    """
    print("Calculating historical stats (this may take a moment)...")

    # Create player-match records
    records = []
    for _, row in df.iterrows():
        date = row['tourney_date']
        surface = row['surface']

        records.append({
            'player_id': row['winner_id'],
            'date': date,
            'surface': surface,
            'won': 1
        })
        records.append({
            'player_id': row['loser_id'],
            'date': date,
            'surface': surface,
            'won': 0
        })

    records_df = pd.DataFrame(records).sort_values('date')

    # Calculate cumulative surface win rates
    surface_wr_lookup = {}
    form_lookup = {}

    # Group by player
    for player_id in records_df['player_id'].unique():
        player_data = records_df[records_df['player_id'] == player_id].copy()

        # Rolling form (last 10 matches)
        player_data['rolling_form'] = player_data['won'].rolling(
            window=10, min_periods=3
        ).mean().shift(1)  # Shift to exclude current match

        for _, row in player_data.iterrows():
            form_lookup[(player_id, row['date'])] = row.get('rolling_form', 0.5)

        # Cumulative surface win rates
        for surface in ['Hard', 'Clay', 'Grass']:
            surface_data = player_data[player_data['surface'] == surface]
            if len(surface_data) == 0:
                continue

            surface_data = surface_data.copy()
            surface_data['cum_wins'] = surface_data['won'].cumsum().shift(1)
            surface_data['cum_matches'] = range(1, len(surface_data) + 1)
            surface_data['cum_matches'] = surface_data['cum_matches'] - 1  # Shift
            surface_data['surface_wr'] = np.where(
                surface_data['cum_matches'] >= 5,
                surface_data['cum_wins'] / surface_data['cum_matches'],
                0.5  # Default if insufficient history
            )

            for _, row in surface_data.iterrows():
                surface_wr_lookup[(player_id, surface, row['date'])] = row['surface_wr']

    print(f"  Calculated stats for {records_df['player_id'].nunique()} players")
    return surface_wr_lookup, form_lookup


def build_features(df: pd.DataFrame, surface_wr_lookup: dict,
                   form_lookup: dict) -> pd.DataFrame:
    """
    Build feature matrix for upset prediction.

    For each match:
    - Favourite = higher-ranked player (lower rank number)
    - Underdog = lower-ranked player (higher rank number)
    - Target: Did underdog win? (1 = upset)
    """
    print("Building feature matrix...")

    features = []

    for idx, row in df.iterrows():
        date = row['tourney_date']
        surface = row['surface']

        # Determine favourite and underdog
        if row['winner_rank'] <= row['loser_rank']:
            fav_id, fav_rank = row['winner_id'], row['winner_rank']
            dog_id, dog_rank = row['loser_id'], row['loser_rank']
            upset = 0
        else:
            fav_id, fav_rank = row['loser_id'], row['loser_rank']
            dog_id, dog_rank = row['winner_id'], row['winner_rank']
            upset = 1

        # Get historical surface win rates (BEFORE this match)
        fav_surface_wr = surface_wr_lookup.get((fav_id, surface, date), 0.5)
        dog_surface_wr = surface_wr_lookup.get((dog_id, surface, date), 0.5)

        # Get rolling form
        fav_form = form_lookup.get((fav_id, date), 0.5)
        dog_form = form_lookup.get((dog_id, date), 0.5)

        # Handle NaN
        fav_surface_wr = fav_surface_wr if pd.notna(fav_surface_wr) else 0.5
        dog_surface_wr = dog_surface_wr if pd.notna(dog_surface_wr) else 0.5
        fav_form = fav_form if pd.notna(fav_form) else 0.5
        dog_form = dog_form if pd.notna(dog_form) else 0.5

        features.append({
            'match_id': idx,
            'date': date,
            'year': row['year'],
            'surface': surface,
            'round': row.get('round', 'R32'),
            'tourney_level': row.get('tourney_level', 'A'),

            # Ranking features
            'fav_rank': fav_rank,
            'dog_rank': dog_rank,
            'rank_diff': dog_rank - fav_rank,
            'rank_ratio': dog_rank / fav_rank if fav_rank > 0 else 1,
            'log_rank_ratio': np.log(dog_rank / fav_rank) if fav_rank > 0 else 0,

            # Surface-specific features (HISTORICAL only)
            'fav_surface_wr': fav_surface_wr,
            'dog_surface_wr': dog_surface_wr,
            'surface_wr_diff': dog_surface_wr - fav_surface_wr,

            # Form features (rolling win rate)
            'fav_form': fav_form,
            'dog_form': dog_form,
            'form_diff': dog_form - fav_form,

            # Derived features
            'fav_is_top10': 1 if fav_rank <= 10 else 0,
            'fav_is_top20': 1 if fav_rank <= 20 else 0,
            'dog_is_top50': 1 if dog_rank <= 50 else 0,

            # Target
            'upset': upset
        })

    return pd.DataFrame(features)


def temporal_train_test_split(features_df: pd.DataFrame,
                               test_year: int = 2024) -> Tuple:
    """
    Split data temporally: train on earlier years, test on specified year.

    This avoids data leakage from using future matches to predict past ones.
    """
    train_df = features_df[features_df['year'] < test_year]
    test_df = features_df[features_df['year'] >= test_year]

    return train_df, test_df


def train_and_evaluate_models(X_train: pd.DataFrame, y_train: pd.Series,
                               X_test: pd.DataFrame, y_test: pd.Series,
                               feature_cols: list) -> Tuple[pd.DataFrame, dict]:
    """Train models and return results."""

    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[feature_cols])
    X_test_scaled = scaler.transform(X_test[feature_cols])

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        )
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        if 'Logistic' in name:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]

            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                        cv=5, scoring='roc_auc')
        else:
            model.fit(X_train[feature_cols], y_train)
            y_pred = model.predict(X_test[feature_cols])
            y_prob = model.predict_proba(X_test[feature_cols])[:, 1]

            cv_scores = cross_val_score(model, X_train[feature_cols], y_train,
                                        cv=5, scoring='roc_auc')

        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob),
            'CV ROC-AUC': cv_scores.mean(),
            'CV Std': cv_scores.std()
        })

        trained_models[name] = {
            'model': model,
            'y_prob': y_prob,
            'scaler': scaler if 'Logistic' in name else None
        }

        print(f"  Accuracy: {results[-1]['Accuracy']:.3f}")
        print(f"  ROC-AUC:  {results[-1]['ROC-AUC']:.3f}")
        print(f"  CV ROC-AUC: {results[-1]['CV ROC-AUC']:.3f} (+/- {results[-1]['CV Std']:.3f})")

    return pd.DataFrame(results), trained_models


def plot_results(results_df: pd.DataFrame, trained_models: dict,
                 y_test: pd.Series, feature_cols: list, X_test: pd.DataFrame):
    """Generate visualization plots."""

    # ROC curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    for name, data in trained_models.items():
        y_prob = data['y_prob']
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax1.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)

    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC=0.500)')
    ax1.set_xlabel('False Positive Rate', fontsize=11)
    ax1.set_ylabel('True Positive Rate', fontsize=11)
    ax1.set_title('ROC Curves (Temporal Split)', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')

    # Model comparison
    ax2 = axes[1]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    x = np.arange(len(metrics))
    width = 0.25

    for i, (_, row) in enumerate(results_df.iterrows()):
        values = [row[m] for m in metrics]
        ax2.bar(x + i*width, values, width, label=row['Model'], alpha=0.8)

    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: model_comparison.png")

    # Feature importance
    rf_model = trained_models['Random Forest']['model']
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_n = min(15, len(importance_df))
    top_features = importance_df.head(top_n)

    colors = ['#e74c3c' if 'surface' in f.lower() or 'form' in f.lower()
              else '#3498db' for f in top_features['Feature']]
    ax.barh(range(top_n), top_features['Importance'].values, color=colors, alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features['Feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title('Top Features for Upset Prediction\n(Red = Surface/Form features)',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: feature_importance.png")

    return importance_df


def print_summary(results_df: pd.DataFrame, importance_df: pd.DataFrame,
                  train_size: int, test_size: int, upset_rate: float):
    """Print findings summary."""
    print("\n" + "=" * 70)
    print("UPSET PREDICTION MODEL - KEY FINDINGS")
    print("=" * 70)

    print(f"\nData Split (Temporal - No Leakage):")
    print(f"  Train: {train_size:,} matches (2020-2023)")
    print(f"  Test:  {test_size:,} matches (2024)")

    best = results_df.loc[results_df['ROC-AUC'].idxmax()]
    print(f"\nBest Model: {best['Model']}")
    print(f"  ROC-AUC: {best['ROC-AUC']:.3f}")
    print(f"  CV ROC-AUC: {best['CV ROC-AUC']:.3f} (+/- {best['CV Std']:.3f})")
    print(f"  Accuracy: {best['Accuracy']:.3f}")

    print(f"\nBaseline Upset Rate: {upset_rate:.1%}")

    print(f"\nTop 5 Predictive Features:")
    for i, (_, row) in enumerate(importance_df.head(5).iterrows()):
        print(f"  {i+1}. {row['Feature']}: {row['Importance']:.3f}")

    # Check if surface/form features are important
    top_5 = importance_df.head(5)['Feature'].tolist()
    surface_form_in_top5 = any('surface' in f or 'form' in f for f in top_5)

    print(f"\nKey Insight:")
    if surface_form_in_top5:
        print("  Surface win rates and form features improve predictions.")
        print("  SDI-based features add value beyond ranking alone.")
    else:
        print("  Ranking features dominate, but surface/form still contribute.")

    print("\n" + "=" * 70)


def main():
    """Main analysis pipeline."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    data_dir = Path('data')

    # Load data
    print("Loading data...")
    df = load_and_prepare_data(data_dir)
    print(f"Loaded {len(df):,} matches")

    # Calculate historical stats (avoids data leakage)
    surface_wr_lookup, form_lookup = calculate_historical_stats(df)

    # Build features
    features_df = build_features(df, surface_wr_lookup, form_lookup)
    print(f"Feature matrix: {features_df.shape}")

    # Define upset
    upset_rate = features_df['upset'].mean()
    print(f"Overall upset rate: {upset_rate:.1%}")

    # Temporal split (CRITICAL for avoiding leakage)
    train_df, test_df = temporal_train_test_split(features_df, test_year=2024)
    print(f"\nTemporal split:")
    print(f"  Train: {len(train_df):,} matches (years < 2024)")
    print(f"  Test:  {len(test_df):,} matches (year = 2024)")

    # Encode categorical features
    train_encoded = pd.get_dummies(train_df, columns=['surface', 'tourney_level'],
                                    drop_first=True)
    test_encoded = pd.get_dummies(test_df, columns=['surface', 'tourney_level'],
                                   drop_first=True)

    # Align columns
    for col in train_encoded.columns:
        if col not in test_encoded.columns:
            test_encoded[col] = 0
    test_encoded = test_encoded[train_encoded.columns]

    # Define feature columns
    exclude_cols = ['match_id', 'date', 'year', 'upset', 'round']
    feature_cols = [c for c in train_encoded.columns if c not in exclude_cols]

    X_train = train_encoded[feature_cols]
    y_train = train_encoded['upset']
    X_test = test_encoded[feature_cols]
    y_test = test_encoded['upset']

    # Train and evaluate
    results_df, trained_models = train_and_evaluate_models(
        X_train, y_train, X_test, y_test, feature_cols
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Generate plots
    print("\nGenerating visualizations...")
    importance_df = plot_results(results_df, trained_models, y_test,
                                  feature_cols, X_test)

    # Print summary
    print_summary(results_df, importance_df, len(train_df), len(test_df), upset_rate)

    # Export results
    results_df.to_csv('model_results.csv', index=False)
    importance_df.to_csv('feature_importance.csv', index=False)
    print("\nExported: model_results.csv, feature_importance.csv")


if __name__ == '__main__':
    main()
