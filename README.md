# Tennis Surface Specialism Analysis

Quantifying surface dependency in men's professional tennis using the **Surface Dependency Index (SDI)** and analyzing how players' surface performance evolves over time.

## Overview

Tennis is played on three primary surfaces—hard court, clay, and grass—each favouring different playing styles. This project provides two analyses:

1. **Surface Dependency Index (SDI)**: Identifies surface specialists vs all-court players
2. **Performance Over Time**: Investigates how surface-specific performance changes with age

## Quick Start

```bash
# Clone the repository
git clone https://github.com/mattr-ta95/tennis_surface_specialism_analysis.git
cd tennis_surface_specialism_analysis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download data
python download_data.py

# Run analyses
python surface_specialism_analysis.py
python surface_performance_over_time.py
```

## Data Source

Match data from [Jeff Sackmann's tennis_atp](https://github.com/JeffSackmann/tennis_atp) repository (CC BY-NC-SA 4.0).

## Scripts

| Script | Description |
|--------|-------------|
| `download_data.py` | Downloads ATP match data (2020-2025) |
| `surface_specialism_analysis.py` | Calculates SDI and identifies specialists |
| `surface_performance_over_time.py` | Analyzes age-related performance changes |

## Key Findings

### 1. Win Rates Vary by Age and Surface

| Age | Clay | Grass | Hard |
|-----|------|-------|------|
| Under 20 | 52.6% | 39.4% | 50.8% |
| 20-22 | 53.3% | 54.8% | 54.0% |
| 23-25 | 53.4% | 50.4% | 52.2% |
| 26-28 | 48.4% | 51.9% | 48.7% |
| 29-31 | 45.4% | 43.0% | 44.0% |
| 35+ | 44.5% | **54.3%** | 49.6% |

**Grass is the outlier**: Young players struggle (39%), while veterans thrive (54%). Grass rewards refined technique over raw athleticism.

### 2. Peak Ages Differ by Surface

- **Hard court**: Peaks at age 20-22
- **Clay court**: Peaks later at 23-25 (requires tactical maturity)
- **Grass court**: Veterans outperform younger players

### 3. Surface Specialization Evolves

- **58.7%** of players see their "best surface" change during their career
- Players develop surface preferences in their early 20s, then adapt in their 30s

### 4. Age-Performance Correlation

| Surface | Correlation |
|---------|-------------|
| Clay | -0.627 (strongest decline with age) |
| Grass | -0.532 |
| Hard | -0.449 (most forgiving for aging players) |

### 5. Notable Player Insights

- **Sinner**: Hard court improved +32.8% (61.9% → 94.7%) from 2020-2024
- **Djokovic** (age 32-37): Elite on all surfaces (87.7% Hard, 82.8% Clay, 92.9% Grass)
- **Nadal** (late career): Clay dominance persisted (80% at 35+), hard declined more

## Surface Dependency Index (SDI)

$$\text{SDI} = \sigma(\text{win\_rate}_{\text{Hard}}, \text{win\_rate}_{\text{Clay}}, \text{win\_rate}_{\text{Grass}})$$

- **Low SDI** (< 0.05): All-court player
- **High SDI** (> 0.10): Surface specialist

## Outputs

### From `surface_specialism_analysis.py`:
- `player_surface_analysis.csv` - Player stats with SDI scores
- `surface_comparison.png` - Specialists vs all-court players
- `sdi_analysis.png` - SDI distribution and correlation plots
- `surface_preferences.png` - Best surface distribution
- `sdi_trend.png` - Temporal trend analysis

### From `surface_performance_over_time.py`:
- `surface_specialists.csv` - Most surface-dependent players
- `all_court_players.csv` - Most consistent players
- `win_rates_by_age.png` - Win rates by age bracket
- `specialization_by_age.png` - How specialization changes with age
- `trajectory_*.png` - Individual player trajectories

## License

MIT
