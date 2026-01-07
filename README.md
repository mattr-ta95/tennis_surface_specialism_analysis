# Tennis Surface Specialism Analysis

Quantifying surface dependency in men's professional tennis using the **Surface Dependency Index (SDI)**.

## Overview

Tennis is played on three primary surfaces—hard court, clay, and grass—each favouring different playing styles. This analysis introduces the SDI metric to identify:

- **Surface specialists**: Players with high performance variance across surfaces
- **All-court players**: Consistent performers regardless of surface

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/tennis_surface_impact.git
cd tennis_surface_impact

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download data
python download_data.py

# Run analysis
python surface_specialism_analysis.py
```

## Data Source

Match data from [Jeff Sackmann's tennis_atp](https://github.com/JeffSackmann/tennis_atp) repository (CC BY-NC-SA 4.0).

## Outputs

The analysis generates:

- `player_surface_analysis.csv` - Player stats with SDI scores
- `surface_comparison.png` - Specialists vs all-court players
- `sdi_analysis.png` - SDI distribution and correlation plots
- `surface_preferences.png` - Best surface distribution
- `sdi_trend.png` - Temporal trend analysis

## Surface Dependency Index (SDI)

$$\text{SDI} = \sigma(\text{win\_rate}_{\text{Hard}}, \text{win\_rate}_{\text{Clay}}, \text{win\_rate}_{\text{Grass}})$$

- **Low SDI** (< 0.05): All-court player
- **High SDI** (> 0.10): Surface specialist

## License

MIT
