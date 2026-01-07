#!/usr/bin/env python3
"""
Download ATP match data from Jeff Sackmann's tennis_atp repository.
This is the gold standard open-source tennis dataset.

Usage:
    python download_data.py

Data source: https://github.com/JeffSackmann/tennis_atp
License: CC BY-NC-SA 4.0
"""

import os
import urllib.request
import ssl
from pathlib import Path

# Years to download (adjust as needed)
YEARS = range(2020, 2026)  # 2020-2025

# Base URL for raw GitHub content
BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"

# Files to download
FILES = [
    "atp_matches_{year}.csv",
    "atp_players.csv",  # Player biographical data
]

def download_file(url: str, dest: Path) -> bool:
    """Download a file from URL to destination."""
    try:
        print(f"  Downloading {url}...")
        context = ssl.create_default_context()
        with urllib.request.urlopen(url, timeout=30, context=context) as response:
            data = response.read()
            dest.write_bytes(data)
            print(f"  ✓ Saved to {dest} ({len(data):,} bytes)")
            return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    # Create data directory
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("ATP Tennis Data Downloader")
    print("Source: github.com/JeffSackmann/tennis_atp")
    print("=" * 60)
    
    # Download player file (once)
    player_url = f"{BASE_URL}/atp_players.csv"
    player_dest = data_dir / "atp_players.csv"
    if not player_dest.exists():
        download_file(player_url, player_dest)
    else:
        print(f"  ⊙ {player_dest} already exists, skipping")
    
    # Download match files by year
    print(f"\nDownloading match data for years {min(YEARS)}-{max(YEARS)}...")
    
    success_count = 0
    for year in YEARS:
        filename = f"atp_matches_{year}.csv"
        url = f"{BASE_URL}/{filename}"
        dest = data_dir / filename
        
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  ⊙ {filename} already exists, skipping")
            success_count += 1
            continue
        
        if download_file(url, dest):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"Downloaded {success_count}/{len(list(YEARS))} year files")
    print("=" * 60)
    
    # Verify and combine
    print("\nVerifying data...")
    import pandas as pd
    
    all_matches = []
    for year in YEARS:
        filepath = data_dir / f"atp_matches_{year}.csv"
        if filepath.exists() and filepath.stat().st_size > 0:
            df = pd.read_csv(filepath)
            all_matches.append(df)
            print(f"  {year}: {len(df):,} matches")
    
    if all_matches:
        combined = pd.concat(all_matches, ignore_index=True)
        combined_path = data_dir / "atp_matches_combined.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n✓ Combined dataset: {len(combined):,} total matches")
        print(f"  Saved to: {combined_path}")
        print(f"  Surfaces: {combined['surface'].unique().tolist()}")
        print(f"  Players: {combined['winner_name'].nunique():,}")


if __name__ == "__main__":
    main()
