#!/usr/bin/env python3
"""Visualize PoR method comparison results.

This script is designed for por_method_comparison_*.csv files which contain
summary statistics for multiple PoR detection methods, NOT time series data.

It creates:
1. A comparison table of PoR methods
2. Bar charts showing release distance for each method
3. Bar charts showing baseline metrics for each method
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def _safe_float(value: Any) -> Optional[float]:
    """Safely convert a value to float."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def _parse_coordinate_list(value: Any) -> Optional[List[float]]:
    """Parse coordinate string like '[359.179,-138.759,1923.916]' to list of floats."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    try:
        # Remove brackets and split
        cleaned = text.strip('[]')
        parts = cleaned.split(',')
        return [float(p.strip()) for p in parts if p.strip()]
    except (ValueError, TypeError):
        return None


def plot_method_comparison(row: Dict[str, Any], output_parent: Path) -> List[Path]:
    """Create comparison plots for a single penalty row."""
    output_files = []
    
    # Extract penalty ID
    penalty_id = str(row.get('penalty_id', row.get('id', 'unknown'))).strip()
    
    # Extract data for each method
    methods = ['method1', 'method2', 'method3']
    method_data = {}
    
    for method in methods:
        por_col = f'PoR_{method}'
        accel_col = f'acceleration_{method}'
        vel_col = f'velocity_{method}'
        dir_col = f'direction_{method}'
        coord_col = f'coordinate_{method}'
        
        method_data[method] = {
            'por_frame': _safe_int(row.get(por_col)),
            'acceleration': _safe_float(row.get(accel_col)),
            'velocity': _safe_float(row.get(vel_col)),
            'direction': _safe_float(row.get(dir_col)),
            'coordinate': _parse_coordinate_list(row.get(coord_col)),
        }
    
    # Extract segment info
    throw_segment = str(row.get('throw_segment', '')).strip()
    segment_start = _safe_int(row.get('segment_start'))
    holding_por_frame = _safe_int(row.get('holding_por_frame'))
    pre_impact_frame = _safe_int(row.get('pre_impact_frame'))
    
    # Extract baseline metrics
    baseline_mean = _safe_float(row.get('method1_baseline_mean_mm'))
    baseline_std = _safe_float(row.get('method1_baseline_std_mm'))
    release_distance = _safe_float(row.get('method1_release_distance_mm'))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'PoR Method Comparison - Penalty {penalty_id}', fontsize=14, fontweight='bold')
    
    # 1. Release distance comparison (if available)
    ax1 = axes[0, 0]
    if release_distance is not None:
        method_names = ['Method 1\n(Fixture)', 'Method 2\n(Mocap)', 'Method 3\n(Other)']
        distances = [release_distance]  # Only method1 has release distance in current format
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        bars = ax1.bar(method_names, distances, color=colors[:len(distances)], alpha=0.7)
        ax1.set_ylabel('Release Distance (mm)')
        ax1.set_title('Release Distance from Baseline')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, dist in zip(bars, distances):
            if dist is not None:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{dist:.1f}', ha='center', va='bottom', fontsize=10)
    else:
        ax1.text(0.5, 0.5, 'No release distance data', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Release Distance from Baseline')
    
    # 2. Baseline metrics
    ax2 = axes[0, 1]
    if baseline_mean is not None and baseline_std is not None:
        categories = ['Baseline\nMean (mm)', 'Baseline\nStd (mm)']
        values = [baseline_mean, baseline_std]
        colors = ['#9b59b6', '#f39c12']
        
        bars = ax2.bar(categories, values, color=colors, alpha=0.7)
        ax2.set_ylabel('Value (mm)')
        ax2.set_title('Baseline Metrics (Method 1)')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            if val is not None:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'No baseline data', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Baseline Metrics')
    
    # 3. Velocity comparison
    ax3 = axes[1, 0]
    velocities = [method_data[m]['velocity'] for m in methods]
    if any(v is not None for v in velocities):
        method_names_short = ['M1', 'M2', 'M3']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        bars = ax3.bar(method_names_short, velocities, color=colors, alpha=0.7)
        ax3.set_ylabel('Velocity (m/s)')
        ax3.set_title('Release Velocity by Method')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, vel in zip(bars, velocities):
            if vel is not None:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{vel:.2f}', ha='center', va='bottom', fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'No velocity data', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Release Velocity by Method')
    
    # 4. Acceleration comparison
    ax4 = axes[1, 1]
    accelerations = [method_data[m]['acceleration'] for m in methods]
    if any(a is not None for a in accelerations):
        method_names_short = ['M1', 'M2', 'M3']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        bars = ax4.bar(method_names_short, accelerations, color=colors, alpha=0.7)
        ax4.set_ylabel('Acceleration (m/s²)')
        ax4.set_title('Release Acceleration by Method')
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar, acc in zip(bars, accelerations):
            if acc is not None:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{acc:.2f}', ha='center', va='bottom', fontsize=10)
    else:
        ax4.text(0.5, 0.5, 'No acceleration data', ha='center', va='center',
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Release Acceleration by Method')
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_parent / f'por_comparison_{penalty_id}.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=160, bbox_inches='tight')
    plt.close()
    output_files.append(output_file)
    
    # Create a summary table plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for method in methods:
        data = method_data[method]
        table_data.append([
            method.upper(),
            str(data['por_frame']) if data['por_frame'] is not None else 'N/A',
            f"{data['velocity']:.2f}" if data['velocity'] is not None else 'N/A',
            f"{data['acceleration']:.2f}" if data['acceleration'] is not None else 'N/A',
            f"{data['direction']:.2f}" if data['direction'] is not None else 'N/A',
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Method', 'PoR Frame', 'Velocity (m/s)', 'Acceleration (m/s²)', 'Direction (°)'],
        loc='center',
        cellLoc='center',
        colWidths=[0.15, 0.2, 0.25, 0.25, 0.2]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Color header
    for i in range(len(table.get_celld()[0, :])):
        table.get_celld()[0, i].set_facecolor('#3498db')
        table.get_celld()[0, i].set_text_props(weight='bold', color='white')
    
    # Color method rows
    colors = ['#d5e8d4', '#dae8fc', '#fff2cc']
    for i, color in enumerate(colors):
        for j in range(len(table.get_celld()[i+1, :])):
            table.get_celld()[i+1, j].set_facecolor(color)
    
    ax.set_title(f'PoR Method Summary - Penalty {penalty_id}', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save table figure
    table_file = output_parent / f'por_table_{penalty_id}.png'
    table_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(table_file, dpi=160, bbox_inches='tight')
    plt.close()
    output_files.append(table_file)
    
    return output_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot PoR method comparison results from por_method_comparison_*.csv files. "
            "These files contain summary statistics for multiple PoR detection methods."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to por_method_comparison_*.csv file",
    )
    parser.add_argument(
        "--id",
        default=None,
        help="Optional single penalty id to plot",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    input_path = args.input.resolve()
    
    # Try to read with different separators
    try:
        df = pd.read_csv(input_path, sep=";", dtype=str)
    except Exception:
        try:
            df = pd.read_csv(input_path, sep=",", dtype=str)
        except Exception as e:
            print(f"Failed to read CSV: {e}")
            return 1
    
    # Check if this is a valid por_method_comparison file
    required_cols = ['PoR_method1', 'PoR_method2', 'PoR_method3']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Input file does not appear to be a PoR method comparison CSV.")
        print(f"Expected columns: {required_cols}")
        print(f"Found columns: {list(df.columns)}")
        return 1
    
    # Filter by ID if specified
    if args.id is not None:
        # Try to find ID column (could be 'penalty_id', 'id', or first column)
        id_col = None
        for col in ['penalty_id', 'id', df.columns[0]]:
            if col in df.columns:
                id_col = col
                break
        
        if id_col:
            rows = df[df[id_col].astype(str).str.strip() == str(args.id).strip()]
        else:
            print("Could not find ID column in CSV")
            return 1
    else:
        rows = df
    
    if rows.empty:
        print("No matching rows found.")
        return 1
    
    out_parent = input_path.parent / 'por_comparison_plots'
    generated = 0
    for i, row in enumerate(rows.to_dict(orient="records"), start=1):
        try:
            output_files = plot_method_comparison(row=row, output_parent=out_parent)
            for f in output_files:
                print(f"Saved: {f}")
            generated += 1
        except Exception as exc:
            penalty_id = str(row.get('penalty_id', row.get('id', f'row_{i}'))).strip()
            print(f"Skipped id={penalty_id}: {exc}")
    
    if generated == 0:
        print("No plots were generated.")
        return 2
    
    print(f"\nGenerated plots for {generated} row(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())