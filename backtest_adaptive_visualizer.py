# backtest_adaptive_visualizer.py
"""
Visualization tools for adaptive DC backtest results

Creates charts comparing static vs adaptive DC performance over time
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

from config import OUTPUT_DIR


def plot_accuracy_over_time(static_results: List[Dict],
                            adaptive_results: List[Dict],
                            market: str = 'OU_2_5'):
    """
    Plot accuracy over time for a specific market
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Extract data for plotting
    static_data = []
    adaptive_data = []

    for static, adaptive in zip(static_results, adaptive_results):
        if market in static.get('markets', {}) and market in adaptive.get('markets', {}):
            period_end = static['period_end']
            static_acc = static['markets'][market]['accuracy'] * 100
            adaptive_acc = adaptive['markets'][market]['accuracy'] * 100

            static_data.append({'date': period_end, 'accuracy': static_acc})
            adaptive_data.append({'date': period_end, 'accuracy': adaptive_acc})

    if not static_data:
        print(f"No data available for {market}")
        return

    static_df = pd.DataFrame(static_data)
    adaptive_df = pd.DataFrame(adaptive_data)

    # Plot lines
    ax.plot(static_df['date'], static_df['accuracy'],
            'o-', label='Static DC', color='#3498db', linewidth=2, markersize=6)
    ax.plot(adaptive_df['date'], adaptive_df['accuracy'],
            's-', label='Adaptive DC', color='#e74c3c', linewidth=2, markersize=6)

    # Add rolling averages
    if len(static_df) >= 4:
        static_rolling = static_df['accuracy'].rolling(window=4, min_periods=1).mean()
        adaptive_rolling = adaptive_df['accuracy'].rolling(window=4, min_periods=1).mean()

        ax.plot(static_df['date'], static_rolling,
                '--', color='#3498db', alpha=0.5, linewidth=1.5, label='Static (4-week avg)')
        ax.plot(adaptive_df['date'], adaptive_rolling,
                '--', color='#e74c3c', alpha=0.5, linewidth=1.5, label='Adaptive (4-week avg)')

    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'{market} Accuracy Over Time: Static vs Adaptive DC', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Horizontal line at 50%
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Random guess')

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / f'backtest_adaptive_{market}_timeline.png'
    plt.savefig(output_path, dpi=150)
    print(f" Saved: {output_path}")

    plt.close()


def plot_brier_score_comparison(static_results: List[Dict],
                                adaptive_results: List[Dict]):
    """
    Compare Brier scores (calibration) across all markets
    """
    # Aggregate Brier scores by market
    static_brier = {}
    adaptive_brier = {}

    for result in static_results:
        for market, stats in result.get('markets', {}).items():
            if market not in static_brier:
                static_brier[market] = []
            static_brier[market].append(stats['brier_score'])

    for result in adaptive_results:
        for market, stats in result.get('markets', {}).items():
            if market not in adaptive_brier:
                adaptive_brier[market] = []
            adaptive_brier[market].append(stats['brier_score'])

    # Calculate averages
    markets = sorted(set(static_brier.keys()) & set(adaptive_brier.keys()))

    if not markets:
        print("No markets to compare")
        return

    static_avg = [np.mean(static_brier[m]) for m in markets]
    adaptive_avg = [np.mean(adaptive_brier[m]) for m in markets]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(markets))
    width = 0.35

    bars1 = ax.bar(x - width/2, static_avg, width, label='Static DC', color='#3498db')
    bars2 = ax.bar(x + width/2, adaptive_avg, width, label='Adaptive DC', color='#e74c3c')

    # Formatting
    ax.set_xlabel('Market', fontsize=12)
    ax.set_ylabel('Brier Score (lower is better)', fontsize=12)
    ax.set_title('Calibration Comparison: Brier Scores', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(markets, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / 'backtest_adaptive_brier_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f" Saved: {output_path}")

    plt.close()


def plot_improvement_heatmap(comparison_df: pd.DataFrame):
    """
    Create heatmap showing which markets improved with adaptive DC
    """
    if comparison_df.empty:
        print("No comparison data")
        return

    # Prepare data for heatmap
    markets = comparison_df['Market'].values
    acc_diff = comparison_df['Acc_Diff_%'].values

    # Create matrix (1 row)
    data = acc_diff.reshape(1, -1)

    fig, ax = plt.subplots(figsize=(14, 3))

    # Heatmap
    sns.heatmap(data,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                center=0,
                cbar_kws={'label': 'Accuracy Difference (%)'},
                xticklabels=markets,
                yticklabels=['Adaptive vs Static'],
                ax=ax,
                vmin=-5,
                vmax=5)

    ax.set_title('Accuracy Improvement by Market (Adaptive - Static)',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / 'backtest_adaptive_improvement_heatmap.png'
    plt.savefig(output_path, dpi=150)
    print(f" Saved: {output_path}")

    plt.close()


def plot_cumulative_improvement(static_results: List[Dict],
                               adaptive_results: List[Dict]):
    """
    Plot cumulative accuracy improvement over time
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Calculate cumulative differences for all markets
    cumulative_diff = []
    dates = []

    for static, adaptive in zip(static_results, adaptive_results):
        date = static['period_end']

        # Average accuracy across all markets in this period
        static_accs = []
        adaptive_accs = []

        for market in static.get('markets', {}).keys():
            if market in adaptive.get('markets', {}):
                static_accs.append(static['markets'][market]['accuracy'])
                adaptive_accs.append(adaptive['markets'][market]['accuracy'])

        if static_accs and adaptive_accs:
            static_avg = np.mean(static_accs) * 100
            adaptive_avg = np.mean(adaptive_accs) * 100

            diff = adaptive_avg - static_avg
            cumulative_diff.append(diff)
            dates.append(date)

    if not cumulative_diff:
        print("No data for cumulative plot")
        return

    # Cumulative sum
    cumulative_sum = np.cumsum(cumulative_diff)

    # Plot
    ax.plot(dates, cumulative_sum, 'o-', linewidth=2.5, markersize=7, color='#9b59b6')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    # Fill area
    ax.fill_between(dates, 0, cumulative_sum,
                    where=(np.array(cumulative_sum) > 0),
                    alpha=0.3, color='green', label='Adaptive better')
    ax.fill_between(dates, 0, cumulative_sum,
                    where=(np.array(cumulative_sum) <= 0),
                    alpha=0.3, color='red', label='Static better')

    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Accuracy Difference (%)', fontsize=12)
    ax.set_title('Cumulative Performance: Adaptive vs Static DC',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / 'backtest_adaptive_cumulative.png'
    plt.savefig(output_path, dpi=150)
    print(f" Saved: {output_path}")

    plt.close()


def generate_all_visualizations(output_dir: Path = OUTPUT_DIR):
    """
    Generate all visualizations from saved backtest results

    This reads the CSV outputs from adaptive backtest and creates charts
    """
    print("\n" + "="*70)
    print(" GENERATING ADAPTIVE DC VISUALIZATIONS")
    print("="*70)

    # Check if comparison file exists
    comparison_path = output_dir / "backtest_dc_comparison.csv"

    if not comparison_path.exists():
        print(f"\n ERROR: {comparison_path} not found")
        print("Run backtest_adaptive_dc.py first to generate results")
        return

    # Load comparison data
    comparison_df = pd.read_csv(comparison_path)

    print(f"\nLoaded comparison data: {len(comparison_df)} markets")

    # Plot improvement heatmap
    print("\nGenerating improvement heatmap...")
    plot_improvement_heatmap(comparison_df)

    # Note: For timeline plots, we need the raw results (not just summary)
    # These would need to be saved during the backtest run

    print("\n" + "="*70)
    print(" VISUALIZATIONS COMPLETE")
    print("="*70)
    print("\nGenerated charts:")
    print("  • backtest_adaptive_improvement_heatmap.png")
    print("\nNote: To generate timeline charts, run backtest and save period-by-period results")


if __name__ == "__main__":
    generate_all_visualizations()
