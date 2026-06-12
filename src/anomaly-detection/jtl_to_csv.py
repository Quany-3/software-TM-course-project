#!/usr/bin/env python3
"""
Convert JMeter JTL results to time-series CSV for anomaly detection.

Usage:
    # Convert normal + high load data, treat high load period as anomaly
    python jtl_to_csv.py --normal results-normal.jtl --anomaly results-high.jtl --output data/jmeter_metrics.csv

    # With ground truth labels (0=normal, 1=anomaly)
    python jtl_to_csv.py --normal results-normal.jtl --anomaly results-high.jtl --output data/jmeter_metrics.csv --labels data/ground_truth.csv
"""

import argparse
import pandas as pd
import numpy as np
import os


def load_jtl(filepath):
    """Load a JMeter JTL file."""
    df = pd.read_csv(filepath)
    # Convert timestamp from ms to seconds
    df['timestamp_sec'] = df['timeStamp'] / 1000.0
    df['timestamp_sec'] = df['timestamp_sec'] - df['timestamp_sec'].min()
    return df


def aggregate_window(df, window_sec=10):
    """
    Aggregate per-request JTL data into fixed time windows.

    Returns a DataFrame with one row per time window, with aggregate metrics.
    """
    df = df.copy()
    df['window'] = (df['timestamp_sec'] / window_sec).astype(int)

    grouped = df.groupby('window').agg(
        timestamp_sec=('timestamp_sec', 'mean'),
        avg_latency_ms=('elapsed', 'mean'),
        median_latency_ms=('elapsed', 'median'),
        max_latency_ms=('elapsed', 'max'),
        min_latency_ms=('elapsed', 'min'),
        p90_latency_ms=('elapsed', lambda x: x.quantile(0.9)),
        p99_latency_ms=('elapsed', lambda x: x.quantile(0.99)),
        throughput_req_per_sec=('elapsed', 'count'),
        avg_connect_ms=('Connect', 'mean'),
        avg_bytes=('bytes', 'mean'),
        avg_sent_bytes=('sentBytes', 'mean'),
        avg_threads=('grpThreads', 'mean'),
        all_threads=('allThreads', 'mean'),
        error_rate=('success', lambda x: 1.0 - x.mean()),
    ).reset_index()

    # Convert count to per-second throughput
    grouped['throughput_req_per_sec'] = grouped['throughput_req_per_sec'] / window_sec
    # Fill NA (e.g., if no errors, quantile can be NaN)
    grouped = grouped.fillna(0)

    return grouped


def create_merged_dataset(normal_df, anomaly_df, window_sec=10):
    """
    Merge normal and anomaly periods into a single time series.
    Adds a 'label' column: 0 = normal, 1 = anomaly.
    """
    normal_agg = aggregate_window(normal_df, window_sec)
    normal_agg['label'] = 0
    normal_agg['phase'] = 'normal'

    anomaly_agg = aggregate_window(anomaly_df, window_sec)
    anomaly_agg['label'] = 1
    anomaly_agg['phase'] = 'anomaly'

    # Offset anomaly timestamps to come after normal
    if len(normal_agg) > 0:
        offset = normal_agg['timestamp_sec'].max() + window_sec
        anomaly_agg['timestamp_sec'] = anomaly_agg['timestamp_sec'] - \
            anomaly_agg['timestamp_sec'].min() + offset

    combined = pd.concat([normal_agg, anomaly_agg], ignore_index=True)
    combined = combined.sort_values('timestamp_sec').reset_index(drop=True)

    return combined


def main():
    parser = argparse.ArgumentParser(description='Convert JTL to time-series CSV')
    parser.add_argument('--normal', type=str, required=True,
                        help='Path to normal-operation JTL file')
    parser.add_argument('--anomaly', type=str,
                        help='Path to anomalous-operation JTL file')
    parser.add_argument('--window', type=int, default=10,
                        help='Aggregation window in seconds (default: 10)')
    parser.add_argument('--output', type=str, default='data/jmeter_metrics.csv',
                        help='Output CSV path')
    parser.add_argument('--labels-output', type=str,
                        help='Output ground-truth labels CSV')
    parser.add_argument('--single', action='store_true',
                        help='Process only --normal as a single file (no labels)')
    args = parser.parse_args()

    print(f"Loading: {args.normal}")
    normal_df = load_jtl(args.normal)
    print(f"  {len(normal_df)} requests, "
          f"duration={normal_df['timestamp_sec'].max():.1f}s")

    if args.single or args.anomaly is None:
        # Single file mode — no ground truth
        print(f"Aggregating with {args.window}s windows...")
        result = aggregate_window(normal_df, args.window)
        print(f"  → {len(result)} time windows")
    else:
        print(f"Loading: {args.anomaly}")
        anomaly_df = load_jtl(args.anomaly)
        print(f"  {len(anomaly_df)} requests, "
              f"duration={anomaly_df['timestamp_sec'].max():.1f}s")

        print(f"Merging normal + anomaly with {args.window}s windows...")
        result = create_merged_dataset(normal_df, anomaly_df, args.window)
        normal_windows = (result['label'] == 0).sum()
        anomaly_windows = (result['label'] == 1).sum()
        print(f"  → {len(result)} total windows "
              f"({normal_windows} normal + {anomaly_windows} anomaly)")

    # Save metrics CSV (without label column for model input)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Columns for model input (excluding label and phase)
    metric_cols = [c for c in result.columns
                   if c not in ('window', 'label', 'phase', 'timestamp_sec')]

    # Save full dataset
    result.to_csv(args.output, index=False)
    print(f"\nMetrics saved to: {args.output}")
    print(f"Features ({len(metric_cols)}): {', '.join(metric_cols)}")

    # Save ground truth labels separately
    if 'label' in result.columns:
        label_path = args.labels_output or args.output.replace('.csv', '_labels.csv')
        pd.DataFrame({
            'label': result['label'].values
        }).to_csv(label_path, index=False)
        print(f"Labels saved to: {label_path}")

    # Quick stats
    print(f"\n=== Quick Stats ===")
    if 'label' in result.columns:
        for label_val, label_name in [(0, 'Normal'), (1, 'Anomaly')]:
            subset = result[result['label'] == label_val]
            print(f"\n{label_name} windows ({len(subset)}):")
            print(f"  avg_latency: {subset['avg_latency_ms'].mean():.1f} ms")
            print(f"  max_latency: {subset['max_latency_ms'].mean():.1f} ms")
            print(f"  throughput:  {subset['throughput_req_per_sec'].mean():.2f} req/s")
            print(f"  threads:     {subset['avg_threads'].mean():.1f}")
    else:
        print(f"\n  avg_latency: {result['avg_latency_ms'].mean():.1f} ms")
        print(f"  max_latency: {result['max_latency_ms'].mean():.1f} ms")
        print(f"  throughput:  {result['throughput_req_per_sec'].mean():.2f} req/s")


if __name__ == '__main__':
    main()
