#!/usr/bin/env python3
"""
Feature selection for anomaly detection: identify features most sensitive to faults.
Computes per-feature statistics during normal vs anomaly periods to rank features.
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import json
import os
import sys


def analyze_features(df, label_col='label'):
    """Rank features by their sensitivity to faults."""
    # Exclude non-feature columns
    exclude = {label_col, 'timestamp'}
    feature_cols = [c for c in df.columns if c not in exclude]

    normal = df[df[label_col] == 0]
    anomaly = df[df[label_col] == 1]

    results = []
    for col in feature_cols:
        n_vals = normal[col].dropna().values
        a_vals = anomaly[col].dropna().values

        if len(n_vals) < 5 or len(a_vals) < 5:
            continue

        # Kolmogorov-Smirnov test: how different are the distributions?
        ks_stat, ks_pval = ks_2samp(n_vals, a_vals)

        # Basic statistics
        n_mean, n_std = np.mean(n_vals), np.std(n_vals)
        a_mean, a_std = np.mean(a_vals), np.std(a_vals)

        # Effect size (Cohen's d-like)
        pooled_std = np.sqrt((n_std**2 + a_std**2) / 2)
        if pooled_std > 0:
            effect_size = abs(a_mean - n_mean) / pooled_std
        else:
            effect_size = 0

        # Variability ratio
        if n_std > 0:
            var_ratio = a_std / n_std
        else:
            var_ratio = 1.0

        results.append({
            'feature': col,
            'normal_mean': float(n_mean),
            'anomaly_mean': float(a_mean),
            'normal_std': float(n_std),
            'anomaly_std': float(a_std),
            'effect_size': float(effect_size),
            'var_ratio': float(var_ratio),
            'ks_stat': float(ks_stat),
            'ks_pval': float(ks_pval),
            'n_normal': len(n_vals),
            'n_anomaly': len(a_vals),
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('effect_size', ascending=False)
    return df_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to metrics.csv')
    parser.add_argument('--top', type=int, default=15, help='Number of top features to keep')
    parser.add_argument('--output', type=str, help='Output directory')
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    label_col = 'label' if 'label' in df.columns else None
    if label_col is None:
        print("ERROR: No 'label' column found in data")
        sys.exit(1)

    print(f"Analyzing {len(df)} rows, {len(df.columns)-2} features...")
    results = analyze_features(df, label_col)

    print(f"\n{'='*70}")
    print(f"Top {min(args.top, len(results))} features by effect size:")
    print(f"{'='*70}")
    print(f"{'Feature':<50} {'Effect':>6} {'KS_stat':>8} {'KS_pval':>8}")
    print("-" * 70)
    for _, row in results.head(args.top).iterrows():
        print(f"{row['feature']:<50} {row['effect_size']:>6.3f} "
              f"{row['ks_stat']:>8.3f} {row['ks_pval']:>8.4f}")

    # Identify features with near-zero variance
    low_var = results[results['normal_std'] < 1e-8]
    if len(low_var) > 0:
        print(f"\nNear-zero variance features ({len(low_var)}):")
        for _, row in low_var.iterrows():
            print(f"  {row['feature']} (std={row['normal_std']:.2e})")

    # Identify highly correlated pairs
    feature_cols = [c for c in df.columns if c not in ('timestamp', 'label')]
    corr = df[feature_cols].corr().abs()
    high_corr = []
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            if corr.iloc[i, j] > 0.95:
                high_corr.append((feature_cols[i], feature_cols[j], corr.iloc[i, j]))

    if high_corr:
        print(f"\nHighly correlated feature pairs (r > 0.95): {len(high_corr)}")
        for f1, f2, r in high_corr[:20]:
            print(f"  {f1} <-> {f2}: r={r:.3f}")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        top_features = results.head(args.top)['feature'].tolist()
        with open(os.path.join(args.output, 'selected_features.json'), 'w') as f:
            json.dump({
                'top_n': args.top,
                'selected_features': top_features,
                'all_rankings': results[['feature', 'effect_size', 'ks_stat']].to_dict('records')
            }, f, indent=2)

        results.to_csv(os.path.join(args.output, 'feature_analysis.csv'), index=False)
        print(f"\nResults saved to {args.output}/")


if __name__ == '__main__':
    main()
