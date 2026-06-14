#!/usr/bin/env python3
"""
Train and evaluate Donut (WWW18) and USAD (KDD20) anomaly detection models.

Usage:
    # Train USAD on multivariate data
    python run.py --model usad --data metrics.csv --window 64 --epochs 50

    # Train Donut on univariate data
    python run.py --model donut --data cpu.csv --window 120 --epochs 50

    # Train both and compare
    python run.py --model both --data metrics.csv --window 64 --epochs 50
"""

import argparse
import os
import sys
import json
from datetime import datetime

import numpy as np
import torch
import pandas as pd

from models import USAD, Donut, train_usad, train_donut
from data_utils import prepare_data, prepare_multivariate_data, create_windows, apply_normalize


def coerce_labels(series):
    """Convert common label formats to 0=normal and 1=anomaly."""
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).values

    normalized = series.astype(str).str.strip().str.lower()
    anomaly_values = {'1', 'true', 'anomaly', 'abnormal', 'fault', 'faulty'}
    return normalized.isin(anomaly_values).astype(int).values


def load_csv(filepath, columns=None, max_rows=None):
    """Load time series data from a CSV file."""
    df = pd.read_csv(filepath, nrows=max_rows)

    if columns is not None:
        missing = set(columns) - set(df.columns)
        if missing:
            raise ValueError(f"Columns not found in CSV: {missing}")
        data = df[columns].values.astype(np.float32)
        feature_names = columns
    else:
        # Use all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude non-feature columns
        exclude_cols = {'label', 'window', 'timestamp_sec', 'phase', 'timestamp'}
        feature_names = [c for c in numeric_cols if c not in exclude_cols
                        and c.lower() not in exclude_cols]
        if not feature_names:
            raise ValueError("No numeric feature columns found in CSV "
                           f"(numeric cols: {numeric_cols})")
        data = df[feature_names].values.astype(np.float32)

    # Handle NaN by forward fill then zero fill
    df_clean = pd.DataFrame(data).ffill().fillna(0)
    return df_clean.values.astype(np.float32), feature_names


def load_labels_from_csv(filepath, n_timestamps, max_rows=None):
    """Load inline labels from a CSV label column, if present."""
    df = pd.read_csv(filepath, nrows=max_rows)
    label_col = next((c for c in df.columns if c.lower() == 'label'), None)
    if label_col is None:
        return None
    labels = coerce_labels(df[label_col])
    return align_labels(labels, n_timestamps)


def align_labels(labels, n_timestamps):
    """Pad or trim labels to match the metric array length."""
    if len(labels) > n_timestamps:
        return labels[:n_timestamps]
    if len(labels) < n_timestamps:
        return np.pad(labels, (0, n_timestamps - len(labels)))
    return labels


def load_prometheus_csv(filepath, max_rows=None):
    """
    Load Prometheus-exported CSV.
    Expected format: first column is timestamp, remaining columns are metric values.
    """
    df = pd.read_csv(filepath, nrows=max_rows)

    # Try to identify timestamp column
    ts_col = None
    for col in df.columns:
        if 'time' in col.lower() or 'timestamp' in col.lower():
            ts_col = col
            break

    if ts_col:
        df = df.drop(columns=[ts_col])

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found")

    data = df[numeric_cols].values.astype(np.float32)
    # Handle NaN
    df_clean = pd.DataFrame(data).ffill().fillna(0)
    return df_clean.values.astype(np.float32), numeric_cols


def score_usad_timestamps(model, data, norm_params, window_size, stride=1,
                          device='cpu'):
    """Score each timestamp by averaging USAD window scores covering it."""
    model.eval()
    model = model.to(device)

    normalized = apply_normalize(data, norm_params)
    windows = create_windows(normalized, window_size, stride)
    windows_t = torch.FloatTensor(windows).to(device)
    if windows_t.dim() == 2:
        windows_t = windows_t.unsqueeze(-1)
    windows_t = windows_t.reshape(windows_t.size(0), -1)

    scores = model.anomaly_score(windows_t).cpu().numpy()

    n_timestamps = len(data)
    ts_scores = np.zeros(n_timestamps)
    ts_counts = np.zeros(n_timestamps)

    for i in range(len(scores)):
        start = i * stride
        end = start + window_size
        ts_scores[start:end] += scores[i]
        ts_counts[start:end] += 1

    ts_counts[ts_counts == 0] = 1
    return ts_scores / ts_counts


def detect_anomalies_usad(model, data, norm_params, window_size, stride=1,
                          threshold_percentile=95, device='cpu',
                          threshold_data=None):
    """Run USAD on full dataset and return anomaly labels."""
    avg_scores = score_usad_timestamps(
        model, data, norm_params, window_size, stride, device
    )

    calibration_data = data if threshold_data is None else threshold_data
    calibration_scores = score_usad_timestamps(
        model, calibration_data, norm_params, window_size, stride, device
    )
    threshold = np.percentile(calibration_scores, threshold_percentile)
    labels = (avg_scores > threshold).astype(int)

    return avg_scores, labels, threshold


def score_donut_timestamps(model, data, norm_params, window_size, stride=1,
                           n_samples=1024, device='cpu', reduction='last',
                           norm_method='zscore'):
    """Score each timestamp with Donut reconstruction probability."""
    model.eval()
    model = model.to(device)

    normalized = apply_normalize(data.reshape(-1, 1), norm_params, method=norm_method)
    # Donut expects univariate, so take first column if multivariate
    if normalized.ndim == 2 and normalized.shape[1] > 1:
        normalized = normalized[:, :1]
    normalized = normalized.flatten()

    windows = create_windows(normalized, window_size, stride)
    windows_t = torch.FloatTensor(windows).to(device)

    scores = model.anomaly_score(
        windows_t, n_samples=n_samples, reduction=reduction
    ).cpu().numpy()

    n_timestamps = len(normalized)
    ts_scores = np.zeros(n_timestamps)
    ts_counts = np.zeros(n_timestamps)

    for i in range(len(scores)):
        if reduction == 'last':
            idx = i * stride + window_size - 1
            ts_scores[idx] += scores[i]
            ts_counts[idx] += 1
        else:
            start = i * stride
            end = start + window_size
            ts_scores[start:end] += scores[i]
            ts_counts[start:end] += 1

    ts_counts[ts_counts == 0] = 1
    avg_scores = ts_scores / ts_counts
    if reduction == 'last' and len(avg_scores) >= window_size:
        avg_scores[:window_size - 1] = avg_scores[window_size - 1]
    return avg_scores


def detect_anomalies_donut(model, data, norm_params, window_size, stride=1,
                           threshold_percentile=95, n_samples=1024, device='cpu',
                           threshold_data=None, reduction='last',
                           norm_method='zscore'):
    """Run Donut on full dataset and return anomaly labels."""
    avg_scores = score_donut_timestamps(
        model, data, norm_params, window_size, stride, n_samples, device, reduction,
        norm_method=norm_method
    )

    calibration_data = data if threshold_data is None else threshold_data
    calibration_scores = score_donut_timestamps(
        model, calibration_data, norm_params, window_size, stride, n_samples,
        device, reduction, norm_method=norm_method
    )
    if reduction == 'last':
        calibration_scores = calibration_scores[window_size - 1:]
    threshold = np.percentile(calibration_scores, threshold_percentile)
    labels = (avg_scores > threshold).astype(int)

    return avg_scores, labels, threshold


def evaluate_with_labels(scores, labels, ground_truth):
    """
    Evaluate anomaly detection performance.
    ground_truth: binary array (1 = anomaly, 0 = normal), same length as scores.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    metrics = {
        'precision': float(precision_score(ground_truth, labels, zero_division=0)),
        'recall': float(recall_score(ground_truth, labels, zero_division=0)),
        'f1': float(f1_score(ground_truth, labels, zero_division=0)),
    }
    try:
        metrics['auc_roc'] = float(roc_auc_score(ground_truth, scores))
    except ValueError:
        metrics['auc_roc'] = None

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train anomaly detection models')
    parser.add_argument('--model', choices=['usad', 'donut', 'both'], default='both')
    parser.add_argument('--data', type=str, help='Path to CSV data file')
    parser.add_argument('--columns', nargs='*', help='Columns to use (default: all numeric)')
    parser.add_argument('--window', type=int, default=64, help='Sliding window size')
    parser.add_argument('--stride', type=int, default=1, help='Window stride')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--latent', type=int, default=10, help='Latent dimension')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold', type=float, default=95,
                        help='Percentile for anomaly threshold')
    parser.add_argument('--prometheus', action='store_true',
                        help='Parse as Prometheus CSV export')
    parser.add_argument('--ground-truth', type=str,
                        help='CSV with ground truth labels (1=anomaly, 0=normal)')
    parser.add_argument('--train-on-all', action='store_true',
                        help='Train on all rows instead of filtering label==0 rows')
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)
    device = args.device

    if not args.data:
        print("Error: --data argument required (path to CSV file)")
        sys.exit(1)

    # Load data
    print(f"\n{'='*60}")
    print(f"Loading data: {args.data}")
    print(f"{'='*60}")

    if args.prometheus:
        raw_data, feature_names = load_prometheus_csv(args.data)
    else:
        raw_data, feature_names = load_csv(args.data, columns=args.columns)

    n_timestamps, n_features = raw_data.shape
    print(f"Shape: {n_timestamps} timestamps x {n_features} features")
    print(f"Features: {feature_names}")

    # Load ground truth if provided, otherwise use an inline label column.
    gt_labels = None
    if args.ground_truth:
        gt_df = pd.read_csv(args.ground_truth)
        gt_col = gt_df.select_dtypes(include=[np.number]).columns[0]
        gt_labels = align_labels(gt_df[gt_col].values.astype(int), n_timestamps)
        print(f"Ground truth labels loaded: {gt_labels.sum()} anomaly points")
    elif not args.prometheus:
        gt_labels = load_labels_from_csv(args.data, n_timestamps)
        if gt_labels is not None:
            print(f"Inline labels loaded: {gt_labels.sum()} anomaly points")

    train_data = raw_data
    if gt_labels is not None and not args.train_on_all:
        normal_mask = gt_labels == 0
        train_data = raw_data[normal_mask]
        print(f"Training on normal rows only: {len(train_data)}/{len(raw_data)} rows")
    elif args.train_on_all:
        print("Training on all rows (--train-on-all enabled)")

    if len(train_data) < args.window:
        raise ValueError(
            f"Need at least {args.window} training rows after filtering, "
            f"got {len(train_data)}"
        )

    results = {}

    # ================================================================
    # Train / Run USAD
    # ================================================================
    if args.model in ('usad', 'both'):
        print(f"\n{'='*60}")
        print(f"Training USAD (KDD 2020)")
        print(f"{'='*60}")

        train_loader, val_loader, norm_params, norm_data, windows, fnames = \
            prepare_multivariate_data(
                {name: train_data[:, i] for i, name in enumerate(feature_names)},
                window_size=args.window, stride=args.stride, val_ratio=0.2,
                random_seed=args.seed
            )

        input_dim = args.window * n_features
        model_usad = USAD(args.window, n_features, latent_dim=args.latent)
        print(f"Model: USAD(input={input_dim}, latent={args.latent})")
        print(f"Train windows: {len(train_loader.dataset)}, Val windows: {len(val_loader.dataset)}")

        history = train_usad(model_usad, train_loader, val_loader,
                             epochs=args.epochs, lr=args.lr, device=device)

        # Detect anomalies
        scores_usad, labels_usad, thresh_usad = detect_anomalies_usad(
            model_usad, raw_data, norm_params, args.window, args.stride,
            threshold_percentile=args.threshold, device=device,
            threshold_data=train_data
        )

        results['usad'] = {
            'history': history,
            'scores': scores_usad.tolist(),
            'labels': labels_usad.tolist(),
            'threshold': float(thresh_usad),
            'n_anomalies': int(labels_usad.sum()),
        }

        if gt_labels is not None:
            results['usad']['metrics'] = evaluate_with_labels(
                scores_usad, labels_usad, gt_labels
            )
            print(f"USAD metrics: {json.dumps(results['usad']['metrics'], indent=2)}")

        print(f"USAD: {results['usad']['n_anomalies']} anomalies detected "
              f"(threshold={thresh_usad:.4f})")

        # Save model
        torch.save(model_usad.state_dict(),
                   os.path.join(args.output, 'usad_model.pt'))

    # ================================================================
    # Train / Run Donut
    # ================================================================
    if args.model in ('donut', 'both'):
        # For Donut, use the first feature (univariate)
        # Or train on each feature separately
        for feat_idx, feat_name in enumerate(feature_names):
            print(f"\n{'='*60}")
            print(f"Training Donut (WWW 2018) on [{feat_name}]")
            print(f"{'='*60}")

            univar_data = raw_data[:, feat_idx:feat_idx+1]
            univar_train_data = train_data[:, feat_idx:feat_idx+1]

            train_loader, val_loader, norm_params, norm_data, windows = prepare_data(
                univar_train_data, window_size=args.window, stride=args.stride,
                val_ratio=0.2, norm_method='zscore', model_type='donut',
                random_seed=args.seed
            )

            model_donut = Donut(args.window, latent_dim=min(args.latent, 5))
            print(f"Model: Donut(window={args.window}, latent={min(args.latent, 5)})")
            print(f"Train windows: {len(train_loader.dataset)}, "
                  f"Val windows: {len(val_loader.dataset)}")

            history = train_donut(model_donut, train_loader, val_loader,
                                  epochs=args.epochs, lr=args.lr, device=device)

            # Detect anomalies
            scores_donut, labels_donut, thresh_donut = detect_anomalies_donut(
                model_donut, univar_data, norm_params, args.window, args.stride,
                threshold_percentile=args.threshold, device=device,
                threshold_data=univar_train_data
            )

            key = f'donut_{feat_name}'
            results[key] = {
                'history': history,
                'scores': scores_donut.tolist(),
                'labels': labels_donut.tolist(),
                'threshold': float(thresh_donut),
                'n_anomalies': int(labels_donut.sum()),
            }

            if gt_labels is not None:
                results[key]['metrics'] = evaluate_with_labels(
                    scores_donut, labels_donut, gt_labels
                )
                print(f"Donut({feat_name}) metrics: "
                      f"{json.dumps(results[key]['metrics'], indent=2)}")

            print(f"Donut({feat_name}): {results[key]['n_anomalies']} anomalies "
                  f"(threshold={thresh_donut:.4f})")

            # Save model
            torch.save(model_donut.state_dict(),
                       os.path.join(args.output, f'donut_{feat_name}_model.pt'))

            if feat_idx >= 2:  # Limit to 3 features for Donut
                print(f"\n(limiting Donut to {feat_idx+1} features; "
                      f"add --columns to specify specific metrics)")
                break

    # ================================================================
    # Save results
    # ================================================================
    results['config'] = {
        'data': args.data,
        'features': feature_names,
        'window_size': args.window,
        'stride': args.stride,
        'epochs': args.epochs,
        'latent_dim': args.latent,
        'threshold_percentile': args.threshold,
        'n_timestamps': n_timestamps,
        'n_train_timestamps': int(len(train_data)),
        'trained_on_all_rows': bool(args.train_on_all or gt_labels is None),
        'timestamp': datetime.now().isoformat(),
    }

    results_path = os.path.join(args.output, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {results_path}")
    print(f"Models saved to {args.output}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
