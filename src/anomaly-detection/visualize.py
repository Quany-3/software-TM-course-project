#!/usr/bin/env python3
"""
LightCAE-style visualization for USAD (KDD20) and Donut (WWW18).
Each algorithm gets a 3-panel figure showing the reconstruction mechanism:

  Panel 1 — Original Metrics (M) with detected anomalies highlighted
  Panel 2 — Original (gray dashed) vs Reconstructed Normal C (green): THE core insight
  Panel 3 — Anomaly Score ||M-C||^2 with threshold

Usage:
    python visualize.py --data data/prometheus_data/20260614_133243/metrics.csv
"""

import argparse, os, sys, json, math
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from models import USAD, Donut, train_usad, train_donut, gaussian_log_prob
from data_utils import prepare_data, prepare_multivariate_data, create_windows, apply_normalize
from run import (
    load_csv, load_labels_from_csv, score_usad_timestamps, score_donut_timestamps,
    detect_anomalies_usad, detect_anomalies_donut, evaluate_with_labels
)


# ──────────────────────────────────────────
# Utility: full-timeline reconstruction
# ──────────────────────────────────────────
def usad_full_reconstruction(model, data, norm_params, window_size, device='cpu'):
    """Get USAD AE1 reconstruction for the full timeline, denormalized."""
    model.eval()
    model = model.to(device)
    n_features = data.shape[1]
    normalized = apply_normalize(data, norm_params)
    windows = create_windows(normalized, window_size, stride=1)
    windows_t = torch.FloatTensor(windows).to(device)
    windows_t = windows_t.reshape(windows_t.size(0), -1)

    with torch.no_grad():
        z = model.encoder(windows_t)
        w1 = model.decoder1(z)  # (n_windows, input_dim)

    w1_np = w1.cpu().numpy()
    recons = np.zeros((len(normalized), n_features))
    counts = np.zeros((len(normalized), n_features))

    for i in range(len(w1_np)):
        w1_reshaped = w1_np[i].reshape(window_size, n_features)
        start, end = i, i + window_size
        recons[start:end] += w1_reshaped
        counts[start:end] += 1

    counts[counts == 0] = 1
    recons /= counts

    # De-normalize
    d_min, d_max = norm_params[0], norm_params[1]
    recon_denorm = recons * (d_max - d_min) + d_min
    return recon_denorm


def donut_full_reconstruction(model, data, norm_params, window_size, device='cpu'):
    """Get Donut VAE reconstruction (mu_x) for the full timeline, denormalized."""
    model.eval()
    model = model.to(device)
    normalized = apply_normalize(data.reshape(-1, 1), norm_params).flatten()
    windows = create_windows(normalized, window_size, stride=1)
    windows_t = torch.FloatTensor(windows).to(device)

    with torch.no_grad():
        mu_z, logvar_z = model.encoder(windows_t)
        z = model.reparameterize(mu_z, logvar_z)
        mu_x, logvar_x = model.decoder(z)

    mu_x_np = mu_x.cpu().numpy()
    recons = np.zeros(len(normalized))
    counts = np.zeros(len(normalized))

    for i in range(len(mu_x_np)):
        start, end = i, i + window_size
        recons[start:end] += mu_x_np[i]
        counts[start:end] += 1

    counts[counts == 0] = 1
    recons /= counts

    # De-normalize
    d_min = float(norm_params[0].item() if hasattr(norm_params[0], 'item') else norm_params[0])
    d_max = float(norm_params[1].item() if hasattr(norm_params[1], 'item') else norm_params[1])
    result = recons * (d_max - d_min) + d_min
    return result.ravel()  # ensure 1-D


# ──────────────────────────────────────────
# USAD — LightCAE-style 3 panels
# ──────────────────────────────────────────
def zscore_normalize(data, normal_mask):
    """Z-score normalize using mean/std from normal (label=0) data only."""
    mean = data[normal_mask].mean(axis=0, keepdims=True)
    std = data[normal_mask].std(axis=0, keepdims=True)
    std[std == 0] = 1.0  # avoid div by zero
    return (data - mean) / std


def pick_display_features(raw_data, feature_names, gt_labels):
    """Pick the 2 features with highest variance ratio (anomaly vs normal) for best visual effect."""
    normal_mask = gt_labels == 0
    ratios = []
    for i, name in enumerate(feature_names):
        n_std = raw_data[normal_mask, i].std()
        a_std = raw_data[~normal_mask, i].std()
        ratio = (a_std / (n_std + 1e-8)) if n_std > 1e-8 else 1.0
        ratios.append((ratio, i, name))
    ratios.sort(reverse=True)
    best = ratios[:2]
    idx1, idx2 = best[0][1], best[1][1]
    feat1, feat2 = best[0][2], best[1][2]
    return idx1, idx2, feat1, feat2


def plot_usad(data_df, raw_data, norm_params, model, scores, labels, threshold,
              metrics, gt_labels, feature_names, window_size, output_dir):
    plt.style.use('seaborn-v0_8-darkgrid')

    n_points = len(scores)
    time_axis = np.arange(n_points)
    anomalies_idx = np.where(labels == 1)[0]
    normal_mask = gt_labels == 0

    # Pick best 2 features based on anomaly/normal variance ratio
    idx1, idx2, feat1, feat2 = pick_display_features(raw_data, feature_names, gt_labels)

    # Compute full-timeline reconstruction
    recon_full = usad_full_reconstruction(model, raw_data, norm_params, window_size)

    # Z-score normalize for display — makes reconstruction differences clearly visible
    raw_z = zscore_normalize(raw_data, normal_mask)
    recon_z = zscore_normalize(recon_full, normal_mask)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # ── Panel 1: Original Metrics (M) in z-score ──
    ax1.plot(time_axis, raw_z[:, idx1], label=feat1[:40], color='blue', linewidth=1)
    ax1.plot(time_axis, raw_z[:, idx2], label=feat2[:40], color='orange', linewidth=1)
    ax1.scatter(anomalies_idx, raw_z[anomalies_idx, idx1],
                color='red', s=50, label='Detected Anomaly', zorder=5)
    ax1.set_ylabel('Z-score')
    ax1.set_title('Microservice System Metrics (M) — Z-score Normalized')
    ax1.legend(loc='upper left')

    # ── Panel 2: Reconstructed Normal (C) in z-score ──
    ax2.plot(time_axis, raw_z[:, idx1], label=f'Original {feat1[:30]} (M)',
             color='lightgray', linestyle='--', linewidth=1)
    ax2.plot(time_axis, recon_z[:, idx1], label='Reconstructed Normal (C)',
             color='green', linewidth=1.5)
    # Highlight anomaly regions where reconstruction deviates
    ax2.set_ylabel('Z-score')
    ax2.set_title('Extracted Normal Data Subspace (C) = D(E(M))  — Z-score Normalized')
    ax2.legend(loc='upper left')

    # ── Panel 3: Anomaly Score (log scale) ──
    # Use log scale because anomaly scores span 3 orders of magnitude
    scores_clipped = np.maximum(scores, 1e-8)  # avoid log(0)
    ax3.semilogy(time_axis, scores_clipped, label='Anomaly Score (||M-C||^2)', color='purple')
    ax3.axhline(y=threshold, color='r', linestyle='--',
                label=f'Threshold ({threshold:.4f})')
    # Fill area above threshold (on log scale)
    ax3.fill_between(time_axis, threshold, scores_clipped, where=scores > threshold,
                      color='red', alpha=0.15)
    ax3.set_ylabel('Anomaly Score (log scale)')
    ax3.set_xlabel('Timestamp Index')
    ax3.set_title(
        f'Anomaly Score & Detection Result  |  '
        f'P={metrics.get("precision", 0):.3f}  '
        f'R={metrics.get("recall", 0):.3f}  '
        f'F1={metrics.get("f1", 0):.3f}  '
        f'AUC={metrics.get("auc_roc", 0):.3f}'
    )
    ax3.legend(loc='upper left')

    plt.tight_layout()
    path = os.path.join(output_dir, 'usad_result.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────
# Donut — LightCAE-style 3 panels
# ──────────────────────────────────────────
def plot_donut(data_df, univar_data, norm_params, model, scores, labels, threshold,
               metrics, gt_labels, feat_name, window_size, output_dir):
    plt.style.use('seaborn-v0_8-darkgrid')

    n_points = len(scores)
    time_axis = np.arange(n_points)
    anomalies_idx = np.where(labels == 1)[0]
    normal_mask = gt_labels == 0
    raw_vals = univar_data.flatten()

    # Compute full-timeline VAE reconstruction
    recon_full = donut_full_reconstruction(model, univar_data, norm_params, window_size)

    # Z-score normalize for display
    n_mean = raw_vals[normal_mask].mean()
    n_std = raw_vals[normal_mask].std()
    if n_std == 0:
        n_std = 1.0
    raw_z = (raw_vals - n_mean) / n_std
    recon_z = (recon_full - n_mean) / n_std

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # ── Panel 1: Original Metric (M) in z-score ──
    ax1.plot(time_axis, raw_z, label=feat_name[:40], color='blue', linewidth=1)
    ax1.scatter(anomalies_idx, raw_z[anomalies_idx],
                color='red', s=50, label='Detected Anomaly', zorder=5)
    ax1.set_ylabel('Z-score')
    ax1.set_title(f'KPI: {feat_name[:50]} (M) — Z-score Normalized')
    ax1.legend(loc='upper left')

    # ── Panel 2: Reconstructed Normal (C) in z-score ──
    ax2.plot(time_axis, raw_z, label=f'Original {feat_name[:30]} (M)',
             color='lightgray', linestyle='--', linewidth=1)
    ax2.plot(time_axis, recon_z, label='VAE Reconstruction (C = p(x|z))',
             color='green', linewidth=1.5)
    ax2.set_ylabel('Z-score')
    ax2.set_title('Extracted Normal Data Subspace (C) — VAE Decoder Output  — Z-score Normalized')
    ax2.legend(loc='upper left')

    # ── Panel 3: Anomaly Score ──
    ax3.plot(time_axis, scores, label='Anomaly Score (-Recon Prob)', color='purple')
    ax3.axhline(y=threshold, color='r', linestyle='--',
                label=f'Threshold ({threshold:.4f})')
    ax3.fill_between(time_axis, threshold, scores, where=scores > threshold,
                      color='red', alpha=0.15)
    ax3.set_ylabel('Anomaly Score')
    ax3.set_xlabel('Timestamp Index')
    ax3.set_title(
        f'Anomaly Score & Detection Result  |  '
        f'P={metrics.get("precision", 0):.3f}  '
        f'R={metrics.get("recall", 0):.3f}  '
        f'F1={metrics.get("f1", 0):.3f}  '
        f'AUC={metrics.get("auc_roc", 0):.3f}'
    )
    ax3.legend(loc='upper left')

    plt.tight_layout()
    safe_name = feat_name.replace('/', '_').replace('\\', '_')[:40]
    path = os.path.join(output_dir, f'donut_{safe_name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────
# Training Loss
# ──────────────────────────────────────────
def plot_training_loss(history_usad, history_donut_all, output_dir):
    plt.style.use('seaborn-v0_8-darkgrid')
    n_donut = len(history_donut_all)
    n_total = 1 + n_donut
    fig, axes = plt.subplots(1, n_total, figsize=(6 * n_total, 5))
    if n_total == 1:
        axes = [axes]

    ax = axes[0]
    epochs = [h['epoch'] for h in history_usad]
    ax.plot(epochs, [h['train_loss1'] for h in history_usad], 'b-', label='Train AE1 (recon)')
    ax.plot(epochs, [h['train_loss2'] for h in history_usad], 'r-', label='Train AE2 (adversarial)')
    ax.plot(epochs, [h['val_loss1'] for h in history_usad], 'b--', label='Val AE1')
    ax.plot(epochs, [h['val_loss2'] for h in history_usad], 'r--', label='Val AE2')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('USAD Training')
    ax.legend(fontsize=8)

    for idx, (feat_name, hist) in enumerate(history_donut_all.items()):
        ax = axes[idx + 1]
        epochs = [h['epoch'] for h in hist]
        ax.plot(epochs, [h['train_loss'] for h in hist], 'b-', label='Train M-ELBO')
        ax.plot(epochs, [h['val_loss'] for h in hist], 'r--', label='Val M-ELBO')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('M-ELBO Loss')
        ax.set_title(f'Donut: {feat_name[:35]}')
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'training_loss.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--window', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--donut-features', type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    print(f"\n{'='*60}")
    print(f"Loading: {args.data}")
    print(f"{'='*60}")

    raw_data, feature_names = load_csv(args.data)
    df = pd.read_csv(args.data)
    n_timestamps, n_features = raw_data.shape
    print(f"Shape: {n_timestamps} x {n_features}")

    gt_labels = load_labels_from_csv(args.data, n_timestamps)
    if gt_labels is None:
        print("ERROR: No 'label' column found")
        sys.exit(1)
    print(f"Labels: {gt_labels.sum()} anomaly, {(1 - gt_labels).sum()} normal")

    normal_mask = gt_labels == 0
    train_data = raw_data[normal_mask]
    print(f"Train: {len(train_data)} normal rows")

    # Feature selection
    data_dir = os.path.dirname(args.data)
    sf_json = os.path.join(data_dir, 'selected_features.json')
    if os.path.exists(sf_json):
        with open(sf_json) as f:
            sel = json.load(f)
        top_feat = [c for c in sel['selected_features'] if c in feature_names][:10]
    else:
        top_feat = feature_names[:10]
    print(f"Features: {len(top_feat)} selected")

    feat_indices = [feature_names.index(f) for f in top_feat if f in feature_names]
    usad_data = raw_data[:, feat_indices]
    usad_train = train_data[:, feat_indices]

    # ─── USAD ───
    print(f"\n{'='*60}")
    print(f"USAD (KDD 2020)")
    print(f"{'='*60}")

    train_loader, val_loader, usad_norm, _, _, _ = prepare_multivariate_data(
        {name: usad_train[:, i] for i, name in enumerate(top_feat)},
        window_size=args.window, stride=1, val_ratio=0.2, random_seed=args.seed
    )
    model_usad = USAD(args.window, len(top_feat), latent_dim=10)
    print(f"USAD(input={args.window * len(top_feat)}, latent=10)")
    history_usad = train_usad(model_usad, train_loader, val_loader,
                               epochs=args.epochs, lr=1e-3, device=device)

    scores_usad, labels_usad, thresh_usad = detect_anomalies_usad(
        model_usad, usad_data, usad_norm, args.window, 1,
        threshold_percentile=95, device=device, threshold_data=usad_train
    )
    usad_metrics = evaluate_with_labels(scores_usad, labels_usad, gt_labels)
    print(f"USAD: P={usad_metrics['precision']:.4f} R={usad_metrics['recall']:.4f} "
          f"F1={usad_metrics['f1']:.4f} AUC={usad_metrics['auc_roc']:.4f}")

    # ─── Donut ───
    print(f"\n{'='*60}")
    print(f"Donut (WWW 2018)")
    print(f"{'='*60}")

    donut_results = {}
    history_donut_all = {}

    for feat_name in top_feat[:args.donut_features]:
        print(f"\n--- {feat_name[:50]} ---")
        feat_idx = feature_names.index(feat_name)
        univar_data = raw_data[:, feat_idx:feat_idx + 1]
        univar_train = train_data[:, feat_idx:feat_idx + 1]

        train_loader, val_loader, d_norm, _, _ = prepare_data(
            univar_train, window_size=args.window, stride=1,
            val_ratio=0.2, norm_method='minmax', model_type='donut',
            random_seed=args.seed
        )
        model_d = Donut(args.window, latent_dim=5)
        hist = train_donut(model_d, train_loader, val_loader,
                           epochs=args.epochs, lr=1e-3, device=device)
        scores_d, labels_d, thresh_d = detect_anomalies_donut(
            model_d, univar_data, d_norm, args.window, 1,
            threshold_percentile=95, device=device, threshold_data=univar_train,
            norm_method='minmax'
        )
        metrics_d = evaluate_with_labels(scores_d, labels_d, gt_labels)
        print(f"Donut: P={metrics_d['precision']:.4f} R={metrics_d['recall']:.4f} "
              f"F1={metrics_d['f1']:.4f} AUC={metrics_d['auc_roc']:.4f}")

        donut_results[feat_name] = {
            'model': model_d, 'norm': d_norm, 'data': univar_data,
            'scores': scores_d, 'labels': labels_d, 'threshold': thresh_d,
            'metrics': metrics_d,
        }
        history_donut_all[feat_name] = hist

    # ─── Plots ───
    print(f"\n{'='*60}")
    print(f"Generating LightCAE-style plots...")
    print(f"{'='*60}")

    plot_usad(df, usad_data, usad_norm, model_usad, scores_usad, labels_usad,
              thresh_usad, usad_metrics, gt_labels, top_feat, args.window, args.output)

    for feat_name, res in donut_results.items():
        plot_donut(df, res['data'], res['norm'], res['model'], res['scores'],
                   res['labels'], res['threshold'], res['metrics'],
                   gt_labels, feat_name, args.window, args.output)

    plot_training_loss(history_usad, history_donut_all, args.output)

    # JSON
    results = {
        'usad': {'metrics': usad_metrics, 'threshold': float(thresh_usad),
                  'n_anomalies': int(labels_usad.sum())},
        'donut': {n: {'metrics': r['metrics'], 'threshold': float(r['threshold'])}
                   for n, r in donut_results.items()},
        'config': {'data': args.data, 'window': args.window, 'epochs': args.epochs,
                   'timestamp': datetime.now().isoformat()},
    }
    with open(os.path.join(args.output, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! Output: {args.output}/")
    print(f"  usad_result.png")
    for fn in donut_results:
        print(f"  donut_{fn[:30]}.png")
    print(f"  training_loss.png")
    print(f"  results.json")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
