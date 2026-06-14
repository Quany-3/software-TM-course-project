#!/usr/bin/env python3
"""
Prometheus data collector: collects container metrics during normal ops and chaos faults.

Usage:
  python metrics_collector.py --baseline 300 --fault 120 --rounds 3 --output data/prometheus_data
"""

import argparse
import os
import sys
import time
import json
import tempfile
import subprocess
from datetime import datetime

import requests
import pandas as pd

PROM_URL = "http://localhost:31090"
INTERVAL = 15


def prom_query(promql):
    resp = requests.get(f"{PROM_URL}/api/v1/query", params={"query": promql}, timeout=30)
    resp.raise_for_status()
    return resp.json()["data"]["result"]


def get_pod_metrics():
    """Get per-pod CPU + memory for default namespace pods."""
    print("Discovering pods in default namespace...")
    pods = prom_query(
        'container_cpu_usage_seconds_total'
        '{namespace="default",pod!=""}'
    )
    pod_names = sorted(set(
        r["metric"]["pod"] for r in pods
        if r["metric"]["pod"] not in ("", "ubuntu")
    ))
    print(f"  Found {len(pod_names)} pods: {pod_names}")

    queries = {}
    for pod in pod_names:
        safe = pod.replace("-", "_")
        queries[f"cpu_{safe}"] = (
            f'rate(container_cpu_usage_seconds_total'
            f'{{namespace="default",pod="{pod}"}}[1m])'
        )
        queries[f"mem_{safe}"] = (
            f'container_memory_working_set_bytes'
            f'{{namespace="default",pod="{pod}"}}'
        )

    # Add aggregate metrics
    queries["pod_count_running"] = (
        'count(kube_pod_status_phase{namespace="default",phase="Running"})'
    )
    queries["node_cpu_pct"] = (
        '100 - avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100'
    )

    print(f"  Total queries: {len(queries)}")
    return queries


CHAOS_EXPERIMENTS = [
    {
        "name": "pod-kill-frontend",
        "kind": "PodChaos",
        "spec": {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "kind": "PodChaos",
            "metadata": {"name": "pod-kill-frontend", "namespace": "chaos-testing"},
            "spec": {
                "action": "pod-kill",
                "mode": "one",
                "selector": {
                    "namespaces": ["default"],
                    "labelSelectors": {"app": "frontend"},
                },
                "duration": "120s",
            },
        },
    },
    {
        "name": "pod-kill-random",
        "kind": "PodChaos",
        "spec": {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "kind": "PodChaos",
            "metadata": {"name": "pod-kill-random", "namespace": "chaos-testing"},
            "spec": {
                "action": "pod-kill",
                "mode": "one",
                "selector": {"namespaces": ["default"]},
                "duration": "60s",
            },
        },
    },
    {
        "name": "cpu-stress",
        "kind": "StressChaos",
        "spec": {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "kind": "StressChaos",
            "metadata": {"name": "cpu-stress", "namespace": "chaos-testing"},
            "spec": {
                "mode": "one",
                "selector": {"namespaces": ["default"]},
                "stressors": {"cpu": {"workers": 2, "load": 80}},
                "duration": "90s",
            },
        },
    },
    {
        "name": "memory-stress",
        "kind": "StressChaos",
        "spec": {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "kind": "StressChaos",
            "metadata": {"name": "memory-stress", "namespace": "chaos-testing"},
            "spec": {
                "mode": "one",
                "selector": {"namespaces": ["default"]},
                "stressors": {"memory": {"workers": 2, "size": "256MB"}},
                "duration": "90s",
            },
        },
    },
    {
        "name": "net-delay",
        "kind": "NetworkChaos",
        "spec": {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "kind": "NetworkChaos",
            "metadata": {"name": "net-delay", "namespace": "chaos-testing"},
            "spec": {
                "action": "delay",
                "mode": "one",
                "selector": {"namespaces": ["default"]},
                "delay": {"latency": "300ms", "jitter": "50ms"},
                "duration": "60s",
            },
        },
    },
]


def apply_chaos(exp):
    """Apply chaos and return True on success."""
    name = exp["spec"]["metadata"]["name"]
    kind = exp["kind"].lower()
    ns = exp["spec"]["metadata"]["namespace"]

    subprocess.run(
        f"kubectl delete {kind} {name} -n {ns} --ignore-not-found",
        shell=True, capture_output=True,
    )
    time.sleep(2)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix=f"chaos_{name}_"
    ) as f:
        json.dump(exp["spec"], f)
        tmp = f.name

    proc = subprocess.run(
        f"kubectl create -f {tmp}", shell=True, capture_output=True, text=True
    )
    os.unlink(tmp)

    ok = proc.returncode == 0
    if ok:
        print(f"  [CHAOS] {exp['name']} applied")
    else:
        print(f"  [CHAOS] {exp['name']} FAILED: {proc.stderr.strip()}")
    return ok


def delete_chaos(exp):
    name = exp["spec"]["metadata"]["name"]
    kind = exp["kind"].lower()
    ns = exp["spec"]["metadata"]["namespace"]
    subprocess.run(
        f"kubectl delete {kind} {name} -n {ns} --ignore-not-found",
        shell=True, capture_output=True,
    )
    print(f"  [CHAOS] {exp['name']} deleted")


def collect_period(queries, duration, label, output_file, interval=INTERVAL):
    """Collect Prometheus data for a period, append to CSV."""
    records = []
    n = duration // interval
    start = time.time()

    for i in range(n):
        row = {"timestamp": int(time.time()), "label": label}
        for name, promql in queries.items():
            try:
                results = prom_query(promql)
                for r in results:
                    val = float(r["value"][1])
                    if len(results) == 1:
                        row[name] = val
                    else:
                        pod = r["metric"].get("pod", "unknown").replace("-", "_")
                        row[f"{name}_{pod}"] = val
            except Exception:
                pass
        records.append(row)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            print(f"    [{label}] {i+1}/{n} ({elapsed:.0f}s elapsed)")

    df = pd.DataFrame(records)
    if os.path.exists(output_file):
        existing = pd.read_csv(output_file)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(output_file, index=False)
    return len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=int, default=300, help="Baseline duration (s)")
    parser.add_argument("--fault", type=int, default=120, help="Fault duration (s)")
    parser.add_argument("--recovery", type=int, default=60, help="Recovery wait (s)")
    parser.add_argument("--rounds", type=int, default=3, help="Fault rounds")
    parser.add_argument("--interval", type=int, default=INTERVAL,
                        help=f"Scrape interval in seconds (default: {INTERVAL})")
    parser.add_argument("--output", type=str, default="data/prometheus_data")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output, ts)
    os.makedirs(out_dir, exist_ok=True)
    metrics_file = os.path.join(out_dir, "metrics.csv")

    # Discover metrics
    queries = get_pod_metrics()
    if not queries:
        print("ERROR: No metrics found")
        sys.exit(1)

    # Save config
    config = {
        "baseline": args.baseline, "fault": args.fault,
        "recovery": args.recovery, "rounds": args.rounds,
        "experiments": [e["name"] for e in CHAOS_EXPERIMENTS[:args.rounds]],
        "n_queries": len(queries),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Data Collection: {args.baseline}s baseline + "
          f"{args.rounds} x {args.fault}s faults")
    print(f"Output: {out_dir}")
    print(f"{'='*60}\n")

    total_rounds = args.rounds + 1  # baseline + faults

    # Baseline
    round_num = 1
    print(f"[{round_num}/{total_rounds}] BASELINE (normal) — {args.baseline}s")
    n = collect_period(queries, args.baseline, label=0, output_file=metrics_file,
                        interval=args.interval)
    print(f"  Collected {n} samples\n")

    # Fault rounds
    for i in range(args.rounds):
        round_num = i + 2
        exp = CHAOS_EXPERIMENTS[i % len(CHAOS_EXPERIMENTS)]

        print(f"[{round_num}/{total_rounds}] FAULT: {exp['name']} — {args.fault}s")
        if apply_chaos(exp):
            time.sleep(10)  # Let chaos take effect
        n = collect_period(queries, args.fault, label=1, output_file=metrics_file,
                           interval=args.interval)
        print(f"  Collected {n} samples")
        delete_chaos(exp)

        if i < args.rounds - 1:
            print(f"  Recovery: {args.recovery}s\n")
            # Collect recovery as anomaly (system still unstable during recovery)
            collect_period(queries, args.recovery, label=1, output_file=metrics_file,
                           interval=args.interval)
            time.sleep(5)

    # Summary
    df = pd.read_csv(metrics_file)
    features = [c for c in df.columns if c not in ("timestamp", "label")]
    n0 = (df["label"] == 0).sum()
    n1 = (df["label"] == 1).sum()

    # Clean: drop columns that are all NaN
    df_clean = df.dropna(axis=1, how="all")
    df_clean.to_csv(metrics_file, index=False)

    labels_df = df_clean[["label"]].copy()
    labels_path = os.path.join(out_dir, "labels.csv")
    labels_df.to_csv(labels_path, index=False)

    print(f"\n{'='*60}")
    print(f"Done! {len(df_clean)} samples ({n0} normal, {n1} anomaly)")
    print(f"Features: {len(features)}")
    print(f"Metrics:  {metrics_file}")
    print(f"Labels:   {labels_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
