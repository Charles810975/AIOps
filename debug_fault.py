#!/usr/bin/env python3
"""Debug: check if the fault actually caused metric changes."""

import pandas as pd

normal = pd.read_csv("data-collection/normal_metrics.csv")
anomaly = pd.read_csv("data-collection/cart_cpu_anomaly.csv")

# Filter to cartservice pods
cart_normal = normal[normal["pod"].str.contains("cartservice", na=False)]
cart_anomaly = anomaly[anomaly["pod"].str.contains("cartservice", na=False)]

print("=== CartService Pods Found ===")
print("Normal pods:", cart_normal["pod"].unique())
print("Anomaly pods:", cart_anomaly["pod"].unique())

print("\n=== Metric count per pod in normal ===")
print(normal.groupby("pod")["metric"].nunique())

print("\n=== CPU usage: normal vs anomaly for ALL pods ===")
cpu_cols = ["pod", "metric", "value"]
normal_cpu = normal[normal["metric"] == "cpu_usage"]
anomaly_cpu = anomaly[anomaly["metric"] == "cpu_usage"]

merged = normal_cpu[["pod", "timestamp", "value"]].merge(
    anomaly_cpu[["pod", "timestamp", "value"]],
    on=["pod", "timestamp"],
    suffixes=("_normal", "_anomaly")
)
merged["change"] = merged["value_anomaly"] - merged["value_normal"]
merged["change_pct"] = (merged["change"] / merged["value_normal"].replace(0, 1e-9)) * 100

print("Top 5 pods by mean CPU change (absolute increase):")
summary = merged.groupby("pod").agg(
    mean_normal=("value_normal", "mean"),
    mean_anomaly=("value_anomaly", "mean"),
    mean_change=("change", "mean"),
    max_change=("change", "max"),
).sort_values("max_change", ascending=False)
print(summary.head(10).round(6).to_string())

print("\n=== CPU throttle ratio: normal vs anomaly for ALL pods ===")
normal_thr = normal[normal["metric"] == "cpu_throttle_ratio"]
anomaly_thr = anomaly[anomaly["metric"] == "cpu_throttle_ratio"]

print(f"Normal throttle - non-zero values: {(normal_thr['value'] > 0).sum()} / {len(normal_thr)}")
print(f"Anomaly throttle - non-zero values: {(anomaly_thr['value'] > 0).sum()} / {len(anomaly_thr)}")

# Show cartservice throttle specifically
cart_thr_norm = normal_thr[normal_thr["pod"].str.contains("cartservice", na=False)]["value"]
cart_thr_ano = anomaly_thr[anomaly_thr["pod"].str.contains("cartservice", na=False)]["value"]
print(f"\nCartservice throttle - normal: mean={cart_thr_norm.mean():.6f}, max={cart_thr_norm.max():.6f}")
print(f"Cartservice throttle - anomaly: mean={cart_thr_ano.mean():.6f}, max={cart_thr_ano.max():.6f}")

# Show all pods throttle in anomaly
print("\nAll pods mean cpu_throttle_ratio in anomaly period:")
print(anomaly_thr.groupby("pod")["value"].mean().sort_values(ascending=False).round(6).to_string())

print("\n=== Time range check ===")
print(f"Normal: {normal['timestamp'].min()} - {normal['timestamp'].max()}")
print(f"Anomaly: {anomaly['timestamp'].min()} - {anomaly['timestamp'].max()}")
print(f"Normal duration: {(normal['timestamp'].max() - normal['timestamp'].min())/60:.1f} minutes")
print(f"Anomaly duration: {(anomaly['timestamp'].max() - anomaly['timestamp'].min())/60:.1f} minutes")
