import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler


def zscore(series):
    arr = np.asarray(series, dtype=float)
    std = np.nanstd(arr)
    if std < 1e-8:
        return np.zeros_like(arr)
    return (arr - np.nanmean(arr)) / std


def sax_encode(values, alphabet_size=5):
    values = zscore(values)
    breakpoints = np.linspace(0, 1, alphabet_size + 1)[1:-1]
    quantiles = np.quantile(values, breakpoints)
    return np.digitize(values, quantiles)


def normalized_mutual_trend(a, b):
    if len(a) < 3 or len(b) < 3:
        return 0.0
    da = np.diff(a)
    db = np.diff(b)
    return float(np.mean(np.sign(da) == np.sign(db)))


def lagged_causality_score(candidate, target, max_lag=5):
    candidate = zscore(candidate)
    target = zscore(target)
    best = 0.0
    best_lag = 0
    for lag in range(1, max_lag + 1):
        if len(candidate) <= lag + 3:
            continue
        x = candidate[:-lag]
        y = target[lag:]
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            corr = 0.0
        else:
            corr = abs(pearsonr(x, y)[0])
        if corr > best:
            best = corr
            best_lag = lag
    return best, best_lag


def prepare_matrix(data):
    data = data.copy()
    data["series"] = data["pod"] + "::" + data["metric"]
    pivot = data.pivot_table(
        index="timestamp", columns="series", values="value", aggfunc="mean"
    ).sort_index()
    pivot = pivot.interpolate(limit_direction="both").fillna(0.0)
    return pivot


def build_target_from_delta(pivot_normal, pivot_anomaly):
    """Build target signal from per-metric delta (anomaly - normal).

    Each column: compute mean value during anomaly vs normal,
    take the absolute delta, z-score across series,
    then build per-timestep global anomaly signal.
    """
    # Align on shared timestamps (outer join, fill missing with 0)
    combined = pivot_normal.join(pivot_anomaly, lsuffix="_normal", rsuffix="_anomaly", how="outer")
    combined = combined.fillna(0.0)

    normal_cols = [c for c in combined.columns if c.endswith("_normal")]
    anomaly_cols = [c for c in combined.columns if c.endswith("_anomaly")]

    delta_matrix = pd.DataFrame(index=combined.index)
    for n_col, a_col in zip(normal_cols, anomaly_cols):
        delta_matrix[n_col.replace("_normal", "")] = (
            combined[a_col].values - combined[n_col].values
        )

    # Global target: mean absolute delta across all series at each timestep
    target = np.mean(np.abs(delta_matrix.values), axis=1)
    return target, delta_matrix


def align_series(values, target):
    """Trim longer array so it matches the shorter one for element-wise ops."""
    min_len = min(len(values), len(target))
    return values[:min_len], target[:min_len]


def compute_anomaly_detection_f1(result_df, ground_truth_pod_prefix):
    """Stage 1: Anomaly Detection F1 Score.

    Treat every (pod, metric) candidate as a binary classifier:
      - Positive  = KPI belongs to the ground-truth fault service
      - Negative  = KPI belongs to any other service
    Threshold at root_score >= threshold (sweep thresholds to find best F1).

    Returns the best F1, Precision, Recall at that threshold,
    and the threshold value itself.
    """
    result_df = result_df.copy()

    # Ground-truth label: 1 if the KPI's pod starts with ground_truth prefix
    result_df["y_true"] = result_df["pod"].str.startswith(ground_truth_pod_prefix).astype(int)

    y_true = result_df["y_true"].values
    y_score = result_df["root_score"].values

    best_f1 = 0.0
    best_p = 0.0
    best_r = 0.0
    best_thresh = 0.0

    # Sweep thresholds across the score range
    thresholds = np.sort(np.unique(y_score))
    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        if f1 >= best_f1:
            best_f1 = f1
            best_p = p
            best_r = r
            best_thresh = thresh

    # Also compute at threshold = 0 (all predicted negative → P=0,R=0,F1=0)
    return {
        "F1_Score": round(best_f1, 4),
        "Precision": round(best_p, 4),
        "Recall": round(best_r, 4),
        "Threshold": round(best_thresh, 4),
        "TP": int(np.sum((y_score >= best_thresh) & (y_true == 1))),
        "FP": int(np.sum((y_score >= best_thresh) & (y_true == 0))),
        "FN": int(np.sum((y_score < best_thresh) & (y_true == 1))),
    }


def compute_hit_at_k(result_df, ground_truth_pod_prefix, top_k_values):
    """Stage 2: Root Cause Localization Hit@K.

    Hit@K = 1 if at least one ground-truth KPI appears in the top-K ranking.
    This is the standard metric used in the paper (Table II/III).
    """
    gt_kpis = set(result_df[result_df["pod"].str.startswith(ground_truth_pod_prefix)].index)

    rows = []
    for k in top_k_values:
        top_k_indices = set(result_df.head(k).index)
        hit = int(len(gt_kpis & top_k_indices) > 0)
        rows.append({"K": k, "Hit@K": hit})

    return pd.DataFrame(rows)


def evaluate_ranking(result_df, ground_truth_pod_prefix, top_k_values=None):
    """Two-stage evaluation matching the paper (ISSRE24).

    Stage 1 – Anomaly Detection F1 Score:
      Threshold-swept binary classification over all (pod, metric) KPIs.
      Positive = belongs to the ground-truth fault service.

    Stage 2 – Root Cause Localization Hit@K:
      Check whether the top-K ranked KPIs contain at least one true root cause.
    """
    if top_k_values is None:
        top_k_values = [1, 3, 5, 10]

    result_df = result_df.copy().reset_index(drop=True)

    gt_kpis = {idx for idx, row in result_df.iterrows()
               if row["pod"].startswith(ground_truth_pod_prefix)}
    print(f"\n  Ground-truth ({ground_truth_pod_prefix}): {len(gt_kpis)} KPI(s)")
    print(f"  Total candidate KPIs: {len(result_df)}")

    # Stage 1: Anomaly Detection F1
    print(f"\n  {'='*50}")
    print(f"  Stage 1 – Anomaly Detection F1 Score (threshold-swept)")
    print(f"  {'='*50}")
    f1_metrics = compute_anomaly_detection_f1(result_df, ground_truth_pod_prefix)
    print(f"  F1 Score : {f1_metrics['F1_Score']:.4f}")
    print(f"  Precision: {f1_metrics['Precision']:.4f}")
    print(f"  Recall   : {f1_metrics['Recall']:.4f}")
    print(f"  Best thr : {f1_metrics['Threshold']:.4f}")
    print(f"  TP={f1_metrics['TP']}  FP={f1_metrics['FP']}  FN={f1_metrics['FN']}")

    # Stage 2: Hit@K
    print(f"\n  {'='*50}")
    print(f"  Stage 2 – Root Cause Localization Hit@K")
    print(f"  {'='*50}")
    hit_df = compute_hit_at_k(result_df, ground_truth_pod_prefix, top_k_values)
    for _, row in hit_df.iterrows():
        print(f"  Hit@{int(row['K']):2d}: {int(row['Hit@K'])}")

    return f1_metrics, hit_df


def plot_ranking_with_gt(result_df, ground_truth_prefix, output_path):
    """Plot ranking bar chart with ground-truth pods highlighted."""
    top = result_df.head(20).copy()
    top = top.iloc[::-1]

    colors = []
    for pod in top["pod"]:
        if pod.startswith(ground_truth_prefix):
            colors.append("#e74c3c")
        else:
            colors.append("#3498db")

    fig, ax = plt.subplots(figsize=(13, 8))
    bars = ax.barh(top["pod"] + "\n" + top["metric"], top["root_score"], color=colors)

    labels = []
    for pod in top["pod"]:
        if pod.startswith(ground_truth_prefix):
            labels.append("TRUE ROOT CAUSE")
        else:
            labels.append("")

    for bar, label in zip(bars, labels):
        if label:
            ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                    label, va="center", fontsize=7, color="#e74c3c", fontweight="bold")

    ax.set_title("KPIRoot Root Cause Ranking\n(Red = Ground-Truth Root Cause)", fontsize=13)
    ax.set_xlabel("root_score")
    ax.axvline(x=0, color="gray", linewidth=0.5)

    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color="#e74c3c", label="Ground-truth root cause")
    blue_patch = mpatches.Patch(color="#3498db", label="Other KPI")
    ax.legend(handles=[red_patch, blue_patch], loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"\n  Ranking chart saved to: {output_path}")


def kpiroot_rank(input_path, output_dir, max_lag=5, alpha=0.45, beta=0.35, gamma=0.20,
                 anomaly_label="anomaly", ground_truth_pod_prefix="cartservice",
                 eval_top_k=None, plot_top_k=20):
    data = pd.read_csv(input_path)
    data["series"] = data["pod"] + "::" + data["metric"]

    if "label" in data.columns and anomaly_label in data["label"].values:
        normal_data = data[data["label"] == "normal"].copy()
        anomaly_data = data[data["label"] == anomaly_label].copy()

        # Build per-(pod, metric) delta from mean-value comparison.
        # Normalise delta within each metric so different metrics are comparable.
        normal_means = (
            normal_data.groupby(["pod", "metric"])["value"]
            .mean()
            .rename("normal_mean")
        )
        anomaly_means = (
            anomaly_data.groupby(["pod", "metric"])["value"]
            .mean()
            .rename("anomaly_mean")
        )
        delta_df = pd.concat([normal_means, anomaly_means], axis=1)
        delta_df["abs_delta"] = (delta_df["anomaly_mean"] - delta_df["normal_mean"]).abs()
        # Normalise delta per metric (0-1 range within each metric family)
        delta_df["abs_delta_norm"] = delta_df.groupby("metric")["abs_delta"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        )

        # Keep pivot for per-timestamp analysis when timestamps overlap
        pivot_normal = prepare_matrix(normal_data)
        pivot_anomaly = prepare_matrix(anomaly_data)

        pivot = pivot_anomaly  # scoring in the anomaly window
        # metric_targets stores per-metric normalised deltas keyed by metric name
        delta_by_metric = (
            delta_df["abs_delta_norm"]
            .groupby(delta_df.index.get_level_values("metric"))
            .mean()
            .to_dict()
        )
        metric_targets = delta_by_metric  # dict: metric_name -> scalar delta score
        # Store per-series delta for volatility scoring
        series_deltas = delta_df["abs_delta_norm"].to_dict()  # ("pod::metric") -> delta
    else:
        pivot = prepare_matrix(data)
        metric_targets = {}
        series_deltas = {}

    rows = []
    for col in pivot.columns:
        values = pivot[col].to_numpy()

        # Determine target: per-metric delta if available, else global mean
        metric_name = col.split("::", 1)[1]
        if isinstance(metric_targets, dict) and metric_targets:
            target_delta = metric_targets.get(metric_name, 0.0)
        else:
            target_delta = 0.0

        # Align lengths
        min_len = len(values)
        values_a = values[:min_len]
        values_z = zscore(values_a)

        # --- Correlation with global anomaly signal (time-series based) ---
        # Use global mean abs delta as pseudo-target when timestamps align
        global_target = np.abs(np.mean(np.abs(pivot.values), axis=1))
        global_target_a = global_target[:min_len]
        global_target_z = zscore(global_target_a)

        try:
            pearson = abs(pearsonr(values_z, global_target_z)[0]) if np.std(values_z) > 1e-8 else 0.0
        except Exception:
            pearson = 0.0
        try:
            spearman = abs(spearmanr(values_z, global_target_z).correlation) if np.std(values_z) > 1e-8 else 0.0
        except Exception:
            spearman = 0.0

        sax_candidate = sax_encode(values_a)
        global_target_for_trend = global_target[:min_len]
        sax_target = sax_encode(global_target_for_trend)
        trend = normalized_mutual_trend(sax_candidate, sax_target)
        causality, lag = lagged_causality_score(values_a, global_target_a, max_lag=max_lag)

        # --- Delta-based scoring (key for non-overlapping timestamps) ---
        series_key = col  # "pod::metric"
        delta_score = series_deltas.get(series_key, 0.0)

        # Combine time-series similarity with delta magnitude
        similarity = 0.5 * pearson + 0.3 * spearman + 0.2 * trend
        # Blend in delta-based score: higher delta = more likely root cause
        similarity = 0.4 * similarity + 0.6 * delta_score
        volatility = float(np.mean(np.abs(values_z) > 2.0))
        score = alpha * similarity + beta * causality + gamma * volatility

        pod, metric = col.split("::", 1)
        rows.append({
            "pod": pod,
            "metric": metric,
            "similarity": round(similarity, 4),
            "causality": round(causality, 4),
            "best_lag": lag,
            "volatility": round(volatility, 4),
            "root_score": round(score, 4),
        })

    result = pd.DataFrame(rows).sort_values("root_score", ascending=False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "kpiroot_ranking.csv", index=False, encoding="utf-8")

    top = result.head(15).copy()
    labels = top["pod"] + "\n" + top["metric"]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(labels[::-1], top["root_score"][::-1])
    ax.set_title("KPIRoot Root Cause KPI Ranking")
    ax.set_xlabel("root_score")
    fig.tight_layout()
    fig.savefig(output_dir / "kpiroot_ranking.png", dpi=160)
    plt.close(fig)

    service_rank = result.groupby("pod", as_index=False)["root_score"].max().sort_values("root_score", ascending=False)
    service_rank.to_csv(output_dir / "kpiroot_service_ranking.csv", index=False, encoding="utf-8")

    # --- Two-Stage Evaluation (matching ISSRE24 paper) ---
    print(f"\n{'='*60}")
    print(f"  KPIRoot Evaluation (ground-truth: '{ground_truth_pod_prefix}')")
    print(f"{'='*60}")

    # Per-(pod, metric) KPI-level evaluation
    print(f"\n{'─'*60}")
    print(f"  [KPI-Level] Per-(pod, metric) ranking ({len(result)} KPIs)")
    print(f"{'─'*60}")
    kpi_f1, kpi_hit = evaluate_ranking(result, ground_truth_pod_prefix, top_k_values=eval_top_k)
    kpi_f1_df = pd.DataFrame([kpi_f1])
    kpi_f1_df.to_csv(output_dir / "kpiroot_kpi_f1.csv", index=False, encoding="utf-8")
    kpi_hit.to_csv(output_dir / "kpiroot_kpi_hit.csv", index=False, encoding="utf-8")

    # Per-pod service-level evaluation
    print(f"\n{'─'*60}")
    print(f"  [Service-Level] Per-pod (service) ranking ({len(service_rank)} pods)")
    print(f"{'─'*60}")
    pod_f1, pod_hit = evaluate_ranking(service_rank.reset_index(drop=True),
                                       ground_truth_pod_prefix, top_k_values=eval_top_k)
    pod_f1_df = pd.DataFrame([pod_f1])
    pod_f1_df.to_csv(output_dir / "kpiroot_service_f1.csv", index=False, encoding="utf-8")
    pod_hit.to_csv(output_dir / "kpiroot_service_hit.csv", index=False, encoding="utf-8")

    # Plot with ground-truth highlighted
    if plot_top_k > 0:
        plot_ranking_with_gt(result.head(plot_top_k), ground_truth_pod_prefix,
                              output_dir / "kpiroot_ranking_with_gt.png")

    print(f"\nKPIRoot results saved to {output_dir}")
    print(service_rank.head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="ISSRE24 KPIRoot core reproduction with similarity and causality analysis")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="reports/kpiroot")
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--anomaly-label", default="anomaly")
    parser.add_argument("--ground-truth", default="cartservice",
                        help="Pod prefix that is the true root cause (for F1 evaluation)")
    parser.add_argument("--eval-top-k", type=int, nargs="+", default=[1, 3, 5, 10],
                        help="K values for Precision@K/Recall@K/F1@K evaluation")
    parser.add_argument("--plot-top-k", type=int, default=20,
                        help="How many top KPIs to show in the ranking chart (0=disable)")
    args = parser.parse_args()
    kpiroot_rank(args.input, args.output_dir, max_lag=args.max_lag,
                 anomaly_label=args.anomaly_label,
                 ground_truth_pod_prefix=args.ground_truth,
                 eval_top_k=args.eval_top_k,
                 plot_top_k=args.plot_top_k)


if __name__ == "__main__":
    main()
