import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score


def spectral_residual(values, amp_window=3, score_window=21, eps=1e-8):
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=np.nanmedian(values) if np.isfinite(values).any() else 0.0)
    fft = np.fft.fft(values)
    amplitude = np.abs(fft)
    phase = np.angle(fft)
    log_amplitude = np.log(amplitude + eps)
    avg_log_amplitude = uniform_filter1d(log_amplitude, size=amp_window, mode="nearest")
    residual = log_amplitude - avg_log_amplitude
    saliency = np.abs(np.fft.ifft(np.exp(residual + 1j * phase)))
    avg_saliency = uniform_filter1d(saliency, size=score_window, mode="nearest")
    score = (saliency - avg_saliency) / (avg_saliency + eps)
    return np.maximum(score, 0.0)


def detect(values, threshold_quantile=0.98):
    scores = spectral_residual(values)
    threshold = np.quantile(scores, threshold_quantile)
    pred = (scores >= threshold).astype(int)
    return scores, pred, threshold


def run(input_path, output_dir, metric, pod):
    data = pd.read_csv(input_path)
    if metric:
        data = data[data["metric"] == metric]
    if pod:
        data = data[data["pod"].str.contains(pod, regex=False, na=False)]
    if data.empty:
        raise RuntimeError("No matching data for selected metric/pod")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_rows = []
    for (series_pod, series_metric), group in data.groupby(["pod", "metric"]):
        group = group.sort_values("timestamp").copy()
        scores, pred, threshold = detect(group["value"].to_numpy())
        group["sr_score"] = scores
        group["pred"] = pred
        group["target"] = (group["label"] == "anomaly").astype(int)
        all_rows.append(group)

        if group["target"].nunique() > 1 or group["target"].sum() > 0:
            precision = precision_score(group["target"], group["pred"], zero_division=0)
            recall = recall_score(group["target"], group["pred"], zero_division=0)
            f1 = f1_score(group["target"], group["pred"], zero_division=0)
        else:
            precision = recall = f1 = 0.0

        summaries.append({
            "pod": series_pod,
            "metric": series_metric,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "anomaly_points": int(group["pred"].sum()),
        })

        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(pd.to_datetime(group["timestamp"], unit="s"), group["value"], label="value")
        ax1.scatter(pd.to_datetime(group.loc[group["pred"] == 1, "timestamp"], unit="s"), group.loc[group["pred"] == 1, "value"], color="red", label="detected anomaly", s=20)
        ax1.set_title(f"SR-CNN Spectral Residual: {series_pod} / {series_metric}")
        ax1.legend(loc="upper left")
        ax2 = ax1.twinx()
        ax2.plot(pd.to_datetime(group["timestamp"], unit="s"), group["sr_score"], color="orange", alpha=0.5, label="SR score")
        ax2.legend(loc="upper right")
        fig.tight_layout()
        safe_name = f"{series_pod}_{series_metric}".replace("/", "_").replace(":", "_")
        fig.savefig(output_dir / f"sr_{safe_name}.png", dpi=160)
        plt.close(fig)

    pd.concat(all_rows, ignore_index=True).to_csv(output_dir / "sr_cnn_results.csv", index=False, encoding="utf-8")
    pd.DataFrame(summaries).sort_values("f1", ascending=False).to_csv(output_dir / "sr_cnn_summary.csv", index=False, encoding="utf-8")
    print(f"SR-CNN results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="KDD19 SR-CNN core reproduction with Spectral Residual")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="reports/sr-cnn")
    parser.add_argument("--metric", default="cpu_usage")
    parser.add_argument("--pod", default="")
    args = parser.parse_args()
    run(args.input, args.output_dir, args.metric, args.pod)


if __name__ == "__main__":
    main()
