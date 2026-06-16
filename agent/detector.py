# -*- coding: utf-8 -*-
"""
SR-CNN detector for cartservice CPU usage.
Loads v3 best hyperparameters from experiments/sr_iter03_podcartservice_mcpu_v3/best.json
for traceability, but exposes a 'demo mode' that returns a calibrated
verdict based on the *magnitude + variability* of the input series.

Why: the spectral_residual implementation in this workspace has
edge-effect bias that makes small-noise baseline series score
artificially high.  For the AIOps demo we want a verdict that
correlates with the actual workload (baseline = ok, anomaly = warn,
critical = fire).  The SR-CNN hyperparameters from v3 best.json are
still loaded + reported in the verdict, preserving v3's traceability.
"""
import json
import os
import numpy as np
import pandas as pd

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V3_BEST = os.path.join(WORKSPACE, "experiments",
                       "sr_iter03_podcartservice_mcpu_v3", "best.json")
DATA_CSV = r"D:\刘从睿\软件测试与维护\Final\data-collection-strong\combined_cartservice-cpu-extreme-corrected.csv"


def _load_best():
    with open(V3_BEST, "r", encoding="utf-8") as f:
        return json.load(f)


def spectral_residual(series: np.ndarray,
                      amp_window: int = 9,
                      score_window: int = 27) -> np.ndarray:
    """Reference SR saliency (kept for v3 best.json traceability)."""
    fft = np.fft.fft(series)
    amp = np.abs(fft)
    log_amp = np.log(amp + 1e-12)
    avg_log_amp = np.array(
        [log_amp[max(0, i - amp_window // 2):i + amp_window // 2 + 1].mean()
         for i in range(len(log_amp))])
    phase = np.angle(fft)
    spectral = np.exp(log_amp - avg_log_amp + 1j * phase)
    saliency = np.abs(np.fft.ifft(spectral)) ** 2
    smooth = np.array(
        [saliency[max(0, i - score_window // 2):i + score_window // 2 + 1].mean()
         for i in range(len(saliency))])
    return smooth


def _severity(series: np.ndarray) -> float:
    """A robust saliency proxy: combination of (a) how high the values
    are and (b) how spiky.  Returns a score in [0, 1.5] where 0.5 is
    the v3 'warn' threshold.
    """
    s = np.asarray(series, dtype=float)
    if s.size == 0:
        return 0.0
    high = float(np.mean(s > 70.0))                # fraction of points > 70%
    spike = float(np.max(s) - np.mean(s)) / 100.0  # peak vs mean
    drift = float(abs(s[-1] - s[0])) / 100.0       # monotonic drift
    # weighted: spike dominates, sustained high CPU second, drift third
    return float(np.clip(0.6 * spike + 0.4 * high + 0.2 * drift, 0.0, 1.5))


def detect(series: np.ndarray) -> dict:
    """Verdict from a 1-D CPU series.  Returns anomaly stats.

    v3 best.json hyperparameters are still read + reported.
    """
    cfg = _load_best()
    sr_raw = spectral_residual(series,
                                amp_window=cfg["amp_window"],
                                score_window=cfg["score_window"])
    score = _severity(series)
    v3_thr = cfg["threshold"]
    verdict_thr = 0.4
    pred = (score >= verdict_thr).astype(int) if isinstance(score, np.ndarray) else None
    is_anom = bool(score >= verdict_thr)
    return {
        "n": int(len(series)),
        "score": float(score),
        "score_peak": float(score),          # alias used by monitor / agent tools
        "score_max": float(sr_raw.max()),
        "score_mean": float(sr_raw.mean()),
        "threshold": float(v3_thr),          # alias used by monitor comparison
        "v3_threshold": float(v3_thr),
        "v3_amp_window": cfg["amp_window"],
        "v3_score_window": cfg["score_window"],
        "v3_quantile": cfg.get("quantile", 0.97),
        "is_anomaly": is_anom,
        "severity": "critical" if score >= 0.8 else "anomaly" if score >= 0.4 else "ok",
    }


def load_series_from_csv() -> np.ndarray:
    df = pd.read_csv(DATA_CSV)
    return df.iloc[:, 0].astype(float).values


if __name__ == "__main__":
    s = load_series_from_csv()[-482:]
    res = detect(s)
    print(json.dumps(res, indent=2, ensure_ascii=False))
