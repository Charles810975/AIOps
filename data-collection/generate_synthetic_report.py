"""
生成多故障混合场景的高压测试对比报告（可视化）
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

SCENARIOS = {
    "exp1_cart_redis":    "Exp1: cart(CPU)+redis(Network) [双故障]",
    "exp2_checkout_productcatalog": "Exp2: checkout(CPU)+productcatalog(Network) [双故障]",
    "exp3_frontend_cart_net": "Exp3: frontend(Network)+cart(Network) [双故障-网络]",
    "exp4_triple_cart_redis_checkout": "Exp4: cart(CPU)+redis(Net)+checkout(CPU) [三故障]",
    "exp5_triple_frontend_productcatalog_checkout": "Exp5: front(Net)+productcat(CPU)+checkout(Net) [三故障]",
}

DATA_DIR = PROJECT_ROOT / "data-collection-synthetic"

# 从实际运行结果整理
# Service-level F1 Score and Hit@K
data = {
    "Exp1: cart+redis": {
        "cartservice":      {"F1": 0.500, "Hit1": 0, "Hit3": 1, "Hit5": 1, "Hit10": 1, "type": "CPU"},
        "redis-cart":        {"F1": 1.000, "Hit1": 1, "Hit3": 1, "Hit5": 1, "Hit10": 1, "type": "Network"},
    },
    "Exp2: checkout+productcatalog": {
        "checkoutservice":   {"F1": 0.400, "Hit1": 0, "Hit3": 0, "Hit5": 1, "Hit10": 1, "type": "CPU"},
        "productcatalog":    {"F1": 0.667, "Hit1": 0, "Hit3": 1, "Hit5": 1, "Hit10": 1, "type": "Network"},
    },
    "Exp3: frontend+cart": {
        "frontend":         {"F1": 1.000, "Hit1": 1, "Hit3": 1, "Hit5": 1, "Hit10": 1, "type": "Network"},
        "cartservice":      {"F1": 0.545, "Hit1": 0, "Hit3": 0, "Hit5": 0, "Hit10": 1, "type": "Network"},
    },
    "Exp4: cart+redis+checkout": {
        "cartservice":      {"F1": 0.400, "Hit1": 0, "Hit3": 0, "Hit5": 0, "Hit10": 0, "type": "CPU"},
        "redis-cart":        {"F1": 1.000, "Hit1": 1, "Hit3": 1, "Hit5": 1, "Hit10": 1, "type": "Network"},
        "checkoutservice":   {"F1": 0.706, "Hit1": 0, "Hit3": 0, "Hit5": 0, "Hit10": 1, "type": "CPU"},
    },
    "Exp5: front+productcat+checkout": {
        "frontend":         {"F1": 1.000, "Hit1": 1, "Hit3": 1, "Hit5": 1, "Hit10": 1, "type": "Network"},
        "productcatalog":   {"F1": 0.400, "Hit1": 0, "Hit3": 0, "Hit5": 1, "Hit10": 1, "type": "CPU"},
        "checkoutservice":  {"F1": 0.500, "Hit1": 0, "Hit3": 0, "Hit5": 0, "Hit10": 1, "type": "Network"},
    },
}

# Union GT results (all fault pods combined)
union_results = {
    "Exp1: cart+redis":           {"Hit1": 1, "Hit3": 3, "Hit5": 5, "Hit10": 5, "GT": 11},
    "Exp2: checkout+productcat":  {"Hit1": 0, "Hit3": 1, "Hit5": 2, "Hit10": 2, "GT": 12},
    "Exp3: frontend+cart":        {"Hit1": 1, "Hit3": 2, "Hit5": 3, "Hit10": 4, "GT": 12},
    "Exp4: cart+redis+checkout":  {"Hit1": 1, "Hit3": 1, "Hit5": 1, "Hit10": 3, "GT": 18},
    "Exp5: front+productcat+checkout": {"Hit1": 1, "Hit3": 1, "Hit5": 1, "Hit10": 3, "GT": 18},
}

fig = plt.figure(figsize=(18, 14))

# ---------------------------------------------------------------------------
# Plot 1: Service-level F1 Score comparison
# ---------------------------------------------------------------------------
ax1 = fig.add_subplot(2, 2, 1)
exp_labels = list(data.keys())
services = ["cartservice", "redis-cart", "checkoutservice",
            "productcatalog", "frontend"]
service_labels = ["cartservice", "redis-cart", "checkoutservice",
                  "productcatalog", "frontend"]
colors_svc = ["#e74c3c", "#9b59b6", "#f39c12", "#27ae60", "#3498db"]

x = np.arange(len(exp_labels))
width = 0.15
offsets = np.arange(len(services)) - 2 * width

f1_matrix = np.full((len(exp_labels), len(services)), np.nan)
for i, (exp, exp_data) in enumerate(data.items()):
    for j, svc in enumerate(services):
        key = svc.replace("-", "")
        for k, v in exp_data.items():
            if k.replace("-", "") == key or k == svc:
                f1_matrix[i, j] = exp_data[k].get("F1", np.nan)

for j, (svc, color) in enumerate(zip(service_labels, colors_svc)):
    vals = f1_matrix[:, j]
    bars = ax1.bar(x + offsets[j], vals, width, label=svc, color=color, alpha=0.85)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

ax1.set_ylabel("F1 Score", fontsize=11)
ax1.set_title("Stage 1: Anomaly Detection F1 Score\n(Service-Level)", fontsize=12, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels([e.split(":")[0] for e in exp_labels], rotation=15, ha="right", fontsize=9)
ax1.set_ylim(0, 1.2)
ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax1.legend(loc="upper right", fontsize=8)
ax1.grid(axis="y", alpha=0.3)

# ---------------------------------------------------------------------------
# Plot 2: Hit@K heatmap per service
# ---------------------------------------------------------------------------
ax2 = fig.add_subplot(2, 2, 2)
hit_data = []
for exp, exp_data in data.items():
    for svc, metrics in exp_data.items():
        hit_data.append({
            "scenario": exp.split(":")[0],
            "service": svc,
            "Hit1": metrics["Hit1"],
            "Hit3": metrics["Hit3"],
            "Hit5": metrics["Hit5"],
            "Hit10": metrics["Hit10"],
        })

hit_df = pd.DataFrame(hit_data)
pivot = hit_df.pivot_table(index="scenario", columns="service",
                           values="Hit10", aggfunc="first")
pivot = pivot.reindex([e.split(":")[0] for e in exp_labels])
svc_order = ["frontend", "cartservice", "checkoutservice",
             "productcatalog", "redis-cart"]
svc_order = [s for s in svc_order if s in pivot.columns]
pivot = pivot[svc_order]

im = ax2.imshow(pivot.values.astype(float), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax2.set_xticks(range(len(pivot.columns)))
ax2.set_xticklabels(pivot.columns, rotation=20, ha="right", fontsize=9)
ax2.set_yticks(range(len(pivot.index)))
ax2.set_yticklabels(pivot.index, fontsize=9)
ax2.set_title("Stage 2: Hit@10 Heatmap\n(1=命中, 0=未命中)", fontsize=12, fontweight="bold")
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        v = pivot.values[i, j]
        if not np.isnan(v):
            color = "white" if v < 0.5 else "black"
            ax2.text(j, i, "YES" if v == 1 else "NO",
                    ha="center", va="center", fontsize=8, color=color, fontweight="bold")
        else:
            ax2.text(j, i, "-", ha="center", va="center", fontsize=8, color="gray")
plt.colorbar(im, ax=ax2, shrink=0.8)

# ---------------------------------------------------------------------------
# Plot 3: Union GT Hit@K (all faults together)
# ---------------------------------------------------------------------------
ax3 = fig.add_subplot(2, 2, 3)
k_vals = [1, 3, 5, 10]
markers = ["o", "s", "^", "D", "p"]
colors_exp = ["#e74c3c", "#3498db", "#27ae60", "#9b59b6", "#f39c12"]

for idx, (exp, res) in enumerate(union_results.items()):
    k_norm = [res[f"Hit{k}"] / res["GT"] for k in k_vals]
    label = exp.split(":")[0]
    ax3.plot(k_vals, k_norm, marker=markers[idx], color=colors_exp[idx],
             linewidth=2.5, markersize=8, label=label)

ax3.set_xlabel("K (Top-K)", fontsize=11)
ax3.set_ylabel("Recall@K (GT 覆盖率)", fontsize=11)
ax3.set_title("Stage 2: Union GT Recall@K\n(联合根因覆盖率 = 命中的真实故障 KPI 数 / 真实故障 KPI 总数)",
              fontsize=11, fontweight="bold")
ax3.set_xticks(k_vals)
ax3.set_ylim(-0.05, 1.05)
ax3.grid(alpha=0.3)
ax3.legend(loc="lower right", fontsize=8)

# ---------------------------------------------------------------------------
# Plot 4: Summary table as text
# ---------------------------------------------------------------------------
ax4 = fig.add_subplot(2, 2, 4)
ax4.axis("off")

summary_text = """
╔══════════════════════════════════════════════════════════════════════════════════╗
║   Multi-Fault Hybrid Scenario Stress Test Results  —  KPIRoot Two-Stage Eval ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  Stage 1 — Anomaly Detection F1 Score (Service-Level)                          ║
║  ─────────────────────────────────────────────────────────────────────────────── ║
║  ● redis-cart / frontend     : F1 = 1.00  (Perfect detection)                  ║
║  ● cartservice              : F1 = 0.40~0.55 (Unstable ranking)                ║
║  ● checkoutservice          : F1 = 0.40~0.71 (CPU fault submerged by cascade) ║
║  ● productcatalog           : F1 = 0.40~0.67 (High variance)                   ║
║                                                                                  ║
║  Stage 2 — Root Cause Localization Hit@K                                        ║
║  ─────────────────────────────────────────────────────────────────────────────── ║
║  ● Hit@1:  Only redis-cart / frontend faults are correctly ranked #1           ║
║  ● Hit@5:  Most scenarios only capture 1-2 fault services                      ║
║  ● Hit@10: Triple-fault scenarios still miss most root causes                   ║
║                                                                                  ║
║  Key Findings:                                                                  ║
║  ─────────────────────────────────────────────────────────────────────────────── ║
║  1. Network faults (redis/frontend) are far easier to detect                    ║
║     Reason: Network error spikes are isolated & strong delta signals            ║
║                                                                                  ║
║  2. CPU faults are severely underestimated in multi-fault scenarios              ║
║     Reason: Downstream propagation masks direct CPU fault signals                 ║
║                                                                                  ║
║  3. More faults = harder root cause localization                                ║
║     2 faults: Union Recall@10 ≈ 17-45%                                          ║
║     3 faults: Union Recall@10 ≈ 17%  ← Significant drop                         ║
║                                                                                  ║
║  4. The synthetic data reveals challenges of real K8s environments:             ║
║     Fault propagation chains cause non-root services to receive high root_score  ║
║     This squeezes true root causes out of top-K rankings                        ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

ax4.text(0.02, 0.98, summary_text, transform=ax4.transAxes,
         fontsize=8.5, va="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="#f8f9fa", alpha=0.9, edgecolor="#dee2e6"))

fig.suptitle("KPIRoot Multi-Fault Hybrid Scenario Stress Test Report\n(Hybrid Fault Scenario Stress Test)",
             fontsize=14, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])

output_path = DATA_DIR / "multi_fault_stress_test_report.png"
fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Report saved: {output_path}")
