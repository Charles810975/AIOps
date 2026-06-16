"""
批量运行 KPIRoot 对所有合成多故障场景进行两阶段评估
Stage 1: Anomaly Detection F1 Score
Stage 2: Root Cause Localization Hit@K
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from algorithms.kpiroot.kpiroot_reproduction import (
    kpiroot_rank,
    compute_anomaly_detection_f1,
    compute_hit_at_k,
)

SCENARIOS = {
    "exp1_cart_redis":    {"name": "cartservice(CPU) + redis-cart(NetworkDelay)",
                           "fault_pods": ["cartservice", "redis-cart"]},
    "exp2_checkout_productcatalog": {"name": "checkoutservice(CPU) + productcatalog(NetworkDelay)",
                           "fault_pods": ["checkoutservice", "productcatalogservice"]},
    "exp3_frontend_cart_net": {"name": "frontend(NetworkDelay) + cartservice(NetworkDelay)",
                           "fault_pods": ["frontend", "cartservice"]},
    "exp4_triple_cart_redis_checkout": {"name": "cart(CPU) + redis(Network) + checkout(CPU)",
                           "fault_pods": ["cartservice", "redis-cart", "checkoutservice"]},
    "exp5_triple_frontend_productcatalog_checkout": {
                           "name": "frontend(Network) + productcatalog(CPU) + checkoutservice(Network)",
                           "fault_pods": ["frontend", "productcatalogservice", "checkoutservice"]},
}


def eval_scenario(s_key: str, scenario: dict, data_dir: Path) -> dict:
    combined_csv = data_dir / s_key / "combined.csv"
    if not combined_csv.exists():
        print(f"  [WARN] Not found: {combined_csv}")
        return None

    output_dir = data_dir / s_key / "kpiroot"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_gt_pods = scenario["fault_pods"]
    all_results = {}

    # 对每个故障 Pod 分别运行 KPIRoot 并评估
    for fault_pod in all_gt_pods:
        try:
            kpiroot_rank(
                input_path=str(combined_csv),
                output_dir=str(output_dir / fault_pod),
                max_lag=5,
                anomaly_label="anomaly",
                ground_truth_pod_prefix=fault_pod,
                eval_top_k=[1, 3, 5, 10],
                plot_top_k=0,
            )
        except Exception as e:
            print(f"  [ERROR] KPIRoot failed for {fault_pod}: {e}")
            all_results[fault_pod] = {"error": str(e)}
            continue

        # 读取排名结果进行精细评估
        ranking_csv = output_dir / fault_pod / "kpiroot_ranking.csv"
        if ranking_csv.exists():
            rank_df = pd.read_csv(ranking_csv)
            f1_info = compute_anomaly_detection_f1(rank_df, fault_pod)
            hit_df  = compute_hit_at_k(rank_df, fault_pod, [1, 3, 5, 10])

            # 保存
            (pd.DataFrame([f1_info])).to_csv(
                output_dir / fault_pod / "kpiroot_f1.csv", index=False)
            hit_df.to_csv(output_dir / fault_pod / "kpiroot_hit.csv", index=False)

            all_results[fault_pod] = {
                "f1": f1_info,
                "hit": {int(r.K): int(r["Hit@K"]) for _, r in hit_df.iterrows()},
            }

    # 联合评估：所有故障 Pod 的 KPI 并集作为 ground truth
    if ranking_csv.exists():
        rank_df = pd.read_csv(ranking_csv)
        union_gt = set()
        for fp in all_gt_pods:
            for idx, row in rank_df.iterrows():
                if row["pod"].startswith(fp):
                    union_gt.add(idx)

        # Hit@K for union ground truth
        hit_rows = []
        for k in [1, 3, 5, 10]:
            top_k_idx = set(rank_df.head(k).index)
            hit_rows.append({
                "K": k,
                "Hit@K": int(len(union_gt & top_k_idx) > 0),
                "GT_size": len(union_gt),
            })
        union_hit_df = pd.DataFrame(hit_rows)
        union_hit_df.to_csv(output_dir / "union_hit.csv", index=False)

        # Count how many GT KPIs appear in each K
        hit_detail = []
        for k in [1, 3, 5, 10]:
            top_k_idx = set(rank_df.head(k).index)
            n_hit = len(union_gt & top_k_idx)
            hit_detail.append(f"Hit@{k}={n_hit}/{len(union_gt)}")
        print(f"  [{s_key}] Union GT ({len(union_gt)} KPIs): {', '.join(hit_detail)}")

    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data-collection-synthetic",
                       help="合成数据根目录")
    parser.add_argument("--scenarios", nargs="+",
                       choices=list(SCENARIOS.keys()) + ["all"],
                       default=["all"])
    args = parser.parse_args()

    if "all" in args.scenarios:
        scenario_list = list(SCENARIOS.keys())
    else:
        scenario_list = args.scenarios

    data_dir = PROJECT_ROOT / args.data_dir

    all_summary = []

    for s_key in scenario_list:
        scenario = SCENARIOS[s_key]
        print(f"\n{'='*70}")
        print(f"[{s_key}] {scenario['name']}")
        print(f"Ground-truth pods: {scenario['fault_pods']}")
        print(f"{'='*70}")

        results = eval_scenario(s_key, scenario, data_dir)

        # 汇总行
        row = {
            "scenario": s_key,
            "name": scenario["name"],
            "num_faults": len(scenario["fault_pods"]),
            "fault_pods": ", ".join(scenario["fault_pods"]),
        }
        if results:
            for fp, res in results.items():
                if "error" in res:
                    row[f"{fp}_F1"] = "ERR"
                    for k in [1, 3, 5, 10]:
                        row[f"{fp}_Hit@{k}"] = "ERR"
                else:
                    row[f"{fp}_F1"] = res["f1"]["F1_Score"]
                    for k, v in res["hit"].items():
                        row[f"{fp}_Hit@{k}"] = v

        all_summary.append(row)

    # 保存汇总
    summary_df = pd.DataFrame(all_summary)
    summary_csv = data_dir / "all_scenarios_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    print(f"\n{'='*70}")
    print(f"  Summary saved to: {summary_csv}")
    print(f"{'='*70}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
