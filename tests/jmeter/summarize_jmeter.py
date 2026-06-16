import argparse
from pathlib import Path

import pandas as pd


def summarize_jtl(path):
    df = pd.read_csv(path)
    if df.empty:
        return None
    elapsed = pd.to_numeric(df["elapsed"], errors="coerce")
    success = df["success"].astype(str).str.lower() == "true"
    duration_seconds = max((df["timeStamp"].max() - df["timeStamp"].min()) / 1000.0, 1.0)
    return {
        "run": path.parent.name,
        "samples": len(df),
        "success": int(success.sum()),
        "failures": int((~success).sum()),
        "error_rate": float((~success).mean()),
        "avg_ms": float(elapsed.mean()),
        "median_ms": float(elapsed.median()),
        "p90_ms": float(elapsed.quantile(0.90)),
        "p95_ms": float(elapsed.quantile(0.95)),
        "p99_ms": float(elapsed.quantile(0.99)),
        "min_ms": float(elapsed.min()),
        "max_ms": float(elapsed.max()),
        "throughput_req_per_sec": float(len(df) / duration_seconds),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize JMeter JTL results")
    parser.add_argument("--input-dir", default="reports/jmeter")
    parser.add_argument("--output", default="reports/jmeter/summary.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    rows = []
    for jtl in input_dir.glob("*/results.jtl"):
        summary = summarize_jtl(jtl)
        if summary:
            rows.append(summary)

    if not rows:
        raise RuntimeError(f"No JMeter results found in {input_dir}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(rows).sort_values("run")
    data.to_csv(output, index=False, encoding="utf-8")
    print(data.to_string(index=False))
    print(f"JMeter summary saved to {output}")


if __name__ == "__main__":
    main()
