import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["timestamp", "pod", "metric", "value", "label", "fault_service"]


def load_and_normalize(path, label, fault_service):
    data = pd.read_csv(path)
    missing = [col for col in ["timestamp", "pod", "metric", "value"] if col not in data.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    data = data.copy()
    data["label"] = label
    data["fault_service"] = fault_service
    data = data[REQUIRED_COLUMNS]
    return data


def main():
    parser = argparse.ArgumentParser(description="Merge normal and anomaly Prometheus metric CSV files")
    parser.add_argument("--normal", required=True, help="Normal metrics CSV path")
    parser.add_argument("--anomaly", required=True, help="Anomaly metrics CSV path")
    parser.add_argument("--output", default="data-collection/combined_metrics.csv")
    parser.add_argument("--fault-service", default="cartservice")
    args = parser.parse_args()

    normal = load_and_normalize(args.normal, "normal", "none")
    anomaly = load_and_normalize(args.anomaly, "anomaly", args.fault_service)

    # Simply concatenate - do NOT normalize timestamps.
    # KPIRoot compares label="normal" vs label="anomaly" within the same dataframe.
    # Preserving original timestamps lets KPIRoot see the actual time ordering.
    merged = pd.concat([normal, anomaly], ignore_index=True)
    merged = merged.sort_values(["timestamp", "pod", "metric"]).reset_index(drop=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False, encoding="utf-8")

    print(f"Normal rows: {len(normal)}")
    print(f"Anomaly rows: {len(anomaly)}")
    print(f"Merged rows: {len(merged)}")
    print(f"Normal period: {normal['timestamp'].min()} - {normal['timestamp'].max()}")
    print(f"Anomaly period: {anomaly['timestamp'].min()} - {anomaly['timestamp'].max()}")
    print("Metrics:")
    print(merged["metric"].value_counts().to_string())
    print("Labels:")
    print(merged["label"].value_counts().to_string())
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
