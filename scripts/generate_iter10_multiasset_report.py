import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def fmt(value, digits=4):
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def render_table(headers, rows):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def as_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def total_signal_frequency(value):
    if isinstance(value, dict):
        return as_float(value.get("breakout")) + as_float(value.get("reversion"))
    return as_float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_path = Path(args.out).resolve() if args.out else (run_dir / "iter10_report.md")
    title = args.title or "iter10 multi-asset report"

    epoch_rows = load_jsonl(run_dir / "epoch_metrics.jsonl")
    per_asset_rows = load_jsonl(run_dir / "per_asset_metrics.jsonl")

    best_epoch = None
    best_score = None
    for row in epoch_rows:
        score = row.get("composite_score")
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_epoch = row.get("epoch")

    lines = [f"# {title} (run_dir={run_dir.as_posix()})", ""]
    if best_epoch is not None:
        lines.append(f"- best_epoch: {best_epoch}")
        lines.append(f"- best_score: {fmt(best_score)}")
    lines.append(f"- epochs: {len(epoch_rows)}")
    lines.append(f"- per_asset_records: {len(per_asset_rows)}")
    lines.append("")

    if epoch_rows:
        headers = [
            "epoch",
            "val_loss",
            "composite_score",
            "public_violation",
            "score_veto_passed",
            "avg_signal_frequency",
        ]
        table = []
        for row in epoch_rows:
            score_veto = row.get("score_veto") or {}
            avg_signal_frequency = (row.get("recent_score_components") or {}).get("avg_signal_frequency")
            table.append(
                [
                    str(row.get("epoch", "")),
                    fmt(row.get("val_loss")),
                    fmt(row.get("composite_score")),
                    fmt(row.get("public_below_directional_violation_rate")),
                    str(bool(score_veto.get("passed", False))),
                    fmt(avg_signal_frequency),
                ]
            )
        lines.append("## Epoch overview")
        lines.append(render_table(headers, table))
        lines.append("")

    if best_epoch is not None and per_asset_rows:
        best_asset_rows = [row for row in per_asset_rows if row.get("epoch") == best_epoch]
        grouped_by_timeframe = defaultdict(list)
        for row in best_asset_rows:
            grouped_by_timeframe[row.get("timeframe")].append(row)

        lines.append("## Best epoch by timeframe")
        timeframe_rows = []
        for timeframe in sorted(grouped_by_timeframe.keys()):
            rows = grouped_by_timeframe[timeframe]
            timeframe_rows.append(
                [
                    timeframe,
                    str(sum(row.get("sample_count", 0) for row in rows)),
                    fmt(sum(row.get("breakout_f1", 0.0) for row in rows) / max(len(rows), 1)),
                    fmt(sum(row.get("reversion_f1", 0.0) for row in rows) / max(len(rows), 1)),
                    fmt(
                        sum(row.get("public_below_directional_violation_rate", 0.0) for row in rows)
                        / max(len(rows), 1)
                    ),
                    fmt(sum(total_signal_frequency(row.get("signal_frequency")) for row in rows) / max(len(rows), 1)),
                ]
            )
        lines.append(
            render_table(
                [
                    "timeframe",
                    "sample_count",
                    "breakout_f1(avg)",
                    "reversion_f1(avg)",
                    "public_violation(avg)",
                    "signal_frequency(avg)",
                ],
                timeframe_rows,
            )
        )
        lines.append("")

        lines.append("## Best epoch by asset and timeframe")
        detail_headers = [
            "asset",
            "timeframe",
            "breakout_f1",
            "reversion_f1",
            "breakout_precision",
            "reversion_precision",
            "public_violation",
            "signal_frequency",
            "thresholds_frozen",
            "frozen_thresholds",
        ]
        detail_rows = []
        for row in sorted(best_asset_rows, key=lambda item: (item.get("asset", ""), item.get("timeframe", ""))):
            detail_rows.append(
                [
                    str(row.get("asset", "")),
                    str(row.get("timeframe", "")),
                    fmt(row.get("breakout_f1")),
                    fmt(row.get("reversion_f1")),
                    fmt(row.get("breakout_precision")),
                    fmt(row.get("reversion_precision")),
                    fmt(row.get("public_below_directional_violation_rate")),
                    fmt(total_signal_frequency(row.get("signal_frequency"))),
                    str(bool(row.get("thresholds_frozen", False))),
                    json.dumps(row.get("frozen_thresholds"), ensure_ascii=False),
                ]
            )
        lines.append(render_table(detail_headers, detail_rows))
        lines.append("")

    backtest_path = run_dir / "iter10_backtest_summary.csv"
    if backtest_path.exists():
        with backtest_path.open("r", encoding="utf-8") as handle:
            backtest_rows = list(csv.DictReader(handle))
        lines.append("## Minimal backtest summary")
        table_rows = []
        for row in backtest_rows:
            table_rows.append(
                [
                    row.get("series", ""),
                    row.get("khaos_total", ""),
                    row.get("khaos_sharpe", ""),
                    row.get("baseline_total", ""),
                    row.get("baseline_sharpe", ""),
                ]
            )
        lines.append(
            render_table(
                ["series", "khaos_total", "khaos_sharpe", "baseline_total", "baseline_sharpe"],
                table_rows,
            )
        )
        lines.append("")

    if per_asset_rows:
        lines.append("## Frozen thresholds by epoch/timeframe")
        threshold_rows = []
        seen = set()
        for row in sorted(
            per_asset_rows,
            key=lambda item: (item.get("epoch", 0), item.get("timeframe", ""), item.get("asset", "")),
        ):
            key = (row.get("epoch"), row.get("timeframe"))
            if key in seen:
                continue
            seen.add(key)
            threshold_rows.append(
                [
                    str(row.get("epoch", "")),
                    str(row.get("timeframe", "")),
                    str(bool(row.get("thresholds_frozen", False))),
                    json.dumps(row.get("frozen_thresholds"), ensure_ascii=False),
                ]
            )
        lines.append(
            render_table(
                ["epoch", "timeframe", "thresholds_frozen", "frozen_thresholds"],
                threshold_rows,
            )
        )
        lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(out_path.as_posix())


if __name__ == "__main__":
    main()
