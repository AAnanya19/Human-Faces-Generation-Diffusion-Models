#!/usr/bin/env python3
"""
Aggregate ``fid.json`` and ``run_config.json`` across many training runs into a
single CSV/Markdown table, ready to paste into the report.

Each training run lives in its own folder (e.g. on Google Drive
``MyDrive/aml/ddpm_runs/<RUN_NAME>/``) and contains:

- ``run_config.json`` — written by ``src/training/train.py``
- ``fid.json``        — written by ``scripts/evaluate_fid.py``

This script scans a parent ``--runs-dir`` for sub-folders with both files,
joins them on RUN_NAME, and writes a CSV plus a Markdown table.

Example:
    python scripts/aggregate_fid.py \\
        --runs-dir /content/drive/MyDrive/aml/ddpm_runs \\
        --out-csv  results/celebahq/eval/fid_summary.csv \\
        --out-md   results/celebahq/eval/fid_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CONFIG_KEYS = (
    "epochs",
    "batch_size",
    "image_size",
    "learning_rate",
    "timesteps",
    "weight_decay",
)
MODEL_KEYS = (
    "base_channels",
    "time_dim",
    "num_res_blocks",
    "dropout",
    "channel_mults",
    "attention_resolutions",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join run_config.json + fid.json across runs into a table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Parent directory containing one folder per run (e.g. ddpm_runs/).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/celebahq/eval/fid_summary.csv"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("results/celebahq/eval/fid_summary.md"),
    )
    parser.add_argument(
        "--require-fid",
        action="store_true",
        help="Skip runs that do not yet have fid.json (useful to filter).",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        print(f"WARN: bad JSON in {path}: {exc}")
        return None


def flatten_row(run_name: str, cfg: dict[str, Any] | None, fid: dict[str, Any] | None) -> dict[str, Any]:
    row: dict[str, Any] = {"run_name": run_name}
    cfg = cfg or {}
    model_cfg = cfg.get("model") or {}
    for key in CONFIG_KEYS:
        row[key] = cfg.get(key)
    for key in MODEL_KEYS:
        row[key] = model_cfg.get(key)
    row["fid"] = (fid or {}).get("fid")
    row["n_real"] = (fid or {}).get("n_real")
    row["n_fake"] = (fid or {}).get("n_fake")
    return row


def write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_csv.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows: list[dict[str, Any]], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_md.write_text("_No runs found._\n")
        return
    fieldnames = list(rows[0].keys())
    header = "| " + " | ".join(fieldnames) + " |"
    sep = "| " + " | ".join("---" for _ in fieldnames) + " |"

    def fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        if isinstance(value, list):
            return ",".join(str(v) for v in value)
        return str(value)

    body = "\n".join(
        "| " + " | ".join(fmt(row[key]) for key in fieldnames) + " |" for row in rows
    )
    out_md.write_text("\n".join([header, sep, body]) + "\n")


def main() -> None:
    args = parse_args()
    runs_dir = args.runs_dir
    if not runs_dir.is_dir():
        raise SystemExit(f"runs-dir does not exist: {runs_dir}")

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        cfg = load_json(run_dir / "run_config.json")
        fid = load_json(run_dir / "fid.json")
        if cfg is None and fid is None:
            continue
        if args.require_fid and fid is None:
            continue
        rows.append(flatten_row(run_dir.name, cfg, fid))

    rows.sort(key=lambda r: (r.get("fid") is None, r.get("fid") or 0.0))

    write_csv(rows, args.out_csv)
    write_markdown(rows, args.out_md)
    print(f"Wrote {len(rows)} rows to {args.out_csv} and {args.out_md}")
    if rows:
        best = rows[0]
        print(
            f"Best run by FID: {best['run_name']} "
            f"(fid={best.get('fid')}, image_size={best.get('image_size')})"
        )


if __name__ == "__main__":
    main()
