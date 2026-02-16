import argparse
import csv
import logging
import math
import os
import subprocess
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _detect_delimiter(line: str) -> str:
    if line.count("\t") > line.count(","):
        return "\t"
    return ","


def _read_pairs_from_sheet(sheet_path: str) -> List[Tuple[str, str, str, str, str]]:
    with open(sheet_path, "r") as handle:
        lines = handle.readlines()

    if not lines:
        raise ValueError("Sample sheet is empty")

    epic_col_l = "epic_id"
    cfrrbs_col_l = "npz_path"
    epic_bins_col_l = "epic_bins_path"
    epic_segments_col_l = "epic_segments_path"
    header_idx = None
    delimiter = ","

    for idx, line in enumerate(lines):
        if (epic_col_l in line.lower() and cfrrbs_col_l in line.lower()
                and epic_bins_col_l in line.lower() and epic_segments_col_l in line.lower()):
            delimiter = _detect_delimiter(line)
            header = [c.strip().lower() for c in line.split(delimiter)]
            if (epic_col_l in header and cfrrbs_col_l in header
                    and epic_bins_col_l in header and epic_segments_col_l in header):
                header_idx = idx
                break

    if header_idx is None:
        raise ValueError("Could not locate required columns in sample sheet")

    df = pd.read_csv(sheet_path, sep=delimiter, skiprows=header_idx, header=0)
    col_map = {c.lower(): c for c in df.columns}
    if (epic_col_l not in col_map or cfrrbs_col_l not in col_map
            or epic_bins_col_l not in col_map or epic_segments_col_l not in col_map):
        raise ValueError("Sample sheet is missing required columns")

    epic_col_real = col_map[epic_col_l]
    cfrrbs_col_real = col_map[cfrrbs_col_l]
    epic_bins_col_real = col_map[epic_bins_col_l]
    epic_segments_col_real = col_map[epic_segments_col_l]

    pairs = []
    for _, row in df.iterrows():
        epic_id = str(row.get(epic_col_real, "")).strip()
        cfrrbs_path = str(row.get(cfrrbs_col_real, "")).strip()
        epic_bins_path = str(row.get(epic_bins_col_real, "")).strip()
        epic_segments_path = str(row.get(epic_segments_col_real, "")).strip()
        if epic_id.lower() == "nan":
            epic_id = ""
        if cfrrbs_path.lower() == "nan":
            cfrrbs_path = ""
        if epic_bins_path.lower() == "nan":
            epic_bins_path = ""
        if epic_segments_path.lower() == "nan":
            epic_segments_path = ""
        if not epic_id or not cfrrbs_path or not epic_bins_path or not epic_segments_path:
            continue
        cfrrbs_id = os.path.splitext(os.path.basename(cfrrbs_path))[0]
        pairs.append((epic_id, cfrrbs_path, cfrrbs_id, epic_bins_path, epic_segments_path))

    if not pairs:
        raise ValueError("No valid EPIC/cfRRBS pairs found in sample sheet")

    return pairs


def _run_cfrrbs_predict(npz_path: str, reference: str, outid: str, args) -> None:
    cmd = [
        "WisecondorX",
        "predict",
        npz_path,
        reference,
        outid,
        "--bed",
    ]

    if args.beta is not None:
        cmd += ["--beta", str(args.beta)]
    if args.zscore is not None:
        cmd += ["--zscore", str(args.zscore)]
    if args.blacklist:
        cmd += ["--blacklist", args.blacklist]
    if args.gender:
        cmd += ["--gender", args.gender]

    subprocess.check_call(cmd)


def _normalize_chr(chr_value: str) -> str:
    val = str(chr_value).replace("chr", "")
    return val


def _load_bins_bed(bed_path: str) -> pd.DataFrame:
    df = pd.read_csv(bed_path, sep="\t")
    df["chr"] = df["chr"].apply(_normalize_chr)
    df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
    return df[["chr", "start", "end", "ratio"]]


def _load_epic_bins(tsv_path: str) -> pd.DataFrame:
    if tsv_path.endswith(".npz"):
        data = np.load(tsv_path, allow_pickle=True)
        df = pd.DataFrame(
            {
                "chr": data["chr"],
                "start": data["start"],
                "end": data["end"],
                "ratio": data["ratio"],
            }
        )
    else:
        df = pd.read_csv(tsv_path, sep="\t")
        if "start" not in df.columns and "bin_start" in df.columns:
            df = df.rename(columns={"bin_start": "start", "bin_end": "end"})
    df["chr"] = df["chr"].apply(_normalize_chr)
    df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
    return df[["chr", "start", "end", "ratio"]]


def _load_segments_bed(bed_path: str) -> pd.DataFrame:
    df = pd.read_csv(bed_path, sep="\t")
    df["chr"] = df["chr"].apply(_normalize_chr)
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df = df.dropna(subset=["start", "end"])
    return df


def _segments_overlap(seg_a, seg_b) -> bool:
    return seg_a["end"] >= seg_b["start"] and seg_b["end"] >= seg_a["start"]


def _compute_correlations(cf_bins: pd.DataFrame, epic_bins: pd.DataFrame) -> Dict[str, float]:
    merged = cf_bins.merge(epic_bins, on=["chr", "start", "end"], suffixes=("_cf", "_epic"))
    merged = merged.dropna(subset=["ratio_cf", "ratio_epic"])

    if merged.empty or len(merged) < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "n_bins": 0}

    pearson = merged[["ratio_cf", "ratio_epic"]].corr(method="pearson").iloc[0, 1]
    spearman = merged[["ratio_cf", "ratio_epic"]].corr(method="spearman").iloc[0, 1]

    return {"pearson": pearson, "spearman": spearman, "n_bins": len(merged)}


def _plot_scatter(df: pd.DataFrame, out_png: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not available; skipping scatter plot")
        return

    if df.empty:
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(df["ratio_cf"], df["ratio_epic"], s=8, alpha=0.6)
    plt.xlabel("cfRRBS log2 ratio")
    plt.ylabel("EPIC log2 ratio")
    plt.title("EPIC vs cfRRBS")
    plt.axhline(0, color="grey", linewidth=0.5)
    plt.axvline(0, color="grey", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def tool_epic_cfrrbs(args: argparse.Namespace) -> None:
    logging.info("Starting EPIC vs cfRRBS correlation")

    pairs = _read_pairs_from_sheet(args.sample_sheet)
    if args.max_replicates is not None and args.max_replicates > 0:
        pairs = pairs[: args.max_replicates]

    ref_file = np.load(args.reference, encoding="latin1", allow_pickle=True)
    binsize = int(ref_file["binsize"])
    del ref_file

    out_dir = os.path.abspath(args.outdir)
    cfrrbs_out = os.path.join(out_dir, "cfrrbs")
    report_dir = os.path.join(out_dir, "report")
    os.makedirs(cfrrbs_out, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    report_path = os.path.join(report_dir, "correlation_report.tsv")
    novel_path = os.path.join(report_dir, "novel_cfrrbs.tsv")

    report_rows = []
    novel_rows = []

    for epic_id, npz_path, cfrrbs_id, epic_bins_path, epic_segments_path in pairs:
        cf_outid = os.path.join(cfrrbs_out, cfrrbs_id, cfrrbs_id)
        os.makedirs(os.path.dirname(cf_outid), exist_ok=True)

        if not args.skip_cfrrbs_predict:
            _run_cfrrbs_predict(npz_path, args.reference, cf_outid, args)

        cf_bins_path = cf_outid + "_bins.bed"
        if not os.path.exists(cf_bins_path):
            logging.warning("Missing cfRRBS bins: %s", cf_bins_path)
            continue
        if not os.path.exists(epic_bins_path):
            logging.warning("Missing EPIC bins: %s", epic_bins_path)
            continue

        cf_bins = _load_bins_bed(cf_bins_path)
        epic_bins = _load_epic_bins(epic_bins_path)
        corr = _compute_correlations(cf_bins, epic_bins)

        report_rows.append(
            {
                "epic_id": epic_id,
                "cfrrbs_id": cfrrbs_id,
                "pearson": corr["pearson"],
                "spearman": corr["spearman"],
                "n_bins": corr["n_bins"],
            }
        )

        merged = cf_bins.merge(epic_bins, on=["chr", "start", "end"], suffixes=("_cf", "_epic"))
        merged = merged.dropna(subset=["ratio_cf", "ratio_epic"])
        plot_path = os.path.join(report_dir, "{}_scatter.png".format(cfrrbs_id))
        _plot_scatter(merged, plot_path)

        cf_aberrations = cf_outid + "_aberrations.bed"
        epic_segments = epic_segments_path
        if os.path.exists(cf_aberrations) and os.path.exists(epic_segments):
            cf_df = _load_segments_bed(cf_aberrations)
            epic_df = _load_segments_bed(epic_segments)
            for _, row in cf_df.iterrows():
                overlaps = epic_df[epic_df["chr"] == row["chr"]]
                has_overlap = False
                for _, e_row in overlaps.iterrows():
                    if _segments_overlap(row, e_row):
                        has_overlap = True
                        break
                novel_rows.append(
                    {
                        "epic_id": epic_id,
                        "cfrrbs_id": cfrrbs_id,
                        "chr": row["chr"],
                        "start": row["start"],
                        "end": row["end"],
                        "type": row.get("type", ""),
                        "overlaps_epic": 1 if has_overlap else 0,
                    }
                )

    pd.DataFrame(report_rows).to_csv(report_path, sep="\t", index=False)
    if novel_rows:
        pd.DataFrame(novel_rows).to_csv(novel_path, sep="\t", index=False)

    logging.info("Correlation report written to %s", report_path)

