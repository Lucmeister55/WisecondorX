import argparse
import csv
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from statistics import NormalDist
from typing import Dict, List, Set, Tuple

import numpy as np
import re
import pandas as pd

from wisecondorx.overall_tools import exec_R

# Files written exclusively by predict_output.py — epic_cfrrbs must never overwrite these.
_PREDICT_OUTPUT_OWNED = {
    "_bins.bed",
    "_segments.bed",
    "_aberrations.bed",
    "_regions.bed",
    "_statistics.txt",
    "_plot_bins_stats.tsv",
    "_focal_amplified_genes.tsv",
    "_focal_deleted_genes.tsv",
    "_focal_segments.tsv",
    "_purity_estimate.txt",
}


def _safe_write(path: str, df_or_func, *args, **kwargs):
    """Write a file only if it is not owned by predict_output.py."""
    basename = os.path.basename(path)
    for suffix in _PREDICT_OUTPUT_OWNED:
        if basename.endswith(suffix):
            raise RuntimeError(
                f"epic_cfrrbs attempted to overwrite predict_output.py file: {path}"
            )
    if callable(df_or_func):
        df_or_func(*args, **kwargs)
    else:
        df_or_func.to_csv(path, *args, **kwargs)


def _detect_delimiter(line: str) -> str:
    if line.count("\t") > line.count(","):
        return "\t"
    return ","


def _read_pairs_from_sheet(sheet_path: str) -> List[Tuple[str, str, str, str]]:
    """
    Reads sample sheet mapping EPIC samples to cfRRBS NPZ and conumee RDS files.
    Expected columns: epic_id, cfrrbs_npz_path, rds_path
    Returns: List of (epic_id, cfrrbs_npz_path, cfrrbs_id, rds_path)
    """
    with open(sheet_path, "r") as handle:
        lines = handle.readlines()

    if not lines:
        raise ValueError("Sample sheet is empty")

    epic_col_l = "epic_id"
    cfrrbs_col_l = "cfrrbs_npz_path"
    rds_col_l = "rds_path"
    # sWGS support removed: only require epic_id, cfrrbs_npz_path, rds_path
    header_idx = None
    delimiter = ","

    for idx, line in enumerate(lines):
        if (epic_col_l in line.lower() and cfrrbs_col_l in line.lower()
                and rds_col_l in line.lower()):
            delimiter = _detect_delimiter(line)
            header = [c.strip().lower() for c in line.split(delimiter)]
            if epic_col_l in header and cfrrbs_col_l in header and rds_col_l in header:
                header_idx = idx
                break

    if header_idx is None:
        raise ValueError("Could not locate required columns (epic_id, cfrrbs_npz_path, rds_path) in sample sheet")

    df = pd.read_csv(sheet_path, sep=delimiter, skiprows=header_idx, header=0)
    col_map = {c.lower(): c for c in df.columns}
    if epic_col_l not in col_map or cfrrbs_col_l not in col_map or rds_col_l not in col_map:
        raise ValueError("Sample sheet is missing required columns (epic_id, cfrrbs_npz_path, rds_path)")

    epic_col_real = col_map[epic_col_l]
    cfrrbs_col_real = col_map[cfrrbs_col_l]
    rds_col_real = col_map[rds_col_l]
    swgs_col_real = None

    pairs = []
    for _, row in df.iterrows():
        epic_id = str(row.get(epic_col_real, "")).strip()
        cfrrbs_path = str(row.get(cfrrbs_col_real, "")).strip()
        rds_path = str(row.get(rds_col_real, "")).strip()
        
        if epic_id.lower() == "nan":
            epic_id = ""
        if cfrrbs_path.lower() == "nan":
            cfrrbs_path = ""
        if rds_path.lower() == "nan":
            rds_path = ""
        
        if not epic_id or not cfrrbs_path or not rds_path:
            continue
        
        cfrrbs_id = os.path.splitext(os.path.basename(cfrrbs_path))[0]
        pairs.append((epic_id, cfrrbs_path, cfrrbs_id, rds_path))

    if not pairs:
        raise ValueError("No valid EPIC/cfRRBS pairs found in sample sheet")

    return pairs


def _run_cfrrbs_predict(
    npz_path: str,
    reference: str,
    outid: str,
    blacklist: str = None,
    regions: str = None,
    normalization_method: str = "reference",
) -> None:
    """Run WisecondorX predict with conumee plotting."""
    cmd = [
        "WisecondorX", "predict", npz_path, reference, outid,
        "--bed", "--conumee",
        "--normalization-method", normalization_method,
    ]
    if blacklist:
        cmd.extend(["--blacklist", blacklist])
    if regions:
        cmd.extend(["--regions", regions])
    subprocess.check_call(cmd)


def _normalize_chr(chr_value: str) -> str:
    val = str(chr_value).strip()
    # remove leading 'chr' (case-insensitive)
    val = re.sub(r'(?i)^chr', '', val)
    # normalize numeric chromosomes to no leading zeros
    if val.upper() in ("X", "Y"):
        return val.upper()
    try:
        return str(int(val))
    except Exception:
        return val


def _find_epic_columns(columns: list) -> dict:
    """
    Find EPIC RDS column mappings from dynamic column names.
    EPIC RDS columns are named like: X202995740090_R03C01.chrom, X202995740090_R03C01.loc.start, etc.
    Returns dict with found columns or raises error if required columns not found.
    """
    mapping = {}

    # chromosome variants
    chr_cols = [c for c in columns if c.lower().endswith('.chrom')]
    if not chr_cols:
        chr_cols = [c for c in columns if 'chrom' in c.lower() or c.lower().startswith('chr')]
    if chr_cols:
        mapping['chr'] = chr_cols[0]

    # start variants
    start_cols = [c for c in columns if c.lower().endswith('.loc.start') or c.lower().endswith('.start')]
    if not start_cols:
        start_cols = [c for c in columns if 'start' in c.lower()]
    if start_cols:
        mapping['start'] = start_cols[0]

    # end variants
    end_cols = [c for c in columns if c.lower().endswith('.loc.end') or c.lower().endswith('.end')]
    if not end_cols:
        end_cols = [c for c in columns if 'end' in c.lower()]
    if end_cols:
        mapping['end'] = end_cols[0]

    # name / gene column
    name_cols = [c for c in columns if c.lower().endswith('.name') or 'name' in c.lower()]
    if name_cols:
        mapping['name'] = name_cols[0]

    # ratio / value column
    ratio_cols = [c for c in columns if any(x in c.lower() for x in ['.seg.mean', '.seg.median', 'log2', '.value', 'value', 'ratio', 'val'])]
    if ratio_cols:
        mapping['ratio'] = ratio_cols[0]

    return mapping


def _find_epic_detail_columns(columns: list) -> dict:
    """Find EPIC detail TSV column mappings.
    Detail columns are named like: X204379160012_R02C01.Chromosome, .Start, .End, .Name, .Value
    """
    mapping = {}
    for c in columns:
        low = c.lower()
        if low.endswith('.chromosome'):
            mapping['chr'] = c
        elif low.endswith('.start'):
            mapping.setdefault('start', c)
        elif low.endswith('.end'):
            mapping.setdefault('end', c)
        elif low.endswith('.name'):
            mapping.setdefault('name', c)
        elif low.endswith('.value'):
            mapping.setdefault('ratio', c)
    return mapping


def _classify_cnv(ratio: float, threshold: float = 0.3) -> str:
    """Classify a CNV as gain, deletion, or neutral based on log2 ratio.
    
    Args:
        ratio: log2 ratio value
        threshold: absolute ratio threshold for calling CNV
    
    Returns:
        "gain", "deletion", or "neutral"
    """
    if pd.isna(ratio):
        return "unknown"
    if ratio > threshold:
        return "gain"
    elif ratio < -threshold:
        return "deletion"
    else:
        return "neutral"


def _extract_rds_conumee(rds_path: str, wd: str, outdir: str, epic_id: str) -> Tuple[str, str]:
    """
    Extract bins and segments from conumee RDS file using R script.
    Saves output to outdir instead of temp files.
    Returns: (bins_tsv_path, segments_tsv_path)
    """
    if not os.path.exists(rds_path):
        raise FileNotFoundError(f"RDS file not found: {rds_path}")
    
    logging.info(f"Extracting conumee data from RDS: {rds_path}")
    
    # Create output TSV files in outdir using epic_id
    bins_path = os.path.join(outdir, f"{epic_id}_epic_bins.tsv")
    segments_path = os.path.join(outdir, f"{epic_id}_epic_segments.tsv")
    
    # Create temporary JSON file path for R script input
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as json_tmp:
        json_path = json_tmp.name
    
    r_script = os.path.join(wd, "include", "extract_epic_rds.R")
    
    json_dict = {
        "R_script": r_script,
        "infile": json_path,
        "rds_path": str(os.path.abspath(rds_path)),
        "out_bins_path": bins_path,
        "out_segments_path": segments_path,
    }
    
    try:
        exec_R(json_dict)
        logging.info(f"Successfully extracted bins to {bins_path}")
        logging.info(f"Successfully extracted segments to {segments_path}")
        return bins_path, segments_path
    except Exception as e:
        logging.error(f"R extraction failed: {e}")
        raise
    finally:
        # Clean up JSON file
        if os.path.exists(json_path):
            os.remove(json_path)


def _load_epic_bins_segments_from_tsv(epic_dir: str, epic_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load already-extracted bins and segments TSV files from epic_dir.
    These files were created in Phase 2 during batch EPIC processing.
    Returns: (bins_df, segments_df)
    """
    bins_tsv = os.path.join(epic_dir, f"{epic_id}_bins.tsv")
    segments_tsv = os.path.join(epic_dir, f"{epic_id}_segments.tsv")
    
    if not os.path.exists(bins_tsv) or not os.path.exists(segments_tsv):
        raise FileNotFoundError(f"Missing EPIC TSV files for {epic_id}: bins={os.path.exists(bins_tsv)}, segments={os.path.exists(segments_tsv)}")
    
    try:
        df_bins = pd.read_csv(bins_tsv, sep="\t")
        logging.debug(f"Loaded bins TSV: columns={df_bins.columns.tolist()}, shape={df_bins.shape}")
        
        # Robust dynamic column detection using helper _find_epic_columns
        try:
            detected = _find_epic_columns(list(df_bins.columns))
        except Exception:
            detected = {}

        rename_map = {}
        for std_col, orig_col in detected.items():
            # detected maps std_col -> orig_col
            if orig_col in df_bins.columns:
                rename_map[orig_col] = std_col

        # Fallback heuristics if detection missed columns
        if "chr" not in rename_map.values():
            for c in df_bins.columns:
                if "chrom" in c.lower() or c.lower().startswith("chr") or c.lower() == "seqnames":
                    rename_map[c] = "chr"
                    break
        if "start" not in rename_map.values():
            for c in df_bins.columns:
                if "loc.start" in c.lower() or c.lower().endswith(".start") or c.lower() == "start":
                    rename_map[c] = "start"
                    break
        if "end" not in rename_map.values():
            for c in df_bins.columns:
                if "loc.end" in c.lower() or c.lower().endswith(".end") or c.lower() == "end":
                    rename_map[c] = "end"
                    break

        df_bins = df_bins.rename(columns=rename_map, errors="ignore")

        # Ensure required columns exist
        if "chr" not in df_bins.columns:
            raise ValueError(f"No chromosome column found in bins. Columns: {df_bins.columns.tolist()}")
        if "start" not in df_bins.columns:
            raise ValueError(f"No start column found in bins. Columns: {df_bins.columns.tolist()}")
        if "end" not in df_bins.columns:
            raise ValueError(f"No end column found in bins. Columns: {df_bins.columns.tolist()}")

        # Find ratio column - could be numeric index column if conumee writes it as first data column
        if "ratio" not in df_bins.columns:
            # Try to find the epic_id column (last column typically has the ratio)
            if epic_id in df_bins.columns:
                df_bins["ratio"] = pd.to_numeric(df_bins[epic_id], errors="coerce")
            else:
                # Fallback: look for numeric columns that are NOT start/end
                numeric_cols = [c for c in df_bins.select_dtypes(include=['number']).columns 
                               if c not in ['start', 'end']]
                if numeric_cols:
                    df_bins["ratio"] = pd.to_numeric(df_bins[numeric_cols[0]], errors="coerce")
                else:
                    raise ValueError(f"No numeric ratio column found in bins. Columns: {df_bins.columns.tolist()}")
        else:
            df_bins["ratio"] = pd.to_numeric(df_bins["ratio"], errors="coerce")
        
        # Normalize and filter
        df_bins["chr"] = df_bins["chr"].apply(_normalize_chr)
        df_bins["start"] = pd.to_numeric(df_bins["start"], errors="coerce")
        df_bins["end"] = pd.to_numeric(df_bins["end"], errors="coerce")
        df_bins = df_bins.dropna(subset=["start", "end", "ratio"])
        
        # Keep only required columns
        bins_df = df_bins[["chr", "start", "end", "ratio"]]
        logging.debug(f"Processed bins: {len(bins_df)} rows")
        
        # Load segments
        df_segments = pd.read_csv(segments_tsv, sep="\t")
        logging.debug(f"Loaded segments TSV: columns={df_segments.columns.tolist()}, shape={df_segments.shape}")
        
        # Standardize segment column names with dynamic detection
        try:
            detected_seg = _find_epic_columns(list(df_segments.columns))
        except Exception:
            detected_seg = {}

        rename_map_seg = {}
        for std_col, orig_col in detected_seg.items():
            if orig_col in df_segments.columns:
                rename_map_seg[orig_col] = std_col

        # Fallback heuristics for segments
        if "chr" not in rename_map_seg.values():
            for c in df_segments.columns:
                if "chrom" in c.lower() or c.lower().startswith("chr") or c.lower() == "seqnames":
                    rename_map_seg[c] = "chr"
                    break
        if "start" not in rename_map_seg.values():
            for c in df_segments.columns:
                if "loc.start" in c.lower() or c.lower().endswith(".start") or c.lower() == "start":
                    rename_map_seg[c] = "start"
                    break
        if "end" not in rename_map_seg.values():
            for c in df_segments.columns:
                if "loc.end" in c.lower() or c.lower().endswith(".end") or c.lower() == "end":
                    rename_map_seg[c] = "end"
                    break

        df_segments = df_segments.rename(columns=rename_map_seg, errors="ignore")

        # Ensure required columns exist for segments
        if "chr" not in df_segments.columns:
            raise ValueError(f"No chromosome column found in segments. Columns: {df_segments.columns.tolist()}")
        if "start" not in df_segments.columns:
            raise ValueError(f"No start column found in segments. Columns: {df_segments.columns.tolist()}")
        if "end" not in df_segments.columns:
            raise ValueError(f"No end column found in segments. Columns: {df_segments.columns.tolist()}")

        # Process segment data
        df_segments["chr"] = df_segments["chr"].apply(_normalize_chr)
        df_segments["start"] = pd.to_numeric(df_segments["start"], errors="coerce")
        df_segments["end"] = pd.to_numeric(df_segments["end"], errors="coerce")

        # Find ratio column for segments: require column ending with '.seg.median'
        if "ratio" not in df_segments.columns:
            cand = None
            for col in df_segments.columns:
                if col.lower().endswith('.seg.median'):
                    cand = col
                    break
            if cand is None:
                raise ValueError(f"No '.seg.median' ratio column found in segments. Columns: {df_segments.columns.tolist()}")
            df_segments["ratio"] = pd.to_numeric(df_segments[cand], errors="coerce")
        else:
            # If 'ratio' was provided explicitly, keep it but still prefer explicit '.seg.median' if present
            if any(c.lower().endswith('.seg.median') for c in df_segments.columns):
                for col in df_segments.columns:
                    if col.lower().endswith('.seg.median'):
                        df_segments["ratio"] = pd.to_numeric(df_segments[col], errors="coerce")
                        break
            else:
                df_segments["ratio"] = pd.to_numeric(df_segments["ratio"], errors="coerce")

        df_segments = df_segments.dropna(subset=["start", "end"])

        # Only keep standard columns
        cols_to_keep = ["chr", "start", "end"]
        if "ratio" in df_segments.columns:
            cols_to_keep.append("ratio")

        segments_df = df_segments[cols_to_keep]
        logging.debug(f"Processed segments: {len(segments_df)} rows")
        
        return bins_df, segments_df
    except Exception as e:
        logging.error(f"Failed to load EPIC TSV files for {epic_id}: {e}")
        raise



def _load_bins_bed(bed_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(bed_path, sep="\t")
        logging.debug(f"Loaded cfRRBS bins from {bed_path}, initial columns: {df.columns.tolist()}")
        
        # Ensure required columns exist
        if "chr" not in df.columns:
            raise ValueError(f"No 'chr' column in cfRRBS bins. Columns: {df.columns.tolist()}")
        if "start" not in df.columns:
            raise ValueError(f"No 'start' column in cfRRBS bins. Columns: {df.columns.tolist()}")
        if "end" not in df.columns:
            raise ValueError(f"No 'end' column in cfRRBS bins. Columns: {df.columns.tolist()}")
        if "ratio" not in df.columns:
            raise ValueError(f"No 'ratio' column in cfRRBS bins. Columns: {df.columns.tolist()}")
        
        df["chr"] = df["chr"].apply(_normalize_chr)
        df["start"] = pd.to_numeric(df["start"], errors="coerce")
        df["end"] = pd.to_numeric(df["end"], errors="coerce")
        df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
        result = df[["chr", "start", "end", "ratio"]].dropna()
        logging.debug(f"Processed cfRRBS bins: {len(result)} rows after normalization")
        return result
    except Exception as e:
        logging.error(f"Failed to load cfRRBS bins from {bed_path}: {e}")
        raise


def _load_segments_bed(bed_path: str) -> pd.DataFrame:
    df = pd.read_csv(bed_path, sep="\t")
    df["chr"] = df["chr"].apply(_normalize_chr)
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    if "ratio" in df.columns:
        df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
    df = df.dropna(subset=["start", "end"])
    return df


def _load_aberrations_bed(bed_path: str) -> pd.DataFrame:
    """Load WisecondorX predict aberrations BED and normalize core columns."""
    if not os.path.exists(bed_path):
        return pd.DataFrame(columns=["chr", "start", "end", "type", "ratio", "zscore", "genes", "pval"])

    df = pd.read_csv(bed_path, sep="\t")
    if "chr" not in df.columns:
        # tolerate variants like Chromosome/seqnames
        df = df.rename(columns={"Chromosome": "chr", "seqnames": "chr"}, errors="ignore")

    required = ["chr", "start", "end"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required aberration columns {missing} in {bed_path}")

    df["chr"] = df["chr"].astype(str).apply(_normalize_chr)
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    if "ratio" in df.columns:
        df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
    if "zscore" in df.columns:
        df["zscore"] = pd.to_numeric(df["zscore"], errors="coerce")
    if "pval" in df.columns:
        df["pval"] = pd.to_numeric(df["pval"], errors="coerce")

    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.lower().str.strip()

    return df.dropna(subset=["start", "end"])


def _call_broad_gene_events(
    detail_df: pd.DataFrame,
    bins_df: pd.DataFrame,
    conf: float = 0.99,
) -> pd.DataFrame:
    """
    Broad gene calling: is gene ratio significantly different from diploid neutral?

    K-means (k=3) on autosomal bin ratios identifies the neutral cluster (middle
    centroid). Gene called gain/deletion if its ratio falls outside the neutral
    CI (mu +/- z*sigma at conf).
    """
    from scipy.cluster.vq import kmeans2

    empty = pd.DataFrame(columns=["gene", "chr", "start", "end", "ratio",
                                   "zscore", "broad_call", "pval", "broad_low", "broad_high"])
    if detail_df is None or detail_df.empty:
        return empty

    detail = detail_df.copy()
    if "chr" in detail.columns:
        detail["chr"] = detail["chr"].astype(str).apply(_normalize_chr)
    for col in ["start", "end", "ratio"]:
        if col in detail.columns:
            detail[col] = pd.to_numeric(detail[col], errors="coerce")
    if "zscore" in detail.columns:
        detail["zscore"] = pd.to_numeric(detail["zscore"], errors="coerce")

    OUTLIER_THRESH = 0.8
    MIN_STATE_BINS = 5
    zcrit = float(NormalDist().inv_cdf(1 - (1 - conf) / 2)) if conf < 1 else 2.576
    neutral_low, neutral_high = -0.15, 0.15
    neutral_mu, neutral_sigma = 0.0, 0.05

    if bins_df is not None and not bins_df.empty:
        bins = bins_df.copy()
        if "chr" in bins.columns:
            bins["chr"] = bins["chr"].astype(str).apply(_normalize_chr)
        if "ratio" in bins.columns:
            bins["ratio"] = pd.to_numeric(bins["ratio"], errors="coerce")
        bins = bins.dropna(subset=["ratio"])
        autosome = bins["chr"].apply(lambda c: c not in ("X", "Y", "chrX", "chrY"))
        r_arr = bins.loc[autosome, "ratio"].values.astype(float)
        r_arr = r_arr[np.abs(r_arr) <= OUTLIER_THRESH]

        if len(r_arr) >= 15:
            global_mu = float(np.mean(r_arr))
            global_sigma = max(float(np.std(r_arr, ddof=1)), 0.05)
            fallback_thr = (global_mu - zcrit * global_sigma, global_mu + zcrit * global_sigma)
            init_centers = np.array([
                np.percentile(r_arr, 15),
                np.percentile(r_arr, 50),
                np.percentile(r_arr, 85),
            ])
            try:
                raw_centroids, labels = kmeans2(r_arr, init_centers, iter=20, minit='matrix')
                order = np.argsort(raw_centroids)
                label_remap = {int(old): i for i, old in enumerate(order)}
                labels_sorted = np.array([label_remap[int(l)] for l in labels])
                neutral_vals = r_arr[labels_sorted == 1]  # middle cluster = neutral
                if len(neutral_vals) >= MIN_STATE_BINS:
                    neutral_mu = float(np.mean(neutral_vals))
                    neutral_sigma = max(float(np.std(neutral_vals, ddof=1)), 0.03)
                    neutral_low = neutral_mu - zcrit * neutral_sigma
                    neutral_high = neutral_mu + zcrit * neutral_sigma
                else:
                    neutral_low, neutral_high = fallback_thr
                    neutral_mu = (neutral_low + neutral_high) / 2
                    neutral_sigma = max((neutral_high - neutral_low) / (2 * zcrit), 0.03)
            except Exception:
                neutral_low, neutral_high = fallback_thr
                neutral_mu = (neutral_low + neutral_high) / 2
                neutral_sigma = max((neutral_high - neutral_low) / (2 * zcrit), 0.03)

    rows = []
    for _, row in detail.iterrows():
        gene = str(row.get("name", row.get("gene", ""))).strip()
        ratio = row.get("ratio", np.nan)
        # Use pre-computed zscore if available (e.g. from WisecondorX _regions.bed),
        # otherwise derive from neutral distribution.
        zscore = row.get("zscore", np.nan)
        if pd.isna(zscore) or not np.isfinite(float(zscore) if pd.notna(zscore) else float("nan")):
            zscore = (float(ratio) - neutral_mu) / neutral_sigma if pd.notna(ratio) else np.nan
        else:
            zscore = float(zscore)
        pval = float(2 * (1 - NormalDist().cdf(abs(zscore)))) if np.isfinite(zscore) else np.nan
        call = "neutral"
        if pd.notna(ratio):
            if ratio > neutral_high:
                call = "gain"
            elif ratio < neutral_low:
                call = "deletion"
        rows.append({
            "gene": gene,
            "chr": row.get("chr"),
            "start": row.get("start"),
            "end": row.get("end"),
            "ratio": ratio,
            "zscore": zscore,
            "broad_call": call,
            "pval": pval,
            "broad_low": neutral_low,
            "broad_high": neutral_high,
        })
    return pd.DataFrame(rows)


def _call_focal_gene_events(
    detail_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    bins_df: pd.DataFrame,
    conf: float = 0.99,
) -> pd.DataFrame:
    """
    Focal gene calling: does this gene deviate significantly from its own segment?

    focal_dev = gene_ratio - segment_ratio
    focal_z   = focal_dev / max(within_state_bin_sd, noise_floor)
    Called if |focal_z| > z_crit.

    At cfRRBS resolution (500 kb bins ~ segments) focal deviations will be near
    zero — few focal calls is the expected and correct result.
    """
    from scipy.cluster.vq import kmeans2

    empty = pd.DataFrame(columns=["gene", "chr", "start", "end", "ratio",
                                   "zscore", "seg_ratio", "focal_dev", "focal_z", "focal_call", "pval"])
    if detail_df is None or detail_df.empty:
        return empty

    detail = detail_df.copy()
    if "chr" in detail.columns:
        detail["chr"] = detail["chr"].astype(str).apply(_normalize_chr)
    for col in ["start", "end", "ratio"]:
        if col in detail.columns:
            detail[col] = pd.to_numeric(detail[col], errors="coerce")
    if "zscore" in detail.columns:
        detail["zscore"] = pd.to_numeric(detail["zscore"], errors="coerce")

    segments = pd.DataFrame()
    if segments_df is not None and not segments_df.empty:
        segments = segments_df.copy()
        if "chr" in segments.columns:
            segments["chr"] = segments["chr"].astype(str).apply(_normalize_chr)
        for col in ["start", "end", "ratio"]:
            if col in segments.columns:
                segments[col] = pd.to_numeric(segments[col], errors="coerce")
        segments = segments.dropna(subset=["chr", "start", "end", "ratio"])

    OUTLIER_THRESH = 0.8
    MIN_STATE_BINS = 5
    NOISE_FLOOR = 0.05
    zcrit = float(NormalDist().inv_cdf(1 - (1 - conf) / 2)) if conf < 1 else 2.576
    state_sd = {0: NOISE_FLOOR, 1: NOISE_FLOOR, 2: NOISE_FLOOR}
    centroids = np.array([-0.3, 0.0, 0.3])

    if bins_df is not None and not bins_df.empty:
        bins = bins_df.copy()
        if "chr" in bins.columns:
            bins["chr"] = bins["chr"].astype(str).apply(_normalize_chr)
        if "ratio" in bins.columns:
            bins["ratio"] = pd.to_numeric(bins["ratio"], errors="coerce")
        bins = bins.dropna(subset=["ratio"])
        autosome = bins["chr"].apply(lambda c: c not in ("X", "Y", "chrX", "chrY"))
        r_arr = bins.loc[autosome, "ratio"].values.astype(float)
        r_arr = r_arr[np.abs(r_arr) <= OUTLIER_THRESH]
        if len(r_arr) >= 15:
            init_centers = np.array([
                np.percentile(r_arr, 15),
                np.percentile(r_arr, 50),
                np.percentile(r_arr, 85),
            ])
            try:
                raw_centroids, labels = kmeans2(r_arr, init_centers, iter=20, minit='matrix')
                order = np.argsort(raw_centroids)
                centroids = raw_centroids[order]
                label_remap = {int(old): i for i, old in enumerate(order)}
                labels_sorted = np.array([label_remap[int(l)] for l in labels])
                for i in range(3):
                    state_vals = r_arr[labels_sorted == i]
                    if len(state_vals) >= MIN_STATE_BINS:
                        state_sd[i] = max(float(np.std(state_vals, ddof=1)), NOISE_FLOOR)
            except Exception:
                pass

    def _seg_state(r):
        return int(np.argmin(np.abs(centroids - r))) if pd.notna(r) else 1

    rows = []
    for _, row in detail.iterrows():
        gene = str(row.get("name", row.get("gene", ""))).strip()
        chrom = row.get("chr")
        start = row.get("start")
        end = row.get("end")
        ratio = row.get("ratio", np.nan)

        seg_ratio = float("nan")
        if not segments.empty and pd.notna(chrom) and pd.notna(start) and pd.notna(end):
            ov = segments[
                (segments["chr"] == chrom)
                & (segments["start"] < end)
                & (segments["end"] > start)
            ]
            if not ov.empty:
                seg_ratio = float(ov["ratio"].mean())

        focal_dev = float("nan")
        focal_z = float("nan")
        call = "neutral"
        if pd.notna(ratio) and pd.notna(seg_ratio):
            focal_dev = ratio - seg_ratio
            noise = state_sd.get(_seg_state(seg_ratio), NOISE_FLOOR)
            focal_z = focal_dev / noise
            if abs(focal_z) > zcrit:
                call = "gain" if focal_z > 0 else "deletion"

        zscore = float(row.get("zscore", np.nan))
        focal_pval = float(2 * (1 - NormalDist().cdf(abs(focal_z)))) if np.isfinite(focal_z) else np.nan
        rows.append({
            "gene": gene, "chr": chrom, "start": start, "end": end,
            "ratio": ratio, "zscore": zscore, "seg_ratio": seg_ratio,
            "focal_dev": focal_dev, "focal_z": focal_z, "focal_call": call, "pval": focal_pval,
        })
    return pd.DataFrame(rows)


def _plot_concordance_heatmap(
    records: List[Dict],
    call_type: str,
    outdir: str,
) -> None:
    """
    Gene x sample concordance heatmap for broad or focal calls.
    Directional concordance categories:
      both_gain / both_del — concordant (with direction)
      epic_gain_only / epic_del_only — EPIC only (with direction)
      cfrrbs_gain_only / cfrrbs_del_only — cfRRBS only (with direction)
      discordant_epic_gain / discordant_epic_del — discordant (EPIC direction named)
      neutral — both platforms neutral
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if not records:
        logging.warning(f"No {call_type} concordance records to plot")
        return

    df = pd.DataFrame(records)
    genes = sorted(df["gene"].unique())
    samples = sorted(df["sample_id"].unique())
    if not genes or not samples:
        return

    COLOR_MAP = {
        "both_gain":            "#1a7d4a",  # dark green  — both agree: gain
        "both_del":             "#922b21",  # dark red    — both agree: deletion
        "epic_gain_only":       "#58d68d",  # med green   — EPIC gain, cfRRBS neutral
        "epic_del_only":        "#ec7063",  # med salmon  — EPIC deletion, cfRRBS neutral
        "cfrrbs_gain_only":     "#a9dfbf",  # light green — cfRRBS gain, EPIC neutral
        "cfrrbs_del_only":      "#f1948a",  # light pink  — cfRRBS deletion, EPIC neutral
        "discordant_epic_gain": "#8e44ad",  # purple      — EPIC↑  cfRRBS↓
        "discordant_epic_del":  "#e67e22",  # orange      — EPIC↓  cfRRBS↑
        "neutral":              "#d5d8dc",  # light grey  — both neutral
    }
    LEGEND_LABELS = {
        "both_gain":            "Both: Gain",
        "both_del":             "Both: Deletion",
        "epic_gain_only":       "EPIC only: Gain",
        "epic_del_only":        "EPIC only: Deletion",
        "cfrrbs_gain_only":     "cfRRBS only: Gain",
        "cfrrbs_del_only":      "cfRRBS only: Deletion",
        "discordant_epic_gain": "Discordant (EPIC↑ cfRRBS↓)",
        "discordant_epic_del":  "Discordant (EPIC↓ cfRRBS↑)",
        "neutral":              "Neutral (both)",
    }

    mat = pd.DataFrame("neutral", index=genes, columns=samples)
    for _, r in df.iterrows():
        mat.at[r["gene"], r["sample_id"]] = r["concordance"]

    fig_w = max(8, len(samples) * 0.7 + 3)
    fig_h = max(4, len(genes) * 0.45 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for gi, gene in enumerate(genes):
        for si, sample in enumerate(samples):
            color = COLOR_MAP.get(mat.at[gene, sample], COLOR_MAP["neutral"])
            ax.add_patch(plt.Rectangle((si, gi), 1, 1, color=color, linewidth=0.3, edgecolor="white"))

    ax.set_xlim(0, len(samples))
    ax.set_ylim(0, len(genes))
    ax.set_xticks([i + 0.5 for i in range(len(samples))])
    ax.set_xticklabels(samples, rotation=45, ha="right", fontsize=8)
    ax.set_yticks([i + 0.5 for i in range(len(genes))])
    ax.set_yticklabels(genes, fontsize=8)
    ax.set_title(f"{call_type.capitalize()} gene call concordance (EPIC vs cfRRBS)",
                 fontsize=12, fontweight="bold")

    # Show only categories that actually appear in the data
    present = set(mat.values.flatten())
    legend_handles = [
        mpatches.Patch(color=COLOR_MAP[k], label=LEGEND_LABELS[k])
        for k in COLOR_MAP if k in present
    ]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=8, framealpha=0.9, title="Concordance")
    plt.tight_layout()
    out_path = os.path.join(outdir, f"concordance_heatmap_{call_type}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved {call_type} concordance heatmap to {out_path}")


def _segments_overlap(seg_a, seg_b) -> bool:
    return seg_a["end"] >= seg_b["start"] and seg_b["end"] >= seg_a["start"]


def _pair_bins_by_overlap(cf_bins: pd.DataFrame, epic_bins: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cf_chrs = cf_bins["chr"].unique()
    epic_chrs = epic_bins["chr"].unique()
    common_chrs = set(cf_chrs) & set(epic_chrs)
    logging.debug(f"Common chromosomes: {len(common_chrs)} of cf={len(cf_chrs)}, epic={len(epic_chrs)}")
    
    for chr_name in cf_bins["chr"].unique():
        cf_chr = cf_bins[cf_bins["chr"] == chr_name].sort_values("start")
        ep_chr = epic_bins[epic_bins["chr"] == chr_name].sort_values("start")
        
        if cf_chr.empty or ep_chr.empty:
            logging.debug(f"Chromosome {chr_name}: cf_empty={cf_chr.empty}, ep_empty={ep_chr.empty}")
            continue
        
        logging.debug(f"Chromosome {chr_name}: {len(cf_chr)} cf_bins, {len(ep_chr)} epic_bins")

        ep_starts = ep_chr["start"].to_numpy()
        ep_ends = ep_chr["end"].to_numpy()
        ep_ratios = ep_chr["ratio"].to_numpy()

        ep_idx = 0
        ep_len = len(ep_chr)
        for _, cf_row in cf_chr.iterrows():
            cf_start = cf_row["start"]
            cf_end = cf_row["end"]

            while ep_idx < ep_len and ep_ends[ep_idx] < cf_start:
                ep_idx += 1

            j = ep_idx
            overlap_ratios = []
            while j < ep_len and ep_starts[j] <= cf_end:
                if ep_ends[j] >= cf_start:
                    overlap_ratios.append(ep_ratios[j])
                j += 1

            if overlap_ratios:
                rows.append(
                    {
                        "ratio_cf": cf_row["ratio"],
                        "ratio_epic": float(np.nanmean(overlap_ratios)),
                        "n_overlap": len(overlap_ratios),
                    }
                )
    
    result_df = pd.DataFrame(rows)
    logging.debug(f"Total overlapping bin pairs found: {len(result_df)}")
    return result_df


def _compute_correlations(cf_bins: pd.DataFrame, epic_bins: pd.DataFrame) -> Dict[str, float]:
    logging.debug(f"Computing correlations: cf_bins shape={cf_bins.shape}, epic_bins shape={epic_bins.shape}")
    logging.debug(f"cf_bins columns: {cf_bins.columns.tolist()}, chrs: {cf_bins['chr'].unique().tolist()}")
    logging.debug(f"epic_bins columns: {epic_bins.columns.tolist()}, chrs: {epic_bins['chr'].unique().tolist()}")
    
    pairs = _pair_bins_by_overlap(cf_bins, epic_bins)
    
    logging.debug(f"Paired bins result: {len(pairs)} rows, columns: {pairs.columns.tolist()}")
    
    pairs = pairs.dropna(subset=["ratio_cf", "ratio_epic"])

    if pairs.empty or len(pairs) < 3:
        logging.warning(f"Insufficient paired bins for correlation: {len(pairs)} rows after dropna")
        return {
            "pearson": float("nan"),
            "pearson_p": float("nan"),
            "spearman": float("nan"),
            "spearman_p": float("nan"),
            "n_bins": 0,
        }

    pearson = pairs[["ratio_cf", "ratio_epic"]].corr(method="pearson").iloc[0, 1]
    spearman = pairs[["ratio_cf", "ratio_epic"]].corr(method="spearman").iloc[0, 1]

    try:
        from scipy import stats

        pearson_p = stats.pearsonr(pairs["ratio_cf"], pairs["ratio_epic"]).pvalue
        spearman_p = stats.spearmanr(pairs["ratio_cf"], pairs["ratio_epic"]).pvalue
    except Exception:
        pearson_p = float("nan")
        spearman_p = float("nan")

    return {
        "pearson": pearson,
        "pearson_p": pearson_p,
        "spearman": spearman,
        "spearman_p": spearman_p,
        "n_bins": len(pairs),
    }


def _pair_segments_by_overlap(cf_segments: pd.DataFrame, epic_segments: pd.DataFrame) -> pd.DataFrame:
    """Simplified pairing: for each cfRRBS segment, choose the EPIC segment
    with highest Jaccard (intersection over union). If no overlap exists with
    any EPIC segment, pair to the nearest EPIC segment by distance.

    This returns one partner per cf segment (so downstream correlations are
    computed per cf segment). Columns: chr, cf_start, cf_end, epic_start,
    epic_end, ratio_cf, ratio_epic, overlap_len, jaccard, dist_to_partner.
    """
    rows = []
    # debug counters
    total_cf = 0
    paired_count = 0
    fallback_count = 0
    no_epic_chr_count = 0
    logging.debug(f"_pair_segments_by_overlap: cf_segments={0 if cf_segments is None else len(cf_segments)}, epic_segments={0 if epic_segments is None else len(epic_segments)}")
    if cf_segments is None or cf_segments.empty:
        return pd.DataFrame(columns=["chr", "cf_start", "cf_end", "epic_start", "epic_end", "ratio_cf", "ratio_epic", "overlap_len", "jaccard", "dist_to_partner"]) 
    # ensure epic_segments exists
    if epic_segments is None or epic_segments.empty:
        # still produce rows with NaN partners
        for _, cf in cf_segments.iterrows():
            rows.append({
                "chr": cf.get("chr"),
                "cf_start": cf.get("start"),
                "cf_end": cf.get("end"),
                "epic_start": np.nan,
                "epic_end": np.nan,
                "ratio_cf": cf.get("ratio", np.nan),
                "ratio_epic": np.nan,
                "overlap_len": 0,
                "jaccard": 0.0,
                "dist_to_partner": np.nan,
            })
        return pd.DataFrame(rows)

    # operate per chromosome
    for chr_name in sorted(cf_segments["chr"].unique()):
        cf_chr = cf_segments[cf_segments["chr"] == chr_name].sort_values("start").reset_index(drop=True)
        epic_chr = epic_segments[epic_segments["chr"] == chr_name].sort_values("start").reset_index(drop=True)
        if epic_chr.empty:
            # no epic segments on this chr; pair to NaN and log
            logging.debug(f"_pair_segments_by_overlap: no EPIC segments for chr {chr_name}; pairing CF segments to NaN")
            for _, cf in cf_chr.iterrows():
                total_cf += 1
                no_epic_chr_count += 1
                rows.append({
                    "chr": chr_name,
                    "cf_start": cf.get("start"),
                    "cf_end": cf.get("end"),
                    "epic_start": np.nan,
                    "epic_end": np.nan,
                    "ratio_cf": cf.get("ratio", np.nan),
                    "ratio_epic": np.nan,
                    "overlap_len": 0,
                    "jaccard": 0.0,
                    "dist_to_partner": np.nan,
                })
            continue

        logging.debug(f"_pair_segments_by_overlap: epic_chr columns: {list(epic_chr.columns)}")
        e_starts = epic_chr["start"].to_numpy()
        e_ends = epic_chr["end"].to_numpy()
        # Prefer explicit 'ratio' column if present (recent code renames '.seg.median' -> 'ratio')
        if "ratio" in epic_chr.columns:
            e_ratios = pd.to_numeric(epic_chr["ratio"], errors="coerce").to_numpy()
        else:
            # Otherwise require a '.seg.median' column
            cand = None
            for c in epic_chr.columns:
                if c.lower().endswith('.seg.median'):
                    cand = c
                    break
            if cand is None:
                raise ValueError(f"EPIC segments for chr {chr_name} missing required 'ratio' or '.seg.median' column. Columns: {epic_chr.columns.tolist()}")
            logging.debug(f"_pair_segments_by_overlap: using epic column '{cand}' for ratios")
            e_ratios = pd.to_numeric(epic_chr[cand], errors="coerce").to_numpy()

        for _, cf in cf_chr.iterrows():
            total_cf += 1
            cf_start = cf.get("start")
            cf_end = cf.get("end")
            if pd.isna(cf_start) or pd.isna(cf_end):
                continue
            # intersection length (inclusive)
            inter_left = np.maximum(e_starts, cf_start)
            inter_right = np.minimum(e_ends, cf_end)
            inter_len = np.maximum(0, inter_right - inter_left + 1)
            # union length
            union_left = np.minimum(e_starts, cf_start)
            union_right = np.maximum(e_ends, cf_end)
            union_len = np.maximum(1, union_right - union_left + 1)
            jaccard = inter_len / union_len.astype(float)

            if inter_len.sum() > 0:
                best_idx = int(np.nanargmax(jaccard))
                ov = int(inter_len[best_idx])
                jc = float(jaccard[best_idx])
                partner_ratio = e_ratios[best_idx]
                dist = 0.0
                paired_count += 1
            else:
                # fallback: nearest partner by center distance
                cf_center = (cf_start + cf_end) / 2.0
                e_centers = (e_starts + e_ends) / 2.0
                dist_arr = np.abs(e_centers - cf_center)
                best_idx = int(np.nanargmin(dist_arr))
                ov = 0
                jc = 0.0
                partner_ratio = e_ratios[best_idx]
                dist = float(dist_arr[best_idx])
                fallback_count += 1
                logging.debug(
                    f"_pair_segments_by_overlap: fallback pairing chr={chr_name} cf=({int(cf_start)}-{int(cf_end)}) "
                    f"-> epic=({int(e_starts[best_idx])}-{int(e_ends[best_idx])}) dist={dist:.1f}"
                )

            rows.append({
                "chr": chr_name,
                "cf_start": int(cf_start),
                "cf_end": int(cf_end),
                "epic_start": int(e_starts[best_idx]),
                "epic_end": int(e_ends[best_idx]),
                "ratio_cf": cf.get("ratio", np.nan),
                "ratio_epic": float(partner_ratio) if not pd.isna(partner_ratio) else np.nan,
                "overlap_len": ov,
                "jaccard": jc,
                "dist_to_partner": dist,
            })

    return pd.DataFrame(rows)


def _aggregate_epic_bins_to_cfrrbs(epic_bins: pd.DataFrame, cf_bins: pd.DataFrame) -> pd.DataFrame:
    """Aggregate EPIC bins to match cfRRBS bin boundaries for better pairing.
    
    Since cfRRBS bins are much larger, we group EPIC bins that fall within each cfRRBS bin
    and take the mean ratio.
    """
    aggregated = []
    for _, cf_row in cf_bins.iterrows():
        cf_chr = cf_row["chr"]
        cf_start = cf_row["start"]
        cf_end = cf_row["end"]
        
        # Find all EPIC bins within this cfRRBS bin
        overlapping = epic_bins[
            (epic_bins["chr"] == cf_chr) & 
            (epic_bins["start"] < cf_end) & 
            (epic_bins["end"] > cf_start)
        ]
        
        if not overlapping.empty:
            # Aggregate with mean ratio
            mean_ratio = overlapping["ratio"].mean()
            aggregated.append({
                "chr": cf_chr,
                "start": cf_start,
                "end": cf_end,
                "ratio": mean_ratio,
            })
    
    return pd.DataFrame(aggregated)


def _compute_segment_correlations(
    cf_segments: pd.DataFrame, epic_segments: pd.DataFrame
) -> Dict[str, float]:
    if "ratio" not in cf_segments.columns or "ratio" not in epic_segments.columns:
        return {
            "pearson": float("nan"),
            "pearson_p": float("nan"),
            "spearman": float("nan"),
            "spearman_p": float("nan"),
            "n_segments": 0,
        }

    pairs = _pair_segments_by_overlap(cf_segments, epic_segments)

    # If returned DataFrame does not contain expected columns, abort gracefully
    if not isinstance(pairs, pd.DataFrame) or "ratio_cf" not in pairs.columns or "ratio_epic" not in pairs.columns:
        return {
            "pearson": float("nan"),
            "pearson_p": float("nan"),
            "spearman": float("nan"),
            "spearman_p": float("nan"),
            "n_segments": 0,
        }

    pairs = pairs.dropna(subset=["ratio_cf", "ratio_epic"])

    if len(pairs) < 3:
        return {
            "pearson": float("nan"),
            "pearson_p": float("nan"),
            "spearman": float("nan"),
            "spearman_p": float("nan"),
            "n_segments": 0,
        }

    pearson = pairs[["ratio_cf", "ratio_epic"]].corr(method="pearson").iloc[0, 1]
    spearman = pairs[["ratio_cf", "ratio_epic"]].corr(method="spearman").iloc[0, 1]

    try:
        from scipy import stats

        pearson_p = stats.pearsonr(pairs["ratio_cf"], pairs["ratio_epic"]).pvalue
        spearman_p = stats.spearmanr(pairs["ratio_cf"], pairs["ratio_epic"]).pvalue
    except Exception:
        pearson_p = float("nan")
        spearman_p = float("nan")

    return {
        "pearson": pearson,
        "pearson_p": pearson_p,
        "spearman": spearman,
        "spearman_p": spearman_p,
        "n_segments": len(pairs),
    }


def _plot_scatter(df: pd.DataFrame, out_png: str, title_prefix: str = "", corr: Dict[str, float] = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not available; skipping scatter plot")
        return

    plt.figure(figsize=(6, 6))
    
    if df.empty:
        plt.text(0.5, 0.5, f'No overlapping {title_prefix.lower()}', 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=12, color='red')
        logging.warning(f"No overlapping data for scatter plot: {out_png}")
    else:
        plt.scatter(df["ratio_cf"], df["ratio_epic"], s=8, alpha=0.6)
        min_val = min(df["ratio_cf"].min(), df["ratio_epic"].min())
        max_val = max(df["ratio_cf"].max(), df["ratio_epic"].max())
        plt.plot([min_val, max_val], [min_val, max_val], color="black", linewidth=1, linestyle="--")

        try:
            fit = np.polyfit(df["ratio_cf"], df["ratio_epic"], 1)
            fit_x = np.array([min_val, max_val])
            fit_y = fit[0] * fit_x + fit[1]
            plt.plot(fit_x, fit_y, color="red", linewidth=1.5)
        except Exception:
            pass
        plt.axhline(0, color="grey", linewidth=0.5)
        plt.axvline(0, color="grey", linewidth=0.5)
    
    plt.xlabel("cfRRBS log2 ratio")
    plt.ylabel("EPIC log2 ratio")
    if corr:
        title = (
            f"{title_prefix}EPIC vs cfRRBS\n"
            f"Pearson r={corr.get('pearson', float('nan')):.3f} "
            f"(p={corr.get('pearson_p', float('nan')):.2e}), "
            f"Spearman r={corr.get('spearman', float('nan')):.3f} "
            f"(p={corr.get('spearman_p', float('nan')):.2e})"
        )
    else:
        title = f"{title_prefix}EPIC vs cfRRBS"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _plot_epic_summary(rds_list: List[Tuple[str, str, str]], outdir: str) -> None:
    """
    Generate individual genomeplot for each EPIC sample + summary plots (summaryplot, heatmap)
    for all samples using plot_epic_combined.R. Loads all packages and RDS files once.
    
    Args:
        rds_list: List of (epic_id, rds_path, epic_dir) tuples
        outdir: Output directory for summary plots (summary folder)
    """
    if not rds_list:
        logging.info("No EPIC samples to generate plots for")
        return
    
    wd = str(os.path.dirname(os.path.realpath(__file__)))
    r_script = os.path.join(wd, "include", "plot_epic_combined.R")
    
    if not os.path.exists(r_script):
        logging.warning(f"R script not found: {r_script}")
        return
    
    # Filter out non-existent RDS files
    valid_rds = [(epic_id, rds_path, epic_dir) for epic_id, rds_path, epic_dir in rds_list if os.path.exists(rds_path)]
    
    if not valid_rds:
        logging.warning("No valid RDS files available for plotting")
        return
    
    # Create temporary JSON file for R input
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as json_tmp:
        json_path = json_tmp.name
    
    try:
        json_dict = {
            "R_script": r_script,
            "infile": json_path,
            "rds_files": [str(os.path.abspath(rds_path)) for _, rds_path, _ in valid_rds],
            "out_dirs": [str(os.path.abspath(epic_dir)) for _, _, epic_dir in valid_rds],
            "epic_ids": [epic_id for epic_id, _, _ in valid_rds],
            "summary_dir": str(os.path.abspath(outdir)),
        }
        
        exec_R(json_dict)
        logging.info(f"Successfully generated EPIC plots for {len(valid_rds)} samples + summary plots")
    except Exception as e:
        logging.error(f"Failed to generate EPIC plots: {e}")
    finally:
        # Clean up JSON file
        if os.path.exists(json_path):
            os.remove(json_path)


def _plot_correlation_summary(report_rows: List[Dict], outdir: str) -> None:
    """
    Generate boxplot correlation summary visualization showing distribution of
    Pearson correlations for bins and segments across all samples.
    
    Args:
        report_rows: List of dicts with correlation stats per sample
        outdir: Output directory for the plot
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        df = pd.DataFrame(report_rows)
        if df.empty:
            logging.warning("No correlation data available for summary plot")
            return
        
        # Extract correlations (backwards compatible: keys may be missing)
        bins_pearson = df["bins_pearson"].dropna().values if "bins_pearson" in df.columns else np.array([])
        segments_pearson = df["segments_pearson"].dropna().values if "segments_pearson" in df.columns else np.array([])
        
        if len(bins_pearson) == 0 and len(segments_pearson) == 0:
            logging.warning("No valid correlation values for summary plot")
            return
        
        # Create figure with boxplots (bins and segments only; aberrant handled separately)
        n_panels = 2
        fig, axes = plt.subplots(1, n_panels, figsize=(10, 5))
        if n_panels == 1:
            axes = [axes]
        
        def _stats_text(vals: np.ndarray) -> str:
            mean_v = float(np.mean(vals)) if len(vals) else float("nan")
            median_v = float(np.median(vals)) if len(vals) else float("nan")
            std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            return f"mean={mean_v:.3f}\nmedian={median_v:.3f}\nsd={std_v:.3f}"

        # Bins boxplot
        if len(bins_pearson) > 0:
            bp1 = axes[0].boxplot([bins_pearson], labels=['cfRRBS vs EPIC'], patch_artist=True,
                                   widths=0.5, showmeans=True)
            bp1['boxes'][0].set_facecolor('#3498db')
            bp1['boxes'][0].set_alpha(0.7)
            axes[0].scatter([1]*len(bins_pearson), bins_pearson, alpha=0.4, s=30, color='#2c3e50', zorder=3)
            axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            axes[0].set_ylabel('Pearson Correlation', fontsize=12, fontweight='bold')
            axes[0].set_title(f'Bins Correlation (n={len(bins_pearson)})', fontsize=13, fontweight='bold')
            axes[0].set_ylim([-1.1, 1.1])
            axes[0].grid(axis='y', alpha=0.3, linestyle=':')
            axes[0].text(
                0.03, 0.97, _stats_text(bins_pearson),
                transform=axes[0].transAxes,
                va='top', ha='left', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#3498db')
            )
        else:
            axes[0].text(0.5, 0.5, 'No bins data', ha='center', va='center', transform=axes[0].transAxes)
        
        # Segments boxplot
        seg_ax_idx = 1
        if len(segments_pearson) > 0:
            bp2 = axes[seg_ax_idx].boxplot([segments_pearson], labels=['cfRRBS vs EPIC'], patch_artist=True,
                                   widths=0.5, showmeans=True)
            bp2['boxes'][0].set_facecolor('#e74c3c')
            bp2['boxes'][0].set_alpha(0.7)
            axes[seg_ax_idx].scatter([1]*len(segments_pearson), segments_pearson, alpha=0.4, s=30, color='#2c3e50', zorder=3)
            axes[seg_ax_idx].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            axes[seg_ax_idx].set_ylabel('Pearson Correlation', fontsize=12, fontweight='bold')
            axes[seg_ax_idx].set_title(f'Segments Correlation (n={len(segments_pearson)})', fontsize=13, fontweight='bold')
            axes[seg_ax_idx].set_ylim([-1.1, 1.1])
            axes[seg_ax_idx].grid(axis='y', alpha=0.3, linestyle=':')
            axes[seg_ax_idx].text(
                0.03, 0.97, _stats_text(segments_pearson),
                transform=axes[seg_ax_idx].transAxes,
                va='top', ha='left', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#e74c3c')
            )
        else:
            axes[seg_ax_idx].text(0.5, 0.5, 'No segments data', ha='center', va='center', transform=axes[seg_ax_idx].transAxes)
        
        plt.suptitle('EPIC vs cfRRBS Correlation Summary', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = os.path.join(outdir, "corr_summary_boxplot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Correlation summary plot saved to {output_path}")
        
    except Exception as e:
        logging.error(f"Failed to create correlation summary plot: {e}")
        raise


def _stack_pair_plots(epic_png: str, cfrrbs_png: str, out_png: str, title: str) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        logging.warning("Pillow not available; skipping stacked plot")
        return

    if not os.path.exists(epic_png):
        logging.warning("Missing EPIC plot for stacking: %s", epic_png)
        return
    if not os.path.exists(cfrrbs_png):
        logging.warning("Missing cfRRBS plot for stacking: %s", cfrrbs_png)
        return

    epic_img = Image.open(epic_png).convert("RGB")
    cfrrbs_img = Image.open(cfrrbs_png).convert("RGB")

    target_width = cfrrbs_img.width
    if epic_img.width != target_width:
        scale = target_width / epic_img.width
        epic_img = epic_img.resize(
            (target_width, int(epic_img.height * scale)), Image.LANCZOS
        )

    def _trim_whitespace(img):
        """Return bounding box of non-white content (with small padding)."""
        from PIL import ImageChops
        bg = Image.new(img.mode, img.size, (255, 255, 255))
        diff = ImageChops.difference(img, bg)
        bbox = diff.getbbox()
        if bbox is None:
            return (0, 0, img.width, img.height)
        pad = 4
        return (
            max(bbox[0] - pad, 0),
            max(bbox[1] - pad, 0),
            min(bbox[2] + pad, img.width),
            min(bbox[3] + pad, img.height),
        )

    # Crop vertical whitespace from each image
    e_box = _trim_whitespace(epic_img)
    c_box = _trim_whitespace(cfrrbs_img)

    # Use shared left/right bounds (tightest common crop to remove right whitespace)
    left = min(e_box[0], c_box[0])
    right = max(e_box[2], c_box[2])

    epic_cropped = epic_img.crop((left, e_box[1], right, e_box[3]))
    cfrrbs_cropped = cfrrbs_img.crop((left, c_box[1], right, c_box[3]))

    title_height = 18
    gap = 6  # small gap between the two panels
    stacked_width = epic_cropped.width
    stacked_height = title_height + epic_cropped.height + gap + cfrrbs_cropped.height
    stacked = Image.new("RGB", (stacked_width, stacked_height), color="white")

    draw = ImageDraw.Draw(stacked)
    font = ImageFont.load_default()
    draw.text((10, 2), title, fill="black", font=font)

    stacked.paste(epic_cropped, (0, title_height))
    stacked.paste(cfrrbs_cropped, (0, title_height + epic_cropped.height + gap))

    stacked.save(out_png)


def _regenerate_cfrrbs_genome_plot(cfrrbs_outid: str, wd: str) -> None:
    """Re-run plotter_conumee.R so genome_wide.png picks up focal gene labels.
    Uses the focal gene TSVs already written by predict_output.py (segment-based gold standard).
    Does NOT overwrite those files.
    """
    plot_input_json = os.path.join(f"{cfrrbs_outid}.plots", "plot_input.json")
    if not os.path.exists(plot_input_json):
        logging.warning(f"Plot input JSON not found: {plot_input_json}; skipping cfRRBS genome plot regeneration")
        return

    import json as _json, tempfile
    from wisecondorx.overall_tools import exec_R

    try:
        with open(plot_input_json) as f:
            plot_dict = _json.load(f)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        plot_dict["infile"] = tmp_path
        plot_dict["R_script"] = os.path.join(wd, "include", "plotter_conumee.R")
        exec_R(plot_dict)
        logging.info(f"Regenerated cfRRBS genome plot for {cfrrbs_outid}")
    except Exception as e:
        logging.warning(f"Failed to regenerate cfRRBS genome plot: {e}")


def _annotate_epic_genome_plots(samples: List[Dict], wd: str) -> None:
    """Re-draw CNV_genomeplot.png for each EPIC sample with broad gene calls annotated in purple.

    Args:
        samples: list of dicts with keys: rds_path, out_dir, epic_id,
                 broad_amp_genes (list[str]), broad_del_genes (list[str])
        wd: wisecondorx package directory
    """
    if not samples:
        return

    import json as _json, tempfile
    from wisecondorx.overall_tools import exec_R

    r_script = os.path.join(wd, "include", "annotate_epic_genomeplot.R")
    if not os.path.exists(r_script):
        logging.warning(f"annotate_epic_genomeplot.R not found: {r_script}")
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json_path = tmp.name

    try:
        json_dict = {
            "R_script": r_script,
            "infile": json_path,
            "samples": samples,
        }
        exec_R(json_dict)
        logging.info(f"EPIC genomeplot annotations applied to {len(samples)} sample(s)")
    except Exception as e:
        logging.warning(f"Failed to annotate EPIC genomeplots: {e}")
    finally:
        if os.path.exists(json_path):
            try:
                os.remove(json_path)
            except Exception:
                pass


def _collect_genome_wide_stacked_plots(outdir: str, pair_metadata: List[Dict]) -> None:
    """
    Collect all genome-wide CNV plots (epic, cfrrbs, stacked) from sample pairs
    and organize them in dedicated subfolders with pair ID as filename.
    
    Args:
        outdir: Output directory (parent of 'samples' and 'summary' folders)
        pair_metadata: List of metadata dicts with 'epic_id', 'cfrrbs_id', 'sample_pair_dir', 'epic_dir', 'cfrrbs_dir'
    """
    # Create main genome_wide_CNV directory and subfolders
    genome_wide_dir = os.path.join(outdir, "genome_wide_CNV")
    epic_subdir = os.path.join(genome_wide_dir, "epic")
    cfrrbs_subdir = os.path.join(genome_wide_dir, "cfrrbs")
    stacked_subdir = os.path.join(genome_wide_dir, "stacked")
    
    os.makedirs(epic_subdir, exist_ok=True)
    os.makedirs(cfrrbs_subdir, exist_ok=True)
    os.makedirs(stacked_subdir, exist_ok=True)
    
    collected_epic = 0
    collected_cfrrbs = 0
    collected_stacked = 0
    
    for metadata in pair_metadata:
        epic_id = metadata.get("epic_id", "unknown")
        cfrrbs_id = metadata.get("cfrrbs_id", "unknown")
        sample_pair_dir = metadata.get("sample_pair_dir")
        epic_dir = metadata.get("epic_dir")
        cfrrbs_dir = metadata.get("cfrrbs_dir")
        pair_id = f"{epic_id}__{cfrrbs_id}"
        
        # Copy EPIC genome-wide plot
        if epic_dir and os.path.exists(epic_dir):
            # EPIC plot is CNV_genomeplot.png in epic_dir
            epic_plot = os.path.join(epic_dir, "CNV_genomeplot.png")
            if os.path.exists(epic_plot):
                output_filename = f"{pair_id}.png"
                output_path = os.path.join(epic_subdir, output_filename)
                try:
                    shutil.copy2(epic_plot, output_path)
                    logging.debug(f"Copied EPIC plot for {pair_id}")
                    collected_epic += 1
                except Exception as e:
                    logging.warning(f"Failed to copy EPIC plot for {pair_id}: {e}")
        
        # Copy cfRRBS genome-wide plot
        if cfrrbs_dir and os.path.exists(cfrrbs_dir):
            # cfRRBS plot is at {cfrrbs_id}.plots/genome_wide.png
            cfrrbs_plot = os.path.join(cfrrbs_dir, f"{cfrrbs_id}.plots", "genome_wide.png")
            if os.path.exists(cfrrbs_plot):
                output_filename = f"{pair_id}.png"
                output_path = os.path.join(cfrrbs_subdir, output_filename)
                try:
                    shutil.copy2(cfrrbs_plot, output_path)
                    logging.debug(f"Copied cfRRBS plot for {pair_id}")
                    collected_cfrrbs += 1
                except Exception as e:
                    logging.warning(f"Failed to copy cfRRBS plot for {pair_id}: {e}")
        
        # Copy stacked genome-wide plot
        if sample_pair_dir and os.path.exists(sample_pair_dir):
            stacked_plot = os.path.join(sample_pair_dir, "CNV_genomewide_stacked.png")
            if os.path.exists(stacked_plot):
                output_filename = f"{pair_id}.png"
                output_path = os.path.join(stacked_subdir, output_filename)
                try:
                    shutil.copy2(stacked_plot, output_path)
                    logging.debug(f"Copied stacked plot for {pair_id}")
                    collected_stacked += 1
                except Exception as e:
                    logging.warning(f"Failed to copy stacked plot for {pair_id}: {e}")
    
    logging.info(f"Genome-wide plots collected: {collected_epic} EPIC, {collected_cfrrbs} cfRRBS, {collected_stacked} stacked "
                 f"to {genome_wide_dir}")


def _load_expected_alterations_from_regions(regions_bed_path: str = None) -> Dict[str, str]:
    """
    Load expected gene alterations from regions BED file (optional 4th column).
    Format: chr, start, end, gene, alteration(amp/del/gain/deletion)
    
    Args:
        regions_bed_path: Path to regions bed file (optional)
    
    Returns:
        Dict mapping gene name (lowercase) -> "amp" or "del" (or empty dict if file not provided)
    """
    if not regions_bed_path or not os.path.exists(regions_bed_path):
        return {}
    
    expected = {}
    try:
        with open(regions_bed_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    gene = parts[3].strip()
                    alteration = parts[4].strip().lower()
                    if alteration in ('amp', 'del', 'gain', 'deletion'):
                        # Normalize alteration types
                        alteration = 'amp' if alteration in ('amp', 'gain') else 'del'
                        expected[gene.lower()] = alteration
        if expected:
            logging.info(f"Loaded {len(expected)} expected gene alterations from --regions file")
    except Exception as e:
        logging.debug(f"No expected alterations found in regions file: {e}")
    
    return expected


def _aggregate_gene_concordance_stats(records: List[Dict], call_type: str, expected_alterations: Dict[str, str] = None) -> pd.DataFrame:
    """
    Aggregate concordance records by gene to compute overall statistics.
    
    For each gene, counts:
    - How many times EPIC only (gain or deletion separately)
    - How many times cfRRBS only (gain or deletion separately)
    - How many times both same direction (both gain or both deletion)
    - How many times both opposite direction
    - Percentages for each category
    
    Args:
        records: List of dicts with keys: gene, sample_id, epic_call, cfrrbs_call, concordance
        call_type: "broad" or "focal" (for labeling)
        expected_alterations: Optional dict mapping gene (lowercase) -> "amp" or "del"
    
    Returns:
        DataFrame with gene-level concordance statistics
    """
    if not records:
        return pd.DataFrame()
    
    if expected_alterations is None:
        expected_alterations = {}
    
    df = pd.DataFrame(records)
    
    # Aggregate by gene
    gene_stats = []
    for gene in sorted(df["gene"].unique()):
        gene_df = df[df["gene"] == gene]
        n_total = len(gene_df)
        
        # Count concordance categories (without correctness/missed concepts)
        both_gain = sum(gene_df["concordance"] == "both_gain")
        both_del = sum(gene_df["concordance"] == "both_del")
        epic_gain_only = sum(gene_df["concordance"] == "epic_gain_only")
        epic_del_only = sum(gene_df["concordance"] == "epic_del_only")
        cfrrbs_gain_only = sum(gene_df["concordance"] == "cfrrbs_gain_only")
        cfrrbs_del_only = sum(gene_df["concordance"] == "cfrrbs_del_only")
        discordant_epic_gain = sum(gene_df["concordance"] == "discordant_epic_gain")
        discordant_epic_del = sum(gene_df["concordance"] == "discordant_epic_del")
        neutral_both = sum(gene_df["concordance"] == "neutral")
        
        # Aggregate into broader categories (without correctness notion)
        epic_only = epic_gain_only + epic_del_only
        cfrrbs_only = cfrrbs_gain_only + cfrrbs_del_only
        both_concordant = both_gain + both_del  # same direction
        both_discordant = discordant_epic_gain + discordant_epic_del  # opposite direction
        
        # Lookup expected alteration
        expected_alt = expected_alterations.get(gene.lower(), "")
        
        gene_stats.append({
            "gene": gene,
            "expected_alteration": expected_alt,
            "call_type": call_type,
            "n_samples": n_total,
            "epic_only_count": epic_only,
            "epic_only_pct": 100 * epic_only / n_total if n_total > 0 else 0,
            "cfrrbs_only_count": cfrrbs_only,
            "cfrrbs_only_pct": 100 * cfrrbs_only / n_total if n_total > 0 else 0,
            "both_same_dir_count": both_concordant,
            "both_same_dir_pct": 100 * both_concordant / n_total if n_total > 0 else 0,
            "both_opp_dir_count": both_discordant,
            "both_opp_dir_pct": 100 * both_discordant / n_total if n_total > 0 else 0,
            "neutral_both_count": neutral_both,
            "neutral_both_pct": 100 * neutral_both / n_total if n_total > 0 else 0,
            "epic_gain_only": epic_gain_only,
            "epic_del_only": epic_del_only,
            "cfrrbs_gain_only": cfrrbs_gain_only,
            "cfrrbs_del_only": cfrrbs_del_only,
            "both_gain": both_gain,
            "both_del": both_del,
            "discordant_epic_gain": discordant_epic_gain,
            "discordant_epic_del": discordant_epic_del,
        })
    
    return pd.DataFrame(gene_stats)


def _plot_gene_concordance_table(gene_stats: pd.DataFrame, call_type: str, outdir: str) -> None:
    """
    Render gene_stats as a prettified PNG table that mirrors the concordance_summary.csv.

    All columns from the DataFrame are shown (minus call_type, which is in the title).
    Numeric cells are shaded with a single blue gradient scaled to each column's max.
    The expected_alteration column is green for 'amp' and red for 'del'.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba
    except ImportError:
        logging.warning("matplotlib not available; skipping gene concordance table")
        return

    if gene_stats.empty:
        return

    df = gene_stats.sort_values("n_samples", ascending=False).reset_index(drop=True)
    # Drop call_type — redundant with title
    df = df.drop(columns=["call_type"], errors="ignore")

    # Round pct columns for display
    pct_cols = [c for c in df.columns if c.endswith("_pct")]
    for c in pct_cols:
        df[c] = df[c].round(1)

    col_names = list(df.columns)
    n_cols = len(col_names)
    n_rows = len(df)

    # Pre-compute per-column max for numeric columns (used for gradient scaling)
    col_max = {}
    for c in col_names:
        if pd.api.types.is_numeric_dtype(df[c]):
            m = df[c].max()
            col_max[c] = m if m > 0 else 1.0

    # Format cell text
    table_rows = []
    for _, row in df.iterrows():
        cells = []
        for c in col_names:
            v = row[c]
            if pd.isna(v) or v == "":
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v:.1f}")
            else:
                cells.append(str(v))
        table_rows.append(cells)

    # Figure sizing: wider for many columns
    fig_h = max(3.0, n_rows * 0.45 + 1.5)
    fig_w = max(10, n_cols * 1.1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # Equal-width columns; gene column slightly wider
    gene_col_idx = col_names.index("gene") if "gene" in col_names else 0
    col_widths = [1.0] * n_cols
    col_widths[gene_col_idx] = 2.0
    total_w = sum(col_widths)
    col_widths = [w / total_w for w in col_widths]

    tbl = ax.table(
        cellText=table_rows,
        colLabels=col_names,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 2.0)

    # Header styling
    for j in range(n_cols):
        cell = tbl[(0, j)]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(weight="bold", color="white", fontsize=8)

    # Gradient color: single steelblue hue
    GRAD_R, GRAD_G, GRAD_B, _ = to_rgba("#2980b9")
    EXP_COLORS = {"amp": "#d5f5e3", "del": "#fadbd8"}  # light green / light red

    exp_col_idx = col_names.index("expected_alteration") if "expected_alteration" in col_names else None

    for i in range(1, n_rows + 1):
        row_bg = "#f2f3f4" if i % 2 == 0 else "white"
        for j, col in enumerate(col_names):
            cell = tbl[(i, j)]
            raw = table_rows[i - 1][j]

            if j == exp_col_idx:
                bg = EXP_COLORS.get(str(raw).strip().lower(), row_bg)
                cell.set_facecolor(bg)
                cell.set_text_props(fontsize=8, weight="bold" if raw else "normal")
            elif col in col_max:
                try:
                    v = float(raw) if raw else 0.0
                except ValueError:
                    v = 0.0
                alpha = min(0.85, 0.08 + 0.77 * (v / col_max[col])) if v > 0 else 0.0
                if alpha > 0:
                    cell.set_facecolor((GRAD_R, GRAD_G, GRAD_B, alpha))
                    cell.set_text_props(fontsize=8,
                                        color="white" if alpha > 0.55 else "black")
                else:
                    cell.set_facecolor(row_bg)
                    cell.set_text_props(fontsize=8, color="#999999")
            else:
                cell.set_facecolor(row_bg)
                cell.set_text_props(fontsize=8,
                                    weight="bold" if col == "gene" else "normal",
                                    ha="left" if col == "gene" else "center")

    ax.set_title(
        f"Gene concordance — {call_type} calls",
        fontsize=11, fontweight="bold", pad=10,
    )
    plt.tight_layout()
    out_path = os.path.join(outdir, f"concordance_table_{call_type}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    logging.info(f"Gene concordance table saved to {out_path}")




def _create_gene_concordance_csv(gene_stats: List[pd.DataFrame], outdir: str) -> None:
    """
    Create a comprehensive CSV file with gene-level concordance statistics
    for both broad and focal calls.
    
    Args:
        gene_stats: List of DataFrames [broad_stats, focal_stats]
        outdir: Output directory
    """
    # Combine broad and focal stats
    combined = pd.concat(gene_stats, ignore_index=True)
    
    if combined.empty:
        logging.warning("No gene concordance data to export to CSV")
        return
    
    # Select and order columns for output
    output_cols = [
        "gene",
        "expected_alteration",
        "call_type",
        "n_samples",
        "epic_only_count",
        "epic_only_pct",
        "cfrrbs_only_count",
        "cfrrbs_only_pct",
        "both_same_dir_count",
        "both_same_dir_pct",
        "both_opp_dir_count",
        "both_opp_dir_pct",
        "neutral_both_count",
        "neutral_both_pct",
        "epic_gain_only",
        "epic_del_only",
        "cfrrbs_gain_only",
        "cfrrbs_del_only",
        "both_gain",
        "both_del",
        "discordant_epic_gain",
        "discordant_epic_del",
    ]
    
    # Filter to only columns that exist in the dataframe
    output_cols = [c for c in output_cols if c in combined.columns]
    output_df = combined[output_cols].copy()
    
    # Round percentage columns to 1 decimal place
    pct_cols = [c for c in output_df.columns if "_pct" in c]
    for col in pct_cols:
        output_df[col] = output_df[col].round(1)
    
    # Sort by call_type then by number of samples (descending) then by gene name
    output_df = output_df.sort_values(
        ["call_type", "n_samples", "gene"],
        ascending=[True, False, True]
    )
    
    output_path = os.path.join(outdir, "concordance_summary.csv")
    output_df.to_csv(output_path, index=False)

    logging.info(f"Gene concordance CSV summary saved to {output_path}")


def tool_epic_cfrrbs(args: argparse.Namespace) -> None:
    logging.info("Starting EPIC vs cfRRBS correlation")

    wd = str(os.path.dirname(os.path.realpath(__file__)))
    pairs = _read_pairs_from_sheet(args.sample_sheet)
    max_replicates = getattr(args, "max_replicates", None)
    if max_replicates is not None and max_replicates <= 0:
        raise ValueError("--max-replicates must be a positive integer")

    ref_file = np.load(args.reference, encoding="latin1", allow_pickle=True)
    binsize = int(ref_file["binsize"])
    del ref_file

    out_dir = os.path.abspath(args.outdir)
    # Handle overwrite option: remove existing outdir content if requested
    summary_dir = os.path.join(out_dir, "summary")
    samples_dir = os.path.join(out_dir, "samples")
    if getattr(args, "overwrite", False):
        if os.path.exists(out_dir):
            try:
                shutil.rmtree(out_dir)
                logging.info(f"Removed existing output directory because --overwrite set: {out_dir}")
            except Exception as e:
                logging.warning(f"Failed to remove existing output directory {out_dir}: {e}")
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    # ===================================================================
    # PHASE 1: Create all pair folders
    # ===================================================================
    logging.info("Phase 1: Creating pair folders")
    pair_metadata = []  # Store metadata for each pair for later processing

    for epic_id, cfrrbs_path, cfrrbs_id, rds_path in pairs:
        # Validate input files exist
        if not os.path.exists(cfrrbs_path):
            logging.warning("Missing cfRRBS NPZ file: %s", cfrrbs_path)
            continue
        if not os.path.exists(rds_path):
            logging.warning("Missing conumee RDS file: %s", rds_path)
            continue
        # Create sample-specific output directory
        pair_id = f"{epic_id}__{cfrrbs_id}"
        sample_pair_dir = os.path.join(samples_dir, pair_id)
        os.makedirs(sample_pair_dir, exist_ok=True)
        epic_dir = os.path.join(sample_pair_dir, "epic")
        cfrrbs_dir = os.path.join(sample_pair_dir, "cfrrbs")

        os.makedirs(epic_dir, exist_ok=True)
        os.makedirs(cfrrbs_dir, exist_ok=True)
        
        # Collect metadata for later processing
        metadata = {
            "epic_id": epic_id,
            "cfrrbs_id": cfrrbs_id,
            "pair_id": pair_id,
            "cfrrbs_path": cfrrbs_path,
            "rds_path": rds_path,
            "sample_pair_dir": sample_pair_dir,
            "epic_dir": epic_dir,
            "cfrrbs_dir": cfrrbs_dir,
        }
        pair_metadata.append(metadata)
        if max_replicates is not None and len(pair_metadata) >= max_replicates:
            logging.info(
                "Reached --max-replicates=%d eligible pairs; stopping pair collection",
                max_replicates,
            )
            break
    
    logging.info(f"Created folders for {len(pair_metadata)} pairs")
    
    if not pair_metadata:
        logging.warning("No valid pairs found; exiting")
        return

    # ===================================================================
    # PHASE 2: Batch process all EPIC data (genomeplot + summary plots)
    # ===================================================================
    logging.info("Phase 2: Processing EPIC RDS files (batch)")
    epic_rds_list = [(m["epic_id"], m["rds_path"], m["epic_dir"]) for m in pair_metadata]
    _plot_epic_summary(epic_rds_list, summary_dir)

    # Remove EPIC focal helper tables from per-sample outputs
    for metadata in pair_metadata:
        focal_probes_path = os.path.join(metadata["epic_dir"], f"{metadata['epic_id']}_focal_probes.tsv")
        focal_ratio_path = os.path.join(metadata["epic_dir"], f"{metadata['epic_id']}_focal_ratio.tsv")
        for p in [focal_probes_path, focal_ratio_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception as e:
                    logging.debug(f"Could not remove focal output {p}: {e}")

    # Note: sWGS support removed — no Phase 2.5 processing

    # ===================================================================
    # PHASE 3: Process each pair's cfRRBS data and correlations
    # ===================================================================
    logging.info("Phase 3: Processing cfRRBS data sample-wise")
    
    report_rows: List[Dict] = []
    all_focal_records: List[Dict] = []

    for metadata in pair_metadata:
        epic_id = metadata["epic_id"]
        cfrrbs_id = metadata["cfrrbs_id"]
        pair_id = metadata["pair_id"]
        cfrrbs_path = metadata["cfrrbs_path"]
        rds_path = metadata["rds_path"]
        sample_pair_dir = metadata["sample_pair_dir"]
        epic_dir = metadata["epic_dir"]
        cfrrbs_dir = metadata["cfrrbs_dir"]
        
        logging.info(f"Processing cfRRBS for {pair_id}")
        
        # Temporary output location for WisecondorX predict
        temp_outid = os.path.join(cfrrbs_dir, cfrrbs_id)

        logging.info(f"Running WisecondorX predict for {cfrrbs_id}")
        _run_cfrrbs_predict(
            cfrrbs_path,
            args.reference,
            temp_outid,
            blacklist=getattr(args, "blacklist", None),
            regions=getattr(args, "regions", None),
            normalization_method=getattr(args, "normalization_method", "reference"),
        )

        cf_bins_path = temp_outid + "_bins.bed"
        if not os.path.exists(cf_bins_path):
            logging.warning("Missing cfRRBS bins: %s", cf_bins_path)
            continue

        try:
            cf_bins = _load_bins_bed(cf_bins_path)
            epic_bins, epic_segments = _load_epic_bins_segments_from_tsv(epic_dir, epic_id)
        except Exception as e:
            logging.error(f"Failed to load data for {cfrrbs_id}: {e}")
            continue

        logging.debug(f"Loaded cfRRBS: {len(cf_bins)} bins, chrs={sorted(cf_bins['chr'].unique().tolist())}")
        logging.debug(f"Loaded EPIC: {len(epic_bins)} bins, chrs={sorted(epic_bins['chr'].unique().tolist())}")

        # sWGS support removed — skip sWGS loading

        # Aggregate EPIC bins to cfRRBS bin boundaries for better pairing
        logging.debug(f"Aggregating {len(epic_bins)} EPIC bins to {len(cf_bins)} cfRRBS bin boundaries")
        aggregated_epic_bins = _aggregate_epic_bins_to_cfrrbs(epic_bins, cf_bins)
        logging.debug(f"Aggregated to {len(aggregated_epic_bins)} bins")

        # Pair and plot bins
        corr_bins = _compute_correlations(cf_bins, aggregated_epic_bins)
        merged = _pair_bins_by_overlap(cf_bins, aggregated_epic_bins)
        logging.debug(f"Paired bins before dropna: {len(merged)} rows")
        merged = merged.dropna(subset=["ratio_cf", "ratio_epic"])
        logging.debug(f"Paired bins after dropna: {len(merged)} rows")
        plot_path = os.path.join(sample_pair_dir, "corr_bins_scatter.png")
        _plot_scatter(merged, plot_path, title_prefix="Bins - ", corr=corr_bins)

        cf_segments_path = temp_outid + "_segments.bed"
        cf_aberrations_path = temp_outid + "_aberrations.bed"
        if os.path.exists(cf_segments_path):
            cf_segments = _load_segments_bed(cf_segments_path)
        else:
            cf_segments = pd.DataFrame(columns=["chr", "start", "end", "ratio"])

        try:
            cf_aberrations = _load_aberrations_bed(cf_aberrations_path)
        except Exception as e:
            logging.warning(f"Failed to load cfRRBS aberrations for {cfrrbs_id}: {e}")
            cf_aberrations = pd.DataFrame(columns=["chr", "start", "end", "type", "ratio", "zscore", "genes", "pval"])

        # Pair segments and plot
        corr_segments = _compute_segment_correlations(cf_segments, epic_segments)
        merged_segments = _pair_segments_by_overlap(cf_segments, epic_segments)
        if not merged_segments.empty:
            merged_segments = merged_segments.dropna(subset=["ratio_cf", "ratio_epic"])
            logging.debug(f"Paired segments: {len(merged_segments)} rows")

            # Save paired segments to a TSV in the sample pair folder for downstream inspection
            paired_segments_path = os.path.join(sample_pair_dir, "paired_segments.tsv")
            try:
                merged_segments.to_csv(paired_segments_path, sep="\t", index=False)
                logging.info(f"Paired segments TSV saved to {paired_segments_path}")
            except Exception as e:
                logging.warning(f"Failed to write paired segments TSV: {e}")

            segments_plot_path = os.path.join(sample_pair_dir, "corr_segments_scatter.png")
            _plot_scatter(merged_segments, segments_plot_path, title_prefix="Segments - ", corr=corr_segments)



        pair_corr_path = os.path.join(sample_pair_dir, "correlation.tsv")
        pd.DataFrame(
            [
                {
                    "level": "bins",
                    "pearson": corr_bins["pearson"],
                    "pearson_p": corr_bins["pearson_p"],
                    "spearman": corr_bins["spearman"],
                    "spearman_p": corr_bins["spearman_p"],
                    "n": corr_bins["n_bins"],
                },
                {
                    "level": "segments",
                    "pearson": corr_segments["pearson"],
                    "pearson_p": corr_segments["pearson_p"],
                    "spearman": corr_segments["spearman"],
                    "spearman_p": corr_segments["spearman_p"],
                    "n": corr_segments["n_segments"],
                },
            ]
        ).to_csv(pair_corr_path, sep="\t", index=False)

        # ── Gene calls: focal only (cfRRBS Python / EPIC R) ─────────────────────
        detail_cf_path = temp_outid + "_regions.bed"
        cf_detail_df = pd.DataFrame()
        if os.path.exists(detail_cf_path):
            try:
                cf_detail_df = pd.read_csv(detail_cf_path, sep="\t")
                if "name" not in cf_detail_df.columns and cf_detail_df.shape[1] >= 4:
                    cf_detail_df.columns = ["chr", "start", "end", "name"] + list(cf_detail_df.columns[4:])
                if "ratio" not in cf_detail_df.columns:
                    numeric_cols = [c for c in cf_detail_df.select_dtypes(include=["number"]).columns
                                    if c not in ["start", "end"]]
                    if numeric_cols:
                        cf_detail_df["ratio"] = cf_detail_df[numeric_cols[0]]
                cf_detail_df = cf_detail_df.rename(
                    columns={"seqnames": "chr", "Chromosome": "chr"}, errors="ignore")
            except Exception as e:
                logging.warning(f"[{pair_id}] Failed to load cfRRBS detail: {e}")
                cf_detail_df = pd.DataFrame()

        cf_broad_df = pd.DataFrame()  # broad calling removed
        cf_focal_df = _call_focal_gene_events(cf_detail_df, cf_segments, cf_bins, conf=0.99)
        logging.info(
            f"[{pair_id}] cfRRBS focal: gain={sum(cf_focal_df['focal_call']=='gain')}, "
            f"deletion={sum(cf_focal_df['focal_call']=='deletion')}"
        )

        # EPIC detail
        epic_detail_df = pd.DataFrame()
        detail_epic_path = None
        candidates = [f for f in os.listdir(epic_dir)
                      if "detail" in f.lower() and f.lower().endswith(".tsv")]
        exact = f"{epic_id}_detail.tsv"
        if exact in candidates:
            detail_epic_path = os.path.join(epic_dir, exact)
        else:
            non_del = [f for f in candidates if "del" not in f.lower()]
            if non_del:
                detail_epic_path = os.path.join(epic_dir, non_del[0])
            elif candidates:
                detail_epic_path = os.path.join(epic_dir, candidates[0])

        if detail_epic_path and os.path.exists(detail_epic_path):
            try:
                epic_detail_df = pd.read_csv(detail_epic_path, sep="\t")
                col_map = _find_epic_detail_columns(list(epic_detail_df.columns))
                rename_map = {v: k for k, v in col_map.items()
                              if k in ("chr", "start", "end", "name", "ratio")}
                epic_detail_df = epic_detail_df.rename(columns=rename_map, errors="ignore")
                if "ratio" not in epic_detail_df.columns:
                    numeric_cols = [c for c in epic_detail_df.select_dtypes(include=["number"]).columns
                                    if c not in ["start", "end"]]
                    if numeric_cols:
                        epic_detail_df["ratio"] = epic_detail_df[numeric_cols[0]]
                # region column → name if not genomic coords
                if "region" in epic_detail_df.columns and "name" not in epic_detail_df.columns:
                    if not epic_detail_df["region"].astype(str).str.contains(":").any():
                        epic_detail_df["name"] = epic_detail_df["region"].astype(str)
            except Exception as e:
                logging.warning(f"[{pair_id}] Failed to load EPIC detail: {e}")
                epic_detail_df = pd.DataFrame()

        # EPIC focal calls from R (conumee2 CNV.focal output)
        epic_focal_gains: Set[str] = set()
        epic_focal_dels: Set[str] = set()
        for p in [os.path.join(epic_dir, f"{epic_id}_amp_detail_regions.tsv"),
                  os.path.join(epic_dir, f"{epic_id}_amplified_genes.tsv")]:
            if os.path.exists(p):
                try:
                    tmp = pd.read_csv(p, sep="\t")
                    gene_col = next((c for c in ["region", "gene", "name"] if c in tmp.columns), None)
                    if gene_col:
                        epic_focal_gains.update(
                            tmp[gene_col].dropna().astype(str).str.strip().tolist())
                except Exception as e:
                    logging.warning(f"[{pair_id}] EPIC amp focal read failed {p}: {e}")
        for p in [os.path.join(epic_dir, f"{epic_id}_del_detail_regions.tsv"),
                  os.path.join(epic_dir, f"{epic_id}_deleted_genes.tsv")]:
            if os.path.exists(p):
                try:
                    tmp = pd.read_csv(p, sep="\t")
                    gene_col = next((c for c in ["region", "gene", "name"] if c in tmp.columns), None)
                    if gene_col:
                        epic_focal_dels.update(
                            tmp[gene_col].dropna().astype(str).str.strip().tolist())
                except Exception as e:
                    logging.warning(f"[{pair_id}] EPIC del focal read failed {p}: {e}")
        logging.info(
            f"[{pair_id}] EPIC focal R output: amp={len(epic_focal_gains)}, del={len(epic_focal_dels)}"
        )

        def _concordance(epic_call: str, cf_call: str) -> str:
            e, c = epic_call, cf_call
            if e == "neutral" and c == "neutral":
                return "neutral"
            if e == "neutral":
                return "cfrrbs_gain_only" if c == "gain" else "cfrrbs_del_only"
            if c == "neutral":
                return "epic_gain_only" if e == "gain" else "epic_del_only"
            if e == c:
                return "both_gain" if e == "gain" else "both_del"
            return "discordant_epic_gain" if e == "gain" else "discordant_epic_del"

        # Per-pair focal concordance
        epic_focal_gains_lower = {g.lower() for g in epic_focal_gains}
        epic_focal_dels_lower = {g.lower() for g in epic_focal_dels}
        focal_rows = []
        for _, cf_r in cf_focal_df.iterrows():
            gene_name = str(cf_r.get("gene", "")).strip()
            gene_lower = gene_name.lower()
            cf_call = cf_r.get("focal_call", "neutral")
            if gene_lower in epic_focal_gains_lower:
                ep_call = "gain"
            elif gene_lower in epic_focal_dels_lower:
                ep_call = "deletion"
            else:
                ep_call = "neutral"
            focal_rows.append({
                "gene": gene_name,
                "chr": cf_r.get("chr"),
                "start": cf_r.get("start"),
                "end": cf_r.get("end"),
                "cfrrbs_ratio": cf_r.get("ratio", float("nan")),
                "cfrrbs_focal": cf_call,
                "epic_focal": ep_call,
                "concordant": _concordance(ep_call, cf_call),
            })
        focal_concordance_df = pd.DataFrame(focal_rows)
        focal_concordance_df.to_csv(
            os.path.join(sample_pair_dir, "gene_calls_focal.tsv"), sep="\t", index=False)

        # Accumulate for summary heatmaps
        for _, row in focal_concordance_df.iterrows():
            all_focal_records.append({
                "gene": row["gene"], "sample_id": pair_id,
                "epic_call": row["epic_focal"], "cfrrbs_call": row["cfrrbs_focal"],
                "concordance": row["concordant"],
            })

        report_rows.append({
            "epic_id": epic_id,
            "cfrrbs_id": cfrrbs_id,
            "bins_pearson": corr_bins["pearson"],
            "bins_pearson_p": corr_bins["pearson_p"],
            "bins_spearman": corr_bins["spearman"],
            "bins_spearman_p": corr_bins["spearman_p"],
            "bins_n": corr_bins["n_bins"],
            "segments_pearson": corr_segments["pearson"],
            "segments_pearson_p": corr_segments["pearson_p"],
            "segments_spearman": corr_segments["spearman"],
            "segments_spearman_p": corr_segments["spearman_p"],
            "segments_n": corr_segments["n_segments"],
        })

        # Regenerate cfRRBS genome_wide.png so focal gene labels are coloured
        _regenerate_cfrrbs_genome_plot(temp_outid, wd)

        # Generate stacked pair plots (after EPIC genomeplot is created in phase 2)
        cfrrbs_subdirs = [f for f in os.listdir(cfrrbs_dir) if f.endswith(".plots")]
        if cfrrbs_subdirs:
            cfrrbs_plot_path = os.path.join(cfrrbs_dir, cfrrbs_subdirs[0], "genome_wide.png")
            epic_plot_path = os.path.join(epic_dir, "CNV_genomeplot.png")
            
            # If both EPIC and cfRRBS genome plots exist, create a stacked genomewide plot
            stacked_path = os.path.join(sample_pair_dir, "CNV_genomewide_stacked.png")
            if os.path.exists(epic_plot_path) and os.path.exists(cfrrbs_plot_path):
                try:
                    _stack_pair_plots(
                        epic_plot_path,
                        cfrrbs_plot_path,
                        stacked_path,
                        title=f"EPIC vs cfRRBS - {pair_id}",
                    )
                except Exception as e:
                    logging.warning(f"Failed to create stacked plot for {pair_id}: {e}")

    # ===================================================================
    # Phase 4: Final reports
    # ===================================================================
    logging.info("Phase 4: Writing final reports")

    concordance_dir = os.path.join(summary_dir, "concordance")
    os.makedirs(concordance_dir, exist_ok=True)

    if report_rows:
        try:
            _plot_correlation_summary(report_rows, summary_dir)
            logging.info("Correlation summary written")
        except Exception as e:
            logging.warning(f"Failed to create correlation summary: {e}")

    if all_focal_records:
        try:
            _plot_concordance_heatmap(all_focal_records, "focal", concordance_dir)
            pd.DataFrame(all_focal_records).to_csv(
                os.path.join(concordance_dir, "concordance_focal.tsv"), sep="\t", index=False)
            logging.info("Focal concordance heatmap and TSV written")
        except Exception as e:
            logging.warning(f"Failed to create focal concordance summary: {e}")

    # Gene-level concordance summaries (aggregated across samples)
    logging.info("Phase 4b: Generating gene-level concordance summaries")

    expected_alterations = {}
    if getattr(args, "regions", None):
        expected_alterations = _load_expected_alterations_from_regions(args.regions)

    try:
        gene_stats_list = []

        if all_focal_records:
            focal_stats = _aggregate_gene_concordance_stats(all_focal_records, "focal", expected_alterations)
            if not focal_stats.empty:
                gene_stats_list.append(focal_stats)
                _plot_gene_concordance_table(focal_stats, "focal", concordance_dir)
                logging.info(f"Focal gene concordance: {len(focal_stats)} genes analyzed")

        if gene_stats_list:
            _create_gene_concordance_csv(gene_stats_list, concordance_dir)
            logging.info("Gene-level concordance summary CSV created")
    except Exception as e:
        logging.warning(f"Failed to create gene-level concordance summaries: {e}")

    # Regenerate stacked plots now that both cfRRBS and EPIC plots have been updated
    for metadata in pair_metadata:
        try:
            cfrrbs_dir_m = metadata["cfrrbs_dir"]
            epic_dir_m   = metadata["epic_dir"]
            pair_id_m    = metadata["pair_id"]
            sample_pair_dir_m = metadata["sample_pair_dir"]
            cfrrbs_subdirs = [f for f in os.listdir(cfrrbs_dir_m) if f.endswith(".plots")]
            if not cfrrbs_subdirs:
                continue
            cfrrbs_plot = os.path.join(cfrrbs_dir_m, cfrrbs_subdirs[0], "genome_wide.png")
            epic_plot   = os.path.join(epic_dir_m, "CNV_genomeplot.png")
            stacked_path = os.path.join(sample_pair_dir_m, "CNV_genomewide_stacked.png")
            if os.path.exists(epic_plot) and os.path.exists(cfrrbs_plot):
                _stack_pair_plots(epic_plot, cfrrbs_plot, stacked_path,
                                  title=f"EPIC vs cfRRBS - {pair_id_m}")
        except Exception as e:
            logging.warning(f"Failed to regenerate stacked plot for {metadata.get('pair_id', '?')}: {e}")

    try:
        _collect_genome_wide_stacked_plots(out_dir, pair_metadata)
        logging.info("Genome-wide CNV plots collected")
    except Exception as e:
        logging.warning(f"Failed to collect genome-wide CNV plots: {e}")

    logging.info(f"EPIC vs cfRRBS pipeline completed successfully ({len(report_rows)} pairs)")


