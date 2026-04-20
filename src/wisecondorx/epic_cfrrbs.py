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
    gene_call_method: str = None,
    gene_call_thr_gain: float = None,
    gene_call_thr_loss: float = None,
) -> None:
    """Run WisecondorX predict with conumee plotting."""
    cmd = [
        "WisecondorX",
        "predict",
        npz_path,
        reference,
        outid,
        "--bed",
        "--conumee",
    ]
    if blacklist:
        cmd.extend(["--blacklist", blacklist])
    if regions:
        cmd.extend(["--regions", regions])
    if gene_call_method:
        cmd.extend(["--gene-call-method", gene_call_method])
    if gene_call_thr_gain is not None:
        cmd.extend(["--gene-call-thr-gain", str(gene_call_thr_gain)])
    if gene_call_thr_loss is not None:
        cmd.extend(["--gene-call-thr-loss", str(gene_call_thr_loss)])
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


def _compute_cnv_concordance(cf_data: pd.DataFrame, epic_data: pd.DataFrame, level_name: str = "regions") -> Dict[str, float]:
    """Compute concordance metrics between cfRRBS and EPIC CNV calls.
    
    Args:
        cf_data: cfRRBS data with 'chr', 'start', 'end', 'ratio'
        epic_data: EPIC data with same columns
        level_name: name of the analysis level (e.g., "segments", "detail regions")
    
    Returns:
        Dictionary with concordance metrics
    """
    # Quick exit if inputs empty or missing
    if cf_data is None or epic_data is None or cf_data.empty or epic_data.empty:
        return {
            f"{level_name}_total": 0,
            "agreement": float("nan"),
            "concordance_gain": float("nan"),
            "concordance_deletion": float("nan"),
            "positive_agreement": float("nan"),
            "negative_agreement": float("nan"),
        }

    # Work on copies
    cf_df = cf_data.copy()
    epic_df = epic_data.copy()

    # Ensure ratio columns exist (fallback to first numeric column)
    if "ratio" not in cf_df.columns:
        numeric_cols = [c for c in cf_df.select_dtypes(include=["number"]).columns if c not in ["start", "end"]]
        if numeric_cols:
            cf_df["ratio"] = cf_df[numeric_cols[0]]
    if "ratio" not in epic_df.columns:
        numeric_cols = [c for c in epic_df.select_dtypes(include=["number"]).columns if c not in ["start", "end"]]
        if numeric_cols:
            epic_df["ratio"] = epic_df[numeric_cols[0]]

    # Classify CNV calls
    cf_df["cnv_call"] = cf_df["ratio"].apply(lambda r: _classify_cnv(r))
    epic_df["cnv_call"] = epic_df["ratio"].apply(lambda r: _classify_cnv(r))

    paired = []

    # Decide pairing strategy: genomic-overlap if epic has coords, otherwise try name-based matching
    epic_has_coords = all(c in epic_df.columns for c in ["chr", "start", "end"]) and not epic_df["start"].isnull().all()

    if epic_has_coords:
        for _, cf_row in cf_df.iterrows():
            cf_chr = cf_row.get("chr")
            cf_start = cf_row.get("start")
            cf_end = cf_row.get("end")
            cf_call = cf_row.get("cnv_call")

            overlaps = epic_df[
                (epic_df.get("chr") == cf_chr) & (epic_df.get("start") < cf_end) & (epic_df.get("end") > cf_start)
            ]

            if not overlaps.empty:
                epic_call_mode = overlaps["cnv_call"].mode()
                epic_call = epic_call_mode.iloc[0] if len(epic_call_mode) > 0 else "unknown"
                epic_ratio_mean = float(overlaps["ratio"].dropna().mean()) if "ratio" in overlaps else float("nan")
                paired.append({
                    "cf_call": cf_call,
                    "epic_call": epic_call,
                    "cf_ratio": cf_row.get("ratio", float("nan")),
                    "epic_ratio": epic_ratio_mean,
                })
    else:
        # Try name-based pairing if both sides provide a 'name' column
        if "name" in cf_df.columns and "name" in epic_df.columns:
            for _, cf_row in cf_df.iterrows():
                cf_name = str(cf_row.get("name", "")).lower()
                cf_call = cf_row.get("cnv_call")
                matches = epic_df[epic_df["name"].astype(str).str.lower() == cf_name]
                if not matches.empty:
                    epic_call_mode = matches["cnv_call"].mode()
                    epic_call = epic_call_mode.iloc[0] if len(epic_call_mode) > 0 else "unknown"
                    epic_ratio_mean = float(matches["ratio"].dropna().mean()) if "ratio" in matches else float("nan")
                    paired.append({
                        "cf_call": cf_call,
                        "epic_call": epic_call,
                        "cf_ratio": cf_row.get("ratio", float("nan")),
                        "epic_ratio": epic_ratio_mean,
                    })
        else:
            # No viable pairing info
            return {
                f"{level_name}_total": 0,
                "agreement": float("nan"),
                "concordance_gain": float("nan"),
                "concordance_deletion": float("nan"),
                "positive_agreement": float("nan"),
                "negative_agreement": float("nan"),
            }

    if not paired:
        return {
            f"{level_name}_total": len(cf_df),
            "agreement": float("nan"),
            "concordance_gain": float("nan"),
            "concordance_deletion": float("nan"),
            "positive_agreement": float("nan"),
            "negative_agreement": float("nan"),
        }

    paired_df = pd.DataFrame(paired)

    agreement = (paired_df["cf_call"] == paired_df["epic_call"]).sum() / len(paired_df)

    gains_cf = paired_df["cf_call"] == "gain"
    gains_epic = paired_df["epic_call"] == "gain"
    concordance_gain = (gains_cf & gains_epic).sum() / gains_cf.sum() if gains_cf.sum() > 0 else float("nan")

    dels_cf = paired_df["cf_call"] == "deletion"
    dels_epic = paired_df["epic_call"] == "deletion"
    concordance_deletion = (dels_cf & dels_epic).sum() / dels_cf.sum() if dels_cf.sum() > 0 else float("nan")

    positives_cf = paired_df["cf_call"] != "neutral"
    positives_epic = paired_df["epic_call"] != "neutral"
    positive_agreement = (positives_cf & positives_epic).sum() / positives_cf.sum() if positives_cf.sum() > 0 else float("nan")

    neutrals_cf = paired_df["cf_call"] == "neutral"
    neutrals_epic = paired_df["epic_call"] == "neutral"
    negative_agreement = (neutrals_cf & neutrals_epic).sum() / neutrals_cf.sum() if neutrals_cf.sum() > 0 else float("nan")

    return {
        f"{level_name}_total": len(paired_df),
        "agreement": agreement,
        "concordance_gain": concordance_gain,
        "concordance_deletion": concordance_deletion,
        "positive_agreement": positive_agreement,
        "negative_agreement": negative_agreement,
    }


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


def _call_cfrrbs_gene_events_segment_wise(
    regions_df: pd.DataFrame,
    segments_df: pd.DataFrame = None,
    hard_thresh: float = 0.3,
) -> pd.DataFrame:
    """
    Segment-wise gene calling: a gene is called as gained/deleted if its region
    falls within a segment that is gained/deleted (using hard threshold).
    
    Args:
        regions_df: Detail regions with chr, start, end, name (or gene)
        segments_df: Segments with chr, start, end, ratio
        hard_thresh: Hard threshold for gain/deletion (absolute log2 ratio)
    
    Returns:
        DataFrame with columns: gene, chr, start, end, call (gain/deletion/neutral)
    """
    if regions_df is None or regions_df.empty:
        return pd.DataFrame(columns=["gene", "chr", "start", "end", "call"])
    
    regions = regions_df.copy()
    # Standardize gene column (may be 'name' or 'gene')
    if "gene" not in regions.columns and "name" in regions.columns:
        regions["gene"] = regions["name"]
    if "gene" not in regions.columns:
        regions["gene"] = "unknown"
    
    # Standardize required columns
    for col in ["chr", "start", "end"]:
        if col not in regions.columns:
            raise ValueError(f"Missing required column '{col}' in regions DataFrame")
    
    # Convert start/end to numeric in regions
    regions["start"] = pd.to_numeric(regions["start"], errors="coerce")
    regions["end"] = pd.to_numeric(regions["end"], errors="coerce")
    
    # NORMALIZE chromosomes in regions to match segments
    regions["chr"] = regions["chr"].astype(str).apply(_normalize_chr)
    
    if segments_df is None or segments_df.empty:
        # No segments -> all neutral
        regions["call"] = "neutral"
        return regions[["gene", "chr", "start", "end", "call"]]
    
    segments = segments_df.copy()
    # Ensure all segment columns are present and numeric
    for col in ["chr", "start", "end", "ratio"]:
        if col not in segments.columns:
            raise ValueError(f"Missing required column '{col}' in segments DataFrame")
        if col in ["start", "end", "ratio"]:
            segments[col] = pd.to_numeric(segments[col], errors="coerce")
    # Normalize segment chromosomes to match regions
    segments["chr"] = segments["chr"].astype(str).apply(_normalize_chr)
    
    rows = []
    
    # DEBUG: check if there are any aberrant segments at all
    aberrant_segments = segments[segments["ratio"].abs() >= hard_thresh]
    ratio_min = segments['ratio'].min()
    ratio_max = segments['ratio'].max()
    ratio_mean = segments['ratio'].mean()
    logging.info(f"Segment-wise gene calling DEBUG: {len(segments)} total segments, {len(aberrant_segments)} ABERRANT (|ratio|>={hard_thresh}), ratio range=[{ratio_min:.6f}, {ratio_max:.6f}], mean={ratio_mean:.6f}, chroms in segments={sorted(segments['chr'].unique().tolist())}, chroms in regions={sorted(regions['chr'].unique().tolist())}")
    if len(aberrant_segments) == 0:
        logging.warning(f"NO aberrant segments found with threshold {hard_thresh}! All genes will be neutral. Consider lowering hard_thresh (e.g., 0.1, 0.05) or using conumee method instead.")
    
    for idx, region in regions.iterrows():
        gene = str(region.get("gene", "unknown")).strip()
        chr_r = str(region.get("chr")).strip()
        start_r = pd.to_numeric(region.get("start"), errors="coerce")
        end_r = pd.to_numeric(region.get("end"), errors="coerce")
        
        # Skip if coordinates invalid
        if pd.isna(start_r) or pd.isna(end_r):
            rows.append({
                "gene": gene,
                "chr": chr_r,
                "start": start_r,
                "end": end_r,
                "call": "neutral",
            })
            continue
        
        # Find overlapping segments
        seg_chr = segments["chr"].astype(str)
        overlaps = segments[
            (seg_chr == chr_r) & 
            (segments["start"] < end_r) & 
            (segments["end"] > start_r)
        ]
        
        if overlaps.empty:
            call = "neutral"
        else:
            # Check if any overlapping segment is aberrant
            aberrant = overlaps[overlaps["ratio"].abs() >= hard_thresh]
            if not aberrant.empty:
                # Determine if gain or deletion based on sign
                gains = aberrant[aberrant["ratio"] > 0]
                losses = aberrant[aberrant["ratio"] < 0]
                
                if not gains.empty and losses.empty:
                    call = "gain"
                elif not losses.empty and gains.empty:
                    call = "deletion"
                else:
                    # Mixed or ambiguous
                    call = "neutral"
                # Log genes that are called as gain/deletion
                if idx < 3 or call != "neutral":  # Log first 3 genes + any non-neutral calls
                    logging.debug(f"  Gene {gene} ({chr_r}:{int(start_r)}-{int(end_r)}): {len(overlaps)} overlapping segments, {len(aberrant)} aberrant (>={hard_thresh}), call={call}, segment ratios={overlaps['ratio'].tolist()}")
            else:
                # All overlapping segments are neutral
                call = "neutral"
                if idx < 3:  # Log first 3 genes
                    logging.debug(f"  Gene {gene} ({chr_r}:{int(start_r)}-{int(end_r)}): {len(overlaps)} overlapping segments but NONE aberrant (>={hard_thresh}), segment ratios={overlaps['ratio'].tolist()}")
        
        rows.append({
            "gene": gene,
            "chr": chr_r,
            "start": start_r,
            "end": end_r,
            "call": call,
        })
    
    return pd.DataFrame(rows)


def _plot_aberrant_summary_scatter(all_aberrant_pairs: List[pd.DataFrame], outdir: str) -> None:
    """
    Gather all aberrant segment pairs across all samples and plot one combined scatter.
    
    Args:
        all_aberrant_pairs: List of DataFrames with aberrant segment pairs from each sample
        outdir: Output directory for the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not available; skipping aberrant summary scatter")
        return
    
    # Concatenate all aberrant pairs
    if not all_aberrant_pairs or all(df.empty for df in all_aberrant_pairs):
        logging.info("No aberrant segment pairs across all samples for summary scatter")
        return
    
    valid_dfs = [df for df in all_aberrant_pairs if isinstance(df, pd.DataFrame) and not df.empty]
    if not valid_dfs:
        return
    
    combined = pd.concat(valid_dfs, ignore_index=True)
    combined = combined.dropna(subset=["ratio_cf", "ratio_epic"])
    
    if combined.empty or len(combined) < 3:
        logging.warning(f"Insufficient aberrant pairs for summary scatter: {len(combined)} pairs")
        return
    
    # Compute correlation
    from scipy import stats
    try:
        pearson_r, pearson_p = stats.pearsonr(combined["ratio_cf"], combined["ratio_epic"])
        spearman_r, spearman_p = stats.spearmanr(combined["ratio_cf"], combined["ratio_epic"])
    except Exception:
        pearson_r = pearson_p = spearman_r = spearman_p = float("nan")
    
    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(combined["ratio_cf"], combined["ratio_epic"], s=8, alpha=0.6)
    
    min_val = min(combined["ratio_cf"].min(), combined["ratio_epic"].min())
    max_val = max(combined["ratio_cf"].max(), combined["ratio_epic"].max())
    plt.plot([min_val, max_val], [min_val, max_val], color="black", linewidth=1, linestyle="--")
    
    try:
        fit = np.polyfit(combined["ratio_cf"], combined["ratio_epic"], 1)
        fit_x = np.array([min_val, max_val])
        fit_y = fit[0] * fit_x + fit[1]
        plt.plot(fit_x, fit_y, color="red", linewidth=1.5)
    except Exception:
        pass
    
    plt.axhline(0, color="grey", linewidth=0.5)
    plt.axvline(0, color="grey", linewidth=0.5)
    plt.xlabel("cfRRBS log2 ratio")
    plt.ylabel("EPIC log2 ratio")
    title = (
        f"Aberrant Segments (All Samples)\n"
        f"Pearson r={pearson_r:.3f} (p={pearson_p:.2e}), "
        f"Spearman r={spearman_r:.3f} (p={spearman_p:.2e})\n"
        f"n={len(combined)} segment pairs"
    )
    plt.title(title)
    plt.tight_layout()
    
    out_path = os.path.join(outdir, "corr_segments_aberrant_scatter_summary.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logging.info(f"Aberrant summary scatter saved to {out_path}")


def _call_cfrrbs_gene_events(
    cf_regions_df: pd.DataFrame,
    aberrations_df: pd.DataFrame,
    cf_segments_df: pd.DataFrame = None,
    cf_bins_df: pd.DataFrame = None,
    conf: float = 0.99,
    fallback_ratio_threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Focal-like gene calling for cfRRBS inspired by conumee CNV.focal:
    - infer copy-number states from segments,
    - derive state-specific dynamic thresholds (mean ± z*sd),
    - call each detail gene relative to thresholds of its overlapping state,
    - use aberration overlap as support/fallback.
    """
    if cf_regions_df is None or cf_regions_df.empty:
        return pd.DataFrame(
            columns=[
                "gene", "chr", "start", "end", "ratio", "zscore",
                "state", "state_low", "state_high", "call", "call_source",
                "aberration_types", "aberration_count",
            ]
        )

    regions = cf_regions_df.copy()
    if "chr" in regions.columns:
        regions["chr"] = regions["chr"].astype(str).apply(_normalize_chr)
    for col in ["start", "end", "ratio", "zscore"]:
        if col in regions.columns:
            regions[col] = pd.to_numeric(regions[col], errors="coerce")

    segments = cf_segments_df.copy() if cf_segments_df is not None and not cf_segments_df.empty else pd.DataFrame()
    if not segments.empty:
        if "chr" in segments.columns:
            segments["chr"] = segments["chr"].astype(str).apply(_normalize_chr)
        for col in ["start", "end", "ratio"]:
            if col in segments.columns:
                segments[col] = pd.to_numeric(segments[col], errors="coerce")
        segments = segments.dropna(subset=["chr", "start", "end", "ratio"])

    bins = cf_bins_df.copy() if cf_bins_df is not None and not cf_bins_df.empty else pd.DataFrame()
    if not bins.empty:
        if "chr" in bins.columns:
            bins["chr"] = bins["chr"].astype(str).apply(_normalize_chr)
        for col in ["start", "end", "ratio"]:
            if col in bins.columns:
                bins[col] = pd.to_numeric(bins[col], errors="coerce")
        bins = bins.dropna(subset=["chr", "start", "end", "ratio"])

    # 1) State inference from segments (conumee-inspired)
    state_thresholds = {1: (-0.6, -0.05), 2: (-0.2, 0.2), 3: (0.05, 0.6)}
    seg_state_df = pd.DataFrame(columns=["chr", "start", "end", "ratio", "state"])

    if not segments.empty:
        seg_central = segments[(segments["ratio"] > -0.8) & (segments["ratio"] < 0.8)].copy()
        if len(seg_central) >= 3:
            q = np.quantile(seg_central["ratio"].values, [0.2, 0.5, 0.8])
            centers = np.array(sorted(q.tolist()))
        else:
            centers = np.array([-0.2, 0.0, 0.2])

        def _state_from_ratio(r):
            if pd.isna(r):
                return 2
            if r <= -0.8:
                return 1
            if r >= 0.8:
                return 3
            idx = int(np.argmin(np.abs(centers - r)))
            return idx + 1

        seg_state_df = segments[["chr", "start", "end", "ratio"]].copy()
        seg_state_df["state"] = seg_state_df["ratio"].apply(_state_from_ratio)

        # 2) Build state distributions (prefer bins overlapping each state)
        state_values = {1: [], 2: [], 3: []}
        if not bins.empty:
            for _, srow in seg_state_df.iterrows():
                overlaps = bins[
                    (bins["chr"] == srow["chr"])
                    & (bins["start"] < srow["end"])
                    & (bins["end"] > srow["start"])
                ]
                if not overlaps.empty:
                    state_values[int(srow["state"])].extend(overlaps["ratio"].dropna().tolist())

        # Fallback to segment ratios if bins sparse
        for s in [1, 2, 3]:
            if len(state_values[s]) < 10:
                state_values[s].extend(seg_state_df[seg_state_df["state"] == s]["ratio"].dropna().tolist())

        zcrit = float(NormalDist().inv_cdf(1 - (1 - conf) / 2)) if conf < 1 else 2.576
        global_vals = np.array([v for s in [1, 2, 3] for v in state_values[s]], dtype=float)
        gmean = float(np.nanmean(global_vals)) if len(global_vals) else 0.0
        gstd = float(np.nanstd(global_vals, ddof=1)) if len(global_vals) > 2 else 0.1
        gstd = max(gstd, 0.05)

        for s in [1, 2, 3]:
            vals = np.array(state_values[s], dtype=float)
            if len(vals) >= 3:
                mean = float(np.nanmean(vals))
                std = float(np.nanstd(vals, ddof=1))
                std = max(std, 0.03)
                state_thresholds[s] = (mean - zcrit * std, mean + zcrit * std)
            else:
                state_thresholds[s] = (gmean - zcrit * gstd, gmean + zcrit * gstd)

    rows = []
    for _, row in regions.iterrows():
        gene = str(row.get("name", "")).strip()
        chrom = row.get("chr", None)
        start = row.get("start", None)
        end = row.get("end", None)
        ratio = row.get("ratio", np.nan)
        zscore = row.get("zscore", np.nan)

        # 3) Determine gene state from overlapping segment(s)
        gene_state = 2
        if not seg_state_df.empty and pd.notna(chrom) and pd.notna(start) and pd.notna(end):
            seg_ov = seg_state_df[
                (seg_state_df["chr"] == chrom)
                & (seg_state_df["start"] < end)
                & (seg_state_df["end"] > start)
            ]
            if not seg_ov.empty:
                gene_state = int(round(float(seg_ov["state"].mean())))
                gene_state = min(3, max(1, gene_state))

        low, high = state_thresholds.get(gene_state, (-fallback_ratio_threshold, fallback_ratio_threshold))

        # 4) Dynamic call using state-specific threshold
        if pd.isna(ratio):
            call = "unknown"
        elif ratio >= high:
            call = "gain"
        elif ratio <= low:
            call = "deletion"
        else:
            call = "neutral"
        call_source = "dynamic_state_threshold"

        # 5) Aberration overlap as support/fallback
        overlap_types = []
        if (
            aberrations_df is not None
            and not aberrations_df.empty
            and pd.notna(chrom)
            and pd.notna(start)
            and pd.notna(end)
            and all(c in aberrations_df.columns for c in ["chr", "start", "end"])
        ):
            overlaps = aberrations_df[
                (aberrations_df["chr"] == chrom)
                & (aberrations_df["start"] < end)
                & (aberrations_df["end"] > start)
            ]
            if not overlaps.empty and "type" in overlaps.columns:
                overlap_types = sorted(set(overlaps["type"].dropna().astype(str).tolist()))
                has_gain = "gain" in overlap_types
                has_loss = "loss" in overlap_types
                if call == "neutral":
                    if has_gain and not has_loss and pd.notna(ratio) and ratio > 0.2:
                        call = "gain"
                        call_source = "aberration_supported_fallback"
                    elif has_loss and not has_gain and pd.notna(ratio) and ratio < -0.2:
                        call = "deletion"
                        call_source = "aberration_supported_fallback"
                elif call == "gain" and has_gain:
                    call_source = "dynamic+aberration_supported"
                elif call == "deletion" and has_loss:
                    call_source = "dynamic+aberration_supported"

        rows.append(
            {
                "gene": gene,
                "chr": chrom,
                "start": start,
                "end": end,
                "ratio": ratio,
                "zscore": zscore,
                "state": gene_state,
                "state_low": low,
                "state_high": high,
                "call": call,
                "call_source": call_source,
                "aberration_types": ",".join(overlap_types) if overlap_types else "",
                "aberration_count": len(overlap_types),
            }
        )

    return pd.DataFrame(rows)


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


def _compute_segment_correlations_aberrant(
    cf_segments: pd.DataFrame, epic_segments: pd.DataFrame, hard_thresh: float = 0.3
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Compute correlations only on segments that are aberrant (gain or deletion)

    Uses a hard threshold (absolute log2 ratio) to define aberrant segments on
    either technology. Returns a dict with correlation stats and the paired
    DataFrame that was used for the calculation (after filtering).
    """
    # reuse pairing routine
    pairs = _pair_segments_by_overlap(cf_segments, epic_segments)

    if not isinstance(pairs, pd.DataFrame) or pairs.empty:
        return ({
            "pearson": float("nan"),
            "pearson_p": float("nan"),
            "spearman": float("nan"),
            "spearman_p": float("nan"),
            "n_segments": 0,
        }, pd.DataFrame())

    # keep only rows where either cf or epic segment is aberrant by hard threshold
    thr = float(hard_thresh)
    filt = (
        (pairs["ratio_cf"].abs() >= thr) | (pairs["ratio_epic"].abs() >= thr)
    )
    aberrant_pairs = pairs[filt].dropna(subset=["ratio_cf", "ratio_epic"]).copy()

    if aberrant_pairs.empty or len(aberrant_pairs) < 3:
        return ({
            "pearson": float("nan"),
            "pearson_p": float("nan"),
            "spearman": float("nan"),
            "spearman_p": float("nan"),
            "n_segments": len(aberrant_pairs),
        }, aberrant_pairs)

    pearson = aberrant_pairs[["ratio_cf", "ratio_epic"]].corr(method="pearson").iloc[0, 1]
    spearman = aberrant_pairs[["ratio_cf", "ratio_epic"]].corr(method="spearman").iloc[0, 1]

    try:
        from scipy import stats

        pearson_p = stats.pearsonr(aberrant_pairs["ratio_cf"], aberrant_pairs["ratio_epic"]).pvalue
        spearman_p = stats.spearmanr(aberrant_pairs["ratio_cf"], aberrant_pairs["ratio_epic"]).pvalue
    except Exception:
        pearson_p = float("nan")
        spearman_p = float("nan")

    return ({
        "pearson": pearson,
        "pearson_p": pearson_p,
        "spearman": spearman,
        "spearman_p": spearman_p,
        "n_segments": len(aberrant_pairs),
    }, aberrant_pairs)


def _compute_segment_correlations_with_aberrant(cf_segments: pd.DataFrame, epic_segments: pd.DataFrame, hard_thresh: float = 0.3) -> Dict[str, object]:
    """Compute both regular segment correlations and aberrant-only correlations.

    Returns a dictionary suitable for inclusion in per-sample report rows with keys:
        'segments_pearson', 'segments_pearson_p', 'segments_spearman', 'segments_spearman_p', 'segments_n',
        'segments_aberrant_pearson', 'segments_aberrant_pearson_p', 'segments_aberrant_spearman', 'segments_aberrant_spearman_p', 'segments_aberrant_n',
        'segments_aberrant_pairs' (DataFrame)
    """
    base = _compute_segment_correlations(cf_segments, epic_segments)
    aberrant_stats, aberrant_pairs = _compute_segment_correlations_aberrant(cf_segments, epic_segments, hard_thresh=hard_thresh)

    out = {
        'segments_pearson': base.get('pearson'),
        'segments_pearson_p': base.get('pearson_p'),
        'segments_spearman': base.get('spearman'),
        'segments_spearman_p': base.get('spearman_p'),
        'segments_n': base.get('n_segments'),

        'segments_aberrant_pearson': aberrant_stats.get('pearson'),
        'segments_aberrant_pearson_p': aberrant_stats.get('pearson_p'),
        'segments_aberrant_spearman': aberrant_stats.get('spearman'),
        'segments_aberrant_spearman_p': aberrant_stats.get('spearman_p'),
        'segments_aberrant_n': aberrant_stats.get('n_segments'),
        'segments_aberrant_pairs': aberrant_pairs,
    }
    return out


def _plot_scatter_aberrant(pairs_df: pd.DataFrame, out_png: str, title_prefix: str = "Aberrant ", corr: Dict[str, float] = None) -> None:
    """Wrapper to plot scatter specifically for aberrant-segment pairings."""
    if isinstance(pairs_df, pd.DataFrame) and not pairs_df.empty:
        _plot_scatter(pairs_df.rename(columns={'ratio_cf': 'ratio_cf', 'ratio_epic': 'ratio_epic'}), out_png, title_prefix=title_prefix, corr=corr)
    else:
        # still create an informative empty scatter
        _plot_scatter(pd.DataFrame(columns=['ratio_cf', 'ratio_epic']), out_png, title_prefix=title_prefix, corr=corr)


def _plot_bin_ratio_distribution(cf_bins: pd.DataFrame, epic_bins: pd.DataFrame, out_png: str, pair_id: str = "") -> None:
    """Plot distribution (histogram + KDE) of bin log2 ratios for cfRRBS and EPIC

    Both distributions are plotted on the same axes for easy comparison.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            logging.warning("matplotlib/seaborn not available; skipping bin distribution plot")
            return

    plt.figure(figsize=(8, 5))

    cf_vals = []
    ep_vals = []
    if cf_bins is not None and not cf_bins.empty and "ratio" in cf_bins.columns:
        cf_vals = pd.to_numeric(cf_bins["ratio"].dropna(), errors="coerce").values
    if epic_bins is not None and not epic_bins.empty and "ratio" in epic_bins.columns:
        ep_vals = pd.to_numeric(epic_bins["ratio"].dropna(), errors="coerce").values

    if (len(cf_vals) == 0) and (len(ep_vals) == 0):
        plt.text(0.5, 0.5, 'No bin ratio data', ha='center', va='center', transform=plt.gca().transAxes)
    else:
        # use seaborn if available for KDE; fall back to hist
        try:
            if len(cf_vals) > 0:
                sns.kdeplot(cf_vals, label='cfRRBS', color='#2b83ba', fill=True, alpha=0.3)
                sns.histplot(cf_vals, bins=60, color='#2b83ba', alpha=0.15)
            if len(ep_vals) > 0:
                sns.kdeplot(ep_vals, label='EPIC', color='#d7191c', fill=True, alpha=0.3)
                sns.histplot(ep_vals, bins=60, color='#d7191c', alpha=0.12)
        except Exception:
            if len(cf_vals) > 0:
                plt.hist(cf_vals, bins=60, alpha=0.4, label='cfRRBS', color='#2b83ba')
            if len(ep_vals) > 0:
                plt.hist(ep_vals, bins=60, alpha=0.4, label='EPIC', color='#d7191c')

        plt.axvline(0, color='black', linestyle='--', linewidth=1)
        plt.xlabel('log2 Ratio')
        plt.ylabel('Density')
        plt.title(f'Bin Ratio Distribution{": " + pair_id if pair_id else ""}')
        plt.legend()

        # Add statistics text boxes
        stats_text_cf = ""
        stats_text_ep = ""
        if len(cf_vals) > 0:
            cf_mean = float(np.mean(cf_vals))
            cf_sd = float(np.std(cf_vals, ddof=1)) if len(cf_vals) > 1 else 0.0
            cf_min = float(np.min(cf_vals))
            cf_max = float(np.max(cf_vals))
            stats_text_cf = f"cfRRBS (n={len(cf_vals)})\nmean={cf_mean:.3f}\nsd={cf_sd:.3f}\nmin={cf_min:.3f}, max={cf_max:.3f}"
        if len(ep_vals) > 0:
            ep_mean = float(np.mean(ep_vals))
            ep_sd = float(np.std(ep_vals, ddof=1)) if len(ep_vals) > 1 else 0.0
            ep_min = float(np.min(ep_vals))
            ep_max = float(np.max(ep_vals))
            stats_text_ep = f"EPIC (n={len(ep_vals)})\nmean={ep_mean:.3f}\nsd={ep_sd:.3f}\nmin={ep_min:.3f}, max={ep_max:.3f}"

        # Place stats on the plot (top-left)
        if stats_text_cf or stats_text_ep:
            combined_text = ""
            if stats_text_cf:
                combined_text += stats_text_cf
            if stats_text_ep:
                if combined_text:
                    combined_text += "\n\n"
                combined_text += stats_text_ep
            plt.text(0.02, 0.97, combined_text,
                    transform=plt.gca().transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    try:
        plt.savefig(out_png, dpi=200, bbox_inches='tight')
    except Exception:
        logging.error(f"Failed to save bin distribution plot to {out_png}")
    plt.close()


def _plot_segment_ratio_distribution(cf_segments: pd.DataFrame, epic_segments: pd.DataFrame, out_png: str, pair_id: str = "") -> None:
    """Plot distribution (histogram + KDE) of segment log2 ratios for cfRRBS and EPIC

    Both distributions are plotted on the same axes for easy comparison.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            logging.warning("matplotlib/seaborn not available; skipping segment distribution plot")
            return

    plt.figure(figsize=(8, 5))

    cf_vals = []
    ep_vals = []
    if cf_segments is not None and not cf_segments.empty and "ratio" in cf_segments.columns:
        cf_vals = pd.to_numeric(cf_segments["ratio"].dropna(), errors="coerce").values
    if epic_segments is not None and not epic_segments.empty and "ratio" in epic_segments.columns:
        ep_vals = pd.to_numeric(epic_segments["ratio"].dropna(), errors="coerce").values

    if (len(cf_vals) == 0) and (len(ep_vals) == 0):
        plt.text(0.5, 0.5, 'No segment ratio data', ha='center', va='center', transform=plt.gca().transAxes)
    else:
        # use seaborn if available for KDE; fall back to hist
        try:
            if len(cf_vals) > 0:
                sns.kdeplot(cf_vals, label='cfRRBS', color='#2b83ba', fill=True, alpha=0.3)
                sns.histplot(cf_vals, bins=60, color='#2b83ba', alpha=0.15)
            if len(ep_vals) > 0:
                sns.kdeplot(ep_vals, label='EPIC', color='#d7191c', fill=True, alpha=0.3)
                sns.histplot(ep_vals, bins=60, color='#d7191c', alpha=0.12)
        except Exception:
            if len(cf_vals) > 0:
                plt.hist(cf_vals, bins=60, alpha=0.4, label='cfRRBS', color='#2b83ba')
            if len(ep_vals) > 0:
                plt.hist(ep_vals, bins=60, alpha=0.4, label='EPIC', color='#d7191c')

        plt.axvline(0, color='black', linestyle='--', linewidth=1)
        plt.xlabel('log2 Ratio')
        plt.ylabel('Density')
        plt.title(f'Segment Ratio Distribution{": " + pair_id if pair_id else ""}')
        plt.legend()

        # Add statistics text boxes
        stats_text_cf = ""
        stats_text_ep = ""
        if len(cf_vals) > 0:
            cf_mean = float(np.mean(cf_vals))
            cf_sd = float(np.std(cf_vals, ddof=1)) if len(cf_vals) > 1 else 0.0
            cf_min = float(np.min(cf_vals))
            cf_max = float(np.max(cf_vals))
            stats_text_cf = f"cfRRBS (n={len(cf_vals)})\nmean={cf_mean:.3f}\nsd={cf_sd:.3f}\nmin={cf_min:.3f}, max={cf_max:.3f}"
        if len(ep_vals) > 0:
            ep_mean = float(np.mean(ep_vals))
            ep_sd = float(np.std(ep_vals, ddof=1)) if len(ep_vals) > 1 else 0.0
            ep_min = float(np.min(ep_vals))
            ep_max = float(np.max(ep_vals))
            stats_text_ep = f"EPIC (n={len(ep_vals)})\nmean={ep_mean:.3f}\nsd={ep_sd:.3f}\nmin={ep_min:.3f}, max={ep_max:.3f}"

        # Place stats on the plot (top-left)
        if stats_text_cf or stats_text_ep:
            combined_text = ""
            if stats_text_cf:
                combined_text += stats_text_cf
            if stats_text_ep:
                if combined_text:
                    combined_text += "\n\n"
                combined_text += stats_text_ep
            plt.text(0.02, 0.97, combined_text,
                    transform=plt.gca().transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    try:
        plt.savefig(out_png, dpi=200, bbox_inches='tight')
    except Exception:
        logging.error(f"Failed to save segment distribution plot to {out_png}")
    plt.close()



# hg38 chromosome lengths (autosomes)
_CHR_LENGTHS_HG38 = {
    1: 248956422, 2: 242193529, 3: 198295559, 4: 190214555, 5: 181538259,
    6: 170805979, 7: 159345973, 8: 145138636, 9: 138394717, 10: 133797422,
    11: 135086622, 12: 133275309, 13: 114364328, 14: 107043718, 15: 101991189,
    16: 90338345, 17: 83257441, 18: 80373285, 19: 58617616, 20: 64444167,
    21: 46709983, 22: 50818468,
}


def _plot_paired_segments_genome_track(
    paired_df: pd.DataFrame,
    out_png: str,
    pair_id: str = "",
) -> None:
    """Draw a genome-track figure showing cfRRBS and EPIC segments as horizontal
    bars on two parallel tracks, with shaded ribbons connecting each paired
    segment to visualise their genomic overlap and ratio concordance."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.collections import PatchCollection
    except ImportError:
        logging.warning("matplotlib not available; skipping paired-segments track plot")
        return

    if paired_df.empty:
        return

    # ---- build cumulative genome coordinates ----
    chroms = sorted(_CHR_LENGTHS_HG38.keys())
    chr_offsets: Dict[int, int] = {}
    cum = 0
    chr_mids: Dict[int, float] = {}
    for c in chroms:
        chr_offsets[c] = cum
        chr_mids[c] = cum + _CHR_LENGTHS_HG38[c] / 2
        cum += _CHR_LENGTHS_HG38[c]
    genome_len = cum

    # ---- colour helper: ±0.3 threshold ----
    _gain_thresh = 0.3
    _loss_thresh = -0.3

    def _ratio_color(r):
        if pd.isna(r):
            return "#bdc3c7"
        v = float(r)
        if v >= _gain_thresh:
            return "#27ae60"   # solid green – gain
        elif v <= _loss_thresh:
            return "#c0392b"   # solid red – deletion
        return "#bdc3c7"       # grey – neutral

    # ---- layout ----
    track_h = 0.28           # height of each segment bar track
    cf_y = 0.62              # y-center of cfRRBS track
    epic_y = 0.22            # y-center of EPIC track
    cf_top = cf_y + track_h / 2
    cf_bot = cf_y - track_h / 2
    epic_top = epic_y + track_h / 2
    epic_bot = epic_y - track_h / 2

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.set_xlim(0, genome_len)
    ax.set_ylim(0, 1)
    ax.set_yticks([epic_y, cf_y])
    ax.set_yticklabels(["EPIC", "cfRRBS"], fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", length=0)

    # chromosome separators + labels
    for c in chroms:
        x = chr_offsets[c]
        ax.axvline(x, color="#d0d0d0", lw=0.5, zorder=0)
        ax.text(
            chr_mids[c], 0.95, str(c),
            ha="center", va="top", fontsize=7, color="#888888",
        )

    # ---- draw segments and ribbons ----
    for _, row in paired_df.iterrows():
        chrom = int(row["chr"])
        if chrom not in chr_offsets:
            continue
        off = chr_offsets[chrom]

        cf_s = off + int(row["cf_start"])
        cf_e = off + int(row["cf_end"])
        ep_s = off + int(row["epic_start"])
        ep_e = off + int(row["epic_end"])

        # cfRRBS segment bar
        cf_col = _ratio_color(row.get("ratio_cf"))
        ax.barh(cf_y, cf_e - cf_s, left=cf_s, height=track_h,
                color=cf_col, edgecolor="black", linewidth=0.4, zorder=2)

        # EPIC segment bar
        ep_col = _ratio_color(row.get("ratio_epic"))
        ax.barh(epic_y, ep_e - ep_s, left=ep_s, height=track_h,
                color=ep_col, edgecolor="black", linewidth=0.4, zorder=2)

        # Connecting lines from cf segment edges to epic segment edges
        cf_r = row.get("ratio_cf", 0)
        ep_r = row.get("ratio_epic", 0)
        cf_gain = cf_r >= _gain_thresh if pd.notna(cf_r) else False
        cf_loss = cf_r <= _loss_thresh if pd.notna(cf_r) else False
        ep_gain = ep_r >= _gain_thresh if pd.notna(ep_r) else False
        ep_loss = ep_r <= _loss_thresh if pd.notna(ep_r) else False

        if cf_gain and ep_gain:
            link_col = "#27ae60"
        elif cf_loss and ep_loss:
            link_col = "#c0392b"
        elif (cf_gain or cf_loss) != (ep_gain or ep_loss):
            link_col = "#e67e22"
        else:
            link_col = "#999999"

        # Draw two diagonal lines connecting matching segment edges
        ax.plot([cf_s, ep_s], [cf_bot, epic_top], color=link_col,
                linewidth=1.0, alpha=0.6, zorder=1)
        ax.plot([cf_e, ep_e], [cf_bot, epic_top], color=link_col,
                linewidth=1.0, alpha=0.6, zorder=1)
        # Light fill between them for context
        verts = [
            (cf_s, cf_bot), (cf_e, cf_bot),
            (ep_e, epic_top), (ep_s, epic_top),
        ]
        poly = mpatches.Polygon(verts, closed=True, facecolor=link_col,
                                alpha=0.08, edgecolor="none", zorder=0)
        ax.add_patch(poly)

    # reference lines
    ax.axhline(cf_bot, color="#aaa", lw=0.5, zorder=0)
    ax.axhline(epic_top, color="#aaa", lw=0.5, zorder=0)

    # formatting
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    title = "Paired Segment Genome Track"
    if pair_id:
        title += f": {pair_id}"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor="#c0392b", edgecolor="black", linewidth=0.5,
                       label="Deletion (ratio \u2264 \u22120.3)"),
        mpatches.Patch(facecolor="#bdc3c7", edgecolor="black", linewidth=0.5,
                       label="Neutral"),
        mpatches.Patch(facecolor="#27ae60", edgecolor="black", linewidth=0.5,
                       label="Gain (ratio \u2265 0.3)"),
        mpatches.Patch(facecolor="#e67e22", edgecolor="black", linewidth=0.5,
                       alpha=0.4, label="Discordant"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8,
              framealpha=0.8, edgecolor="#ccc")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Paired-segments genome track saved to {out_png}")


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


def _compute_venn_counts(
    cf_calls: List[Dict], epic_calls: List[Dict],
) -> Dict[str, int]:
    """Compute Venn diagram counts for a single sample's gene calls.

    Args:
        cf_calls: cfRRBS gene call dicts (must have 'name' and 'call' keys)
        epic_calls: EPIC gene call dicts (must have 'name' and 'call' keys)

    Returns:
        Dict with keys: amp_both, amp_cf_only, amp_epic_only,
                        del_both, del_cf_only, del_epic_only
    """
    def _names_by_call(calls, call_type):
        return {str(c["name"]).strip() for c in calls
                if str(c.get("call", "")).strip() == call_type and str(c.get("name", "")).strip()}

    cf_amp = _names_by_call(cf_calls, "gain")
    cf_del = _names_by_call(cf_calls, "deletion")
    epic_amp = _names_by_call(epic_calls, "gain")
    epic_del = _names_by_call(epic_calls, "deletion")

    return {
        "amp_both": len(cf_amp & epic_amp),
        "amp_cf_only": len(cf_amp - epic_amp),
        "amp_epic_only": len(epic_amp - cf_amp),
        "del_both": len(cf_del & epic_del),
        "del_cf_only": len(cf_del - epic_del),
        "del_epic_only": len(epic_del - cf_del),
    }


def _plot_venn_diagram(
    counts: Dict[str, int],
    outdir: str,
    title: str = "Gene CNV Call Reproducibility: cfRRBS vs EPIC",
    filename: str = "gene_calls_summary_venn.png",
) -> None:
    """Draw a two-panel Venn diagram (amplifications + deletions) and save as PNG.

    Args:
        counts: dict with amp_both, amp_cf_only, amp_epic_only,
                del_both, del_cf_only, del_epic_only
        outdir: output directory
        title: suptitle text
        filename: output filename
    """
    import matplotlib.pyplot as plt

    try:
        from matplotlib_venn import venn2
        use_venn2 = True
    except ImportError:
        use_venn2 = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    amp_cf_only = counts["amp_cf_only"]
    amp_epic_only = counts["amp_epic_only"]
    amp_both = counts["amp_both"]
    del_cf_only = counts["del_cf_only"]
    del_epic_only = counts["del_epic_only"]
    del_both = counts["del_both"]

    if use_venn2:
        v_amp = venn2(
            subsets=[amp_cf_only, amp_epic_only, amp_both],
            set_labels=('cfRRBS', 'EPIC'),
            ax=axes[0],
        )
        if v_amp is not None:
            for t in v_amp.set_labels:
                if t is not None:
                    t.set_fontsize(12)
                    t.set_fontweight('bold')
            for t in v_amp.subset_labels:
                if t is not None:
                    t.set_fontsize(13)
                    t.set_fontweight('bold')
        axes[0].set_title("Amplified Genes", fontsize=13, fontweight="bold")
        axes[0].set_xlim(-1.35, 1.35)
        axes[0].set_ylim(-1.1, 1.2)

        v_del = venn2(
            subsets=[del_cf_only, del_epic_only, del_both],
            set_labels=('cfRRBS', 'EPIC'),
            ax=axes[1],
        )
        if v_del is not None:
            for t in v_del.set_labels:
                if t is not None:
                    t.set_fontsize(12)
                    t.set_fontweight('bold')
            for t in v_del.subset_labels:
                if t is not None:
                    t.set_fontsize(13)
                    t.set_fontweight('bold')
        axes[1].set_title("Deleted Genes", fontsize=13, fontweight="bold")
        axes[1].set_xlim(-1.35, 1.35)
        axes[1].set_ylim(-1.1, 1.2)
    else:
        from matplotlib.patches import Circle

        def draw_venn_fallback(ax, left_only, right_only, both, left_label, right_label, panel_title):
            ax.set_xlim(-1.7, 1.7)
            ax.set_ylim(-1.1, 1.9)
            ax.set_aspect('equal')
            ax.axis('off')
            circle_left = Circle((-0.45, 0.35), 0.78, facecolor='#5DADE2', edgecolor='#2E86C1', alpha=0.35, linewidth=2)
            circle_right = Circle((0.45, 0.35), 0.78, facecolor='#F1948A', edgecolor='#CB4335', alpha=0.35, linewidth=2)
            ax.add_patch(circle_left)
            ax.add_patch(circle_right)
            ax.text(-0.95, 1.35, left_label, fontsize=12, fontweight='bold', ha='center', va='center')
            ax.text(0.95, 1.35, right_label, fontsize=12, fontweight='bold', ha='center', va='center')
            ax.text(-0.88, 0.35, f"{left_only}", fontsize=16, fontweight='bold', ha='center', va='center')
            ax.text(0.0, 0.35, f"{both}", fontsize=16, fontweight='bold', ha='center', va='center')
            ax.text(0.88, 0.35, f"{right_only}", fontsize=16, fontweight='bold', ha='center', va='center')
            ax.set_title(panel_title, fontsize=13, fontweight='bold')

        draw_venn_fallback(axes[0], amp_cf_only, amp_epic_only, amp_both,
                           'cfRRBS', 'EPIC', "Amplified Genes")
        draw_venn_fallback(axes[1], del_cf_only, del_epic_only, del_both,
                           'cfRRBS', 'EPIC', "Deleted Genes")

    fig.suptitle(title, fontsize=15, fontweight="bold")

    os.makedirs(outdir, exist_ok=True)
    repro_plot = os.path.join(outdir, filename)
    plt.savefig(repro_plot, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Venn diagram saved to {repro_plot}")


def _generate_gene_reproducibility_report(cfrrbs_calls: List[Dict], epic_calls: List[Dict], outdir: str) -> None:
    """Generate summary Venn diagram from accumulated per-sample gene calls.

    Computes per-sample Venn counts and sums them across all samples, then
    draws a single summary Venn diagram.
    """
    try:
        if not cfrrbs_calls and not epic_calls:
            logging.warning("No gene calls available for reproducibility report")
            return

        cfrrbs_df = pd.DataFrame(cfrrbs_calls)
        epic_df = pd.DataFrame(epic_calls) if epic_calls else pd.DataFrame()

        # Determine sample ids
        cf_samples = set(cfrrbs_df["sample_id"].dropna().unique()) if "sample_id" in cfrrbs_df.columns else set()
        epic_samples = set(epic_df["sample_id"].dropna().unique()) if "sample_id" in epic_df.columns else set()
        all_samples = sorted(cf_samples | epic_samples) or [None]

        totals = {"amp_both": 0, "amp_cf_only": 0, "amp_epic_only": 0,
                  "del_both": 0, "del_cf_only": 0, "del_epic_only": 0}

        for samp in all_samples:
            if samp is not None:
                cf_sub = [r for r in cfrrbs_calls if r.get("sample_id") == samp]
                ep_sub = [r for r in epic_calls if r.get("sample_id") == samp]
            else:
                cf_sub = cfrrbs_calls
                ep_sub = epic_calls
            counts = _compute_venn_counts(cf_sub, ep_sub)
            for k in totals:
                totals[k] += counts[k]

        _plot_venn_diagram(totals, outdir,
                           title="Gene CNV Call Reproducibility: cfRRBS vs EPIC (all samples summed)",
                           filename="gene_calls_summary_venn.png")

    except Exception as e:
        logging.error(f"Failed to generate gene reproducibility report: {e}", exc_info=True)


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

    # Additionally, if per-sample ratio lists are present in report_rows, plot overall distribution
    try:
        # report_rows may contain keys 'cf_segment_ratios' and 'epic_segment_ratios' as lists
        all_cf = []
        all_ep = []
        for r in report_rows:
            if isinstance(r, dict):
                if 'cf_segment_ratios' in r and r['cf_segment_ratios']:
                    all_cf.extend([float(x) for x in r['cf_segment_ratios'] if pd.notna(x)])
                if 'epic_segment_ratios' in r and r['epic_segment_ratios']:
                    all_ep.extend([float(x) for x in r['epic_segment_ratios'] if pd.notna(x)])

        if len(all_cf) or len(all_ep):
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import seaborn as sns
            except Exception:
                try:
                    import matplotlib.pyplot as plt
                except Exception:
                    logging.warning('matplotlib/seaborn not available; skipping overall segment ratio distribution')
                    return

            plt.figure(figsize=(8, 5))
            try:
                if len(all_cf) > 0:
                    sns.kdeplot(all_cf, label='cfRRBS_all', color='#2b83ba', fill=True, alpha=0.3)
                if len(all_ep) > 0:
                    sns.kdeplot(all_ep, label='EPIC_all', color='#d7191c', fill=True, alpha=0.3)
            except Exception:
                if len(all_cf) > 0:
                    plt.hist(all_cf, bins=120, alpha=0.4, label='cfRRBS_all', color='#2b83ba')
                if len(all_ep) > 0:
                    plt.hist(all_ep, bins=120, alpha=0.4, label='EPIC_all', color='#d7191c')

            plt.axvline(0, color='black', linestyle='--', linewidth=1)
            plt.xlabel('log2 Ratio')
            plt.ylabel('Density')
            plt.title('Segment Ratio Distribution Across All Samples')
            plt.legend()

            # Add statistics text boxes
            stats_text_cf = ""
            stats_text_ep = ""
            if len(all_cf) > 0:
                cf_mean = float(np.mean(all_cf))
                cf_sd = float(np.std(all_cf, ddof=1)) if len(all_cf) > 1 else 0.0
                cf_min = float(np.min(all_cf))
                cf_max = float(np.max(all_cf))
                stats_text_cf = f"cfRRBS (n={len(all_cf)})\nmean={cf_mean:.3f}\nsd={cf_sd:.3f}\nmin={cf_min:.3f}, max={cf_max:.3f}"
            if len(all_ep) > 0:
                ep_mean = float(np.mean(all_ep))
                ep_sd = float(np.std(all_ep, ddof=1)) if len(all_ep) > 1 else 0.0
                ep_min = float(np.min(all_ep))
                ep_max = float(np.max(all_ep))
                stats_text_ep = f"EPIC (n={len(all_ep)})\nmean={ep_mean:.3f}\nsd={ep_sd:.3f}\nmin={ep_min:.3f}, max={ep_max:.3f}"

            # Place stats on the plot (top-left)
            if stats_text_cf or stats_text_ep:
                combined_text = ""
                if stats_text_cf:
                    combined_text += stats_text_cf
                if stats_text_ep:
                    if combined_text:
                        combined_text += "\n\n"
                    combined_text += stats_text_ep
                plt.text(0.02, 0.97, combined_text,
                        transform=plt.gca().transAxes,
                        fontsize=9, verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            outp = os.path.join(outdir, 'distr_segment_ratio_summary.png')
            plt.tight_layout()
            plt.savefig(outp, dpi=200, bbox_inches='tight')
            plt.close()
            logging.info(f"Saved overall segment ratio distribution to {outp}")
    except Exception:
        logging.debug('No overall segment ratio distribution produced')


def _plot_gene_pair_comparison(merged_df: pd.DataFrame, pair_id: str, outdir: str, cf_calls: Set[str], epic_calls: Set[str]) -> None:
    """
    Generate per-pair gene aberration plot showing cfRRBS vs EPIC ratios with CNV status coloring.
    Colors are driven by technology-specific call TSVs passed as call sets:
    green=amplified, red=deleted, grey=neutral.
    Correlation is computed on raw log2 ratios.
    
    Args:
        merged_df: DataFrame with columns: gene, cf_ratio, epic_ratio, chr, start, end
        pair_id: Sample pair identifier
        outdir: Output directory for the plot
        cf_calls: Set of cfRRBS gene calls ("gene:call" format, e.g., "MDM4:gain")
        epic_calls: Set of EPIC gene calls ("gene:call" format)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats
        
        logging.info(f"[{pair_id}] _plot_gene_pair_comparison: merged_df shape={merged_df.shape}, columns={merged_df.columns.tolist()}")
        if merged_df.empty:
            logging.warning(f"[{pair_id}] _plot_gene_pair_comparison: merged_df is EMPTY — returning without plot")
            return
        
        # Filter to genes with both ratios
        valid_df = merged_df[(merged_df["cf_ratio"].notna()) & (merged_df["epic_ratio"].notna())].copy()
        logging.info(f"[{pair_id}] _plot_gene_pair_comparison: {len(valid_df)} genes with both ratios out of {len(merged_df)}")
        if valid_df.empty:
            logging.warning(f"[{pair_id}] _plot_gene_pair_comparison: No genes with both ratios — returning without plot")
            return
        
        # Determine CNV status independently for each bar from provided call sets
        cf_calls_norm = {str(c).strip().lower() for c in cf_calls}
        epic_calls_norm = {str(c).strip().lower() for c in epic_calls}

        def get_status_from_calls(gene_name: str, calls_norm: Set[str]) -> str:
            key_gain = f"{str(gene_name).strip().lower()}:gain"
            key_del = f"{str(gene_name).strip().lower()}:deletion"
            if key_gain in calls_norm:
                return "amp"
            if key_del in calls_norm:
                return "del"
            return "neutral"

        valid_df["cf_color_status"] = valid_df["gene"].apply(lambda g: get_status_from_calls(g, cf_calls_norm))
        valid_df["epic_color_status"] = valid_df["gene"].apply(lambda g: get_status_from_calls(g, epic_calls_norm))
        
        # Calculate correlation on log2 ratios (raw values)
        corr, pval = stats.pearsonr(valid_df['cf_ratio'], valid_df['epic_ratio'])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(len(valid_df)*0.6, 6))
        
        x = np.arange(len(valid_df))
        width = 0.35
        
        # Color map: green=amp, red=del, grey=neutral
        colors_map = {'amp': '#27ae60', 'del': '#c0392b', 'neutral': '#bdc3c7'}
        cf_colors = [colors_map[status] for status in valid_df['cf_color_status']]
        epic_colors = [colors_map[status] for status in valid_df['epic_color_status']]
        
        # Plot bars
        bars1 = ax.bar(x - width/2, valid_df['cf_ratio'], width, label='cfRRBS', 
                       color=cf_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, valid_df['epic_ratio'], width, label='EPIC',
                       color=epic_colors, alpha=0.6, hatch='//', edgecolor='black', linewidth=0.5)
        
        # Add reference lines (constitutional ploidy thresholds)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.8, zorder=1)
        ax.axhline(y=np.log2(3/2), color='green', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
        ax.axhline(y=np.log2(1/2), color='red', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
        
        # Labels and formatting
        ax.set_ylabel('log2 Ratio', fontsize=12, fontweight='bold')
        ax.set_xlabel('Genes', fontsize=12, fontweight='bold')
        ax.set_title(f'Gene Aberrations: {pair_id}\nPearson Correlation r={corr:.3f} (p={pval:.4f})', 
                    fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(valid_df['gene'], rotation=45, ha='right', fontsize=9)
        ax.set_ylim([-1.25, 1.25])
        ax.grid(axis='y', alpha=0.3, linestyle=':')
        
        # Add legend for CNV status colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#27ae60', edgecolor='black', label='Amplified', alpha=0.8),
            Patch(facecolor='#c0392b', edgecolor='black', label='Deleted', alpha=0.8),
            Patch(facecolor='#bdc3c7', edgecolor='black', label='Neutral', alpha=0.8),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, title='CNV Status')
        
        plt.tight_layout()
        output_path = os.path.join(outdir, "ratio_genes_bar.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"[{pair_id}] Gene aberrations bar plot saved to {output_path}")
        
    except Exception as e:
        logging.error(f"[{pair_id}] FAILED to create gene aberrations bar plot: {e}", exc_info=True)


def _plot_gene_call_heatmap(
    gene_data_list: List[Dict],
    gene_call_records: List[Dict],
    outdir: str,
    filename: str = "gene_call_heatmap.png",
) -> None:
    """Single heatmap showing per-sample gene CNV concordance between cfRRBS and EPIC.
    
    Colors represent:
    - Dark green (#1a9641): Both gain
    - Dark red (#d7191c): Both deletion
    - Light green (#a6d96a): cfRRBS only gain
    - Light red (#f4a582): cfRRBS only deletion
    - Lighter green (#d9ef8b): EPIC only gain
    - Lighter red (#fddbc7): EPIC only deletion
    - Gray (#cccccc): Conflicting calls
    - Very light gray (#f5f5f5): No calls in either tech
    - White with X: No measurement data
    
    Rows = samples, columns = genes (ordered by cfRRBS median ratio, high→low).
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from matplotlib.patches import Patch
        from matplotlib.colors import to_rgb
    except ImportError:
        logging.warning("matplotlib not available; skipping gene call heatmap")
        return

    if not gene_data_list:
        return

    gd_df = pd.DataFrame(gene_data_list)

    # Order genes by cfRRBS median ratio (high→low)
    cf_medians = (
        gd_df[gd_df["technology"] == "cfRRBS"]
        .groupby("gene")["ratio"]
        .median()
        .sort_values(ascending=False)
    )
    all_genes = list(cf_medians.index)
    for g in sorted(gd_df["gene"].unique()):
        if g not in all_genes:
            all_genes.append(g)

    all_samples = sorted(gd_df["sample_id"].unique())
    n_genes = len(all_genes)
    n_samples = len(all_samples)
    if n_genes == 0 or n_samples == 0:
        return

    gene_idx = {g: i for i, g in enumerate(all_genes)}
    sample_idx = {s: i for i, s in enumerate(all_samples)}

    # Build call dictionaries: maps (sample_id, gene) -> (cf_call, epic_call)
    # call values: None (no call), 'gain', 'deletion'
    call_matrix = {}
    for sample in all_samples:
        for gene in all_genes:
            call_matrix[(sample, gene)] = {"cf": None, "epic": None}

    # Populate calls from gene_call_records
    # Each record: gene, call (gain/deletion), source (cf_only/epic_only/both), sample_id
    for rec in gene_call_records:
        gene = rec.get("gene")
        sample = rec.get("sample_id")
        call = rec.get("call")  # 'gain' or 'deletion'
        source = rec.get("source")  # 'cf_only', 'epic_only', 'both'
        
        if gene not in all_genes or sample not in all_samples:
            continue
        
        if source in ("cf_only", "both") and call:
            call_matrix[(sample, gene)]["cf"] = call
        if source in ("epic_only", "both") and call:
            call_matrix[(sample, gene)]["epic"] = call

    # Determine which cells have measurement data
    has_data = {}
    for rec in gene_data_list:
        sample = rec.get("sample_id")
        gene = rec.get("gene")
        tech = rec.get("technology")
        if gene in all_genes and sample in all_samples:
            key = (sample, gene)
            if key not in has_data:
                has_data[key] = set()
            has_data[key].add(tech)

    # Color mapping based on concordance and direction
    color_map = {
        ("both", "gain"): "#1a9641",           # Dark green
        ("both", "deletion"): "#d7191c",       # Dark red
        ("cf_only", "gain"): "#a6d96a",        # Light green
        ("cf_only", "deletion"): "#f4a582",    # Light red
        ("epic_only", "gain"): "#d9ef8b",      # Lighter green
        ("epic_only", "deletion"): "#fddbc7",  # Lighter red
        ("conflict",): "#cccccc",               # Gray for conflicting
        ("neutral",): "#f5f5f5",                # Very light gray for no calls
    }

    # Build numeric matrix for display and color array for custom rendering
    mat = np.zeros((n_samples, n_genes), dtype=object)
    for si, sample in enumerate(all_samples):
        for gi, gene in enumerate(all_genes):
            cf_call = call_matrix[(sample, gene)]["cf"]
            epic_call = call_matrix[(sample, gene)]["epic"]
            
            key = (sample, gene)
            cell_data = (cf_call, epic_call, key in has_data)
            mat[si, gi] = cell_data

    # Create figure with single heatmap
    cell_w, cell_h = 0.5, 0.35
    fig_w = max(10, n_genes * cell_w + 2)
    fig_h = max(5, n_samples * cell_h + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Render heatmap manually with custom colors
    for si in range(n_samples):
        for gi in range(n_genes):
            cf_call, epic_call, has_measurement = mat[si, gi]
            
            # Determine color
            if not has_measurement:
                # No measurement data for this cell
                color = "white"
                edge_color = "#cccccc"
                edge_width = 1.5
            elif cf_call == epic_call:
                if cf_call is None:
                    # Both neutral/no call
                    color = color_map[("neutral",)]
                    edge_color = "#ddd"
                    edge_width = 0.5
                elif cf_call == "gain":
                    # Both gain
                    color = color_map[("both", "gain")]
                    edge_color = "white"
                    edge_width = 0.8
                else:  # deletion
                    # Both deletion
                    color = color_map[("both", "deletion")]
                    edge_color = "white"
                    edge_width = 0.8
            elif cf_call is not None and epic_call is None:
                # cfRRBS only
                color = color_map[("cf_only", cf_call)]
                edge_color = "white"
                edge_width = 0.8
            elif epic_call is not None and cf_call is None:
                # EPIC only
                color = color_map[("epic_only", epic_call)]
                edge_color = "white"
                edge_width = 0.8
            else:
                # Conflicting calls (one says gain, other says deletion)
                color = color_map[("conflict",)]
                edge_color = "white"
                edge_width = 0.8
            
            # Draw cell
            rect = plt.Rectangle((gi - 0.5, si - 0.5), 1, 1,
                                 facecolor=color, edgecolor=edge_color,
                                 linewidth=edge_width, zorder=1)
            ax.add_patch(rect)
            
            # Add X for missing data
            if not has_measurement:
                ax.plot([gi - 0.35, gi + 0.35], [si - 0.35, si + 0.35],
                       color="#999", linewidth=1.5, zorder=2)
                ax.plot([gi - 0.35, gi + 0.35], [si + 0.35, si - 0.35],
                       color="#999", linewidth=1.5, zorder=2)

    ax.set_xlim(-0.5, n_genes - 0.5)
    ax.set_ylim(-0.5, n_samples - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    ax.set_xticks(range(n_genes))
    ax.set_xticklabels(all_genes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_samples))
    ax.set_yticklabels(all_samples, fontsize=8)
    ax.set_xlabel("Genes", fontsize=11, fontweight="bold")
    ax.set_ylabel("Samples", fontsize=11, fontweight="bold")
    ax.set_title("Gene CNV Concordance: cfRRBS vs EPIC", fontsize=13, fontweight="bold")

    # Legend
    legend_elements = [
        Patch(facecolor="#1a9641", edgecolor="grey", label="Both gain"),
        Patch(facecolor="#d7191c", edgecolor="grey", label="Both deletion"),
        Patch(facecolor="#a6d96a", edgecolor="grey", label="cfRRBS only gain"),
        Patch(facecolor="#f4a582", edgecolor="grey", label="cfRRBS only deletion"),
        Patch(facecolor="#d9ef8b", edgecolor="grey", label="EPIC only gain"),
        Patch(facecolor="#fddbc7", edgecolor="grey", label="EPIC only deletion"),
        Patch(facecolor="#cccccc", edgecolor="grey", label="Conflicting"),
        Patch(facecolor="#f5f5f5", edgecolor="grey", label="No calls"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1),
             fontsize=8, frameon=True, fancybox=True, framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(outdir, filename)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Gene call heatmap saved to {out_path}")


def _plot_gene_boxplot_summary(gene_data_list: List[Dict], outdir: str) -> None:
    """
    Generate summary boxplot showing distribution of log2 ratios per gene across all samples.
    Side-by-side boxplots for cfRRBS and EPIC for each gene, with trendline connecting medians.
    
    Args:
        gene_data_list: List of dicts with keys: gene, technology, ratio, sample_id
        outdir: Output directory for the plot
    """
    if not gene_data_list:
        logging.warning("No gene data available for summary boxplot")
        return
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Convert to DataFrame
        df = pd.DataFrame(gene_data_list)
        
        # Order genes by cfRRBS median ratio (high to low)
        cfrrbs_medians = df[df["technology"] == "cfRRBS"].groupby("gene")["ratio"].median()
        all_genes = set(df["gene"].unique())
        # Genes with cfRRBS data sorted high→low, then remaining genes alphabetically
        genes_with_cf = cfrrbs_medians.sort_values(ascending=False).index.tolist()
        genes_without_cf = sorted(all_genes - set(genes_with_cf))
        genes = genes_with_cf + genes_without_cf
        
        if len(genes) == 0:
            logging.warning("No genes found in data for summary boxplot")
            return
        
        # Create figure with single subplot
        fig, ax = plt.subplots(figsize=(max(12, len(genes) * 1.2), 7))
        
        # Add constitutional reference lines
        ax.axhline(y=np.log2(1/2), color='red', linestyle='--', linewidth=1.5, 
                  alpha=0.6, zorder=1)
        ax.axhline(y=0, color='lightgrey', linestyle='--', linewidth=1.5, 
                  alpha=0.6, zorder=1)
        ax.axhline(y=np.log2(3/2), color='green', linestyle='--', linewidth=1.5, 
                  alpha=0.6, zorder=1)
        
        # Prepare data: side-by-side boxplots for each gene
        positions = []
        labels = []
        data_list = []
        colors_list = []
        medians_cf = []
        medians_epic = []
        gene_centers = []  # Track center position of each gene
        gene_boundaries = []  # Track boundaries between genes
        pos_counter = 0
        
        cfrrbs_data = df[df["technology"] == "cfRRBS"]
        epic_data = df[df["technology"] == "EPIC"]
        
        for gene in genes:
            gene_start = pos_counter
            
            # cfRRBS boxplot
            cf_vals = cfrrbs_data[cfrrbs_data["gene"] == gene]["ratio"].dropna().values
            if len(cf_vals) > 0:
                positions.append(pos_counter)
                labels.append(gene)  # Only gene name, no technology label
                data_list.append(cf_vals)
                colors_list.append('lightblue')
                medians_cf.append((pos_counter, np.median(cf_vals)))
                pos_counter += 1
            
            # EPIC boxplot
            epic_vals = epic_data[epic_data["gene"] == gene]["ratio"].dropna().values
            if len(epic_vals) > 0:
                positions.append(pos_counter)
                labels.append(gene)  # Only gene name, no technology label
                data_list.append(epic_vals)
                colors_list.append('lightsalmon')
                medians_epic.append((pos_counter, np.median(epic_vals)))
                pos_counter += 1
            
            # Record gene center and boundary
            gene_center = (gene_start + pos_counter - 1) / 2.0
            gene_centers.append(gene_center)
            # Boundary is at the midpoint of the spacing between genes
            gene_boundary = pos_counter + 0.25
            gene_boundaries.append(gene_boundary)
            pos_counter += 0.5  # Add spacing between gene pairs
        
        # Create boxplots
        bp = ax.boxplot(data_list, positions=positions, labels=labels, patch_artist=True,
                       widths=0.6, showfliers=True,
                       boxprops=dict(alpha=0.7, linewidth=1.5),
                       medianprops=dict(linewidth=2, color='darkblue'),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5),
                       flierprops=dict(marker='o', markersize=4, alpha=0.5))
        
        # Update x-axis to show only gene names at gene centers with rotation
        ax.set_xticks(gene_centers)
        ax.set_xticklabels(genes, rotation=45, ha='right', fontsize=9)
        
        # Add vertical lines between gene groups
        for boundary in gene_boundaries[:-1]:  # Skip last boundary
            ax.axvline(x=boundary, color='black', linestyle=':', linewidth=1, alpha=0.4, zorder=1)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_edgecolor('C0' if 'blue' in color else 'C1')
        
        # Color whiskers and caps
        for i, (whisker, cap) in enumerate(zip(bp['whiskers'], bp['caps'])):
            color = 'C0' if i % 4 < 2 else 'C1'
            whisker.set_color(color)
            cap.set_color(color)
        
        # Connect cfRRBS medians with straight line segments
        if len(medians_cf) >= 2:
            cf_pos = np.array([m[0] for m in medians_cf])
            cf_med = np.clip(np.array([m[1] for m in medians_cf]), -1.25, 1.25)
            ax.plot(cf_pos, cf_med, color='C0', linewidth=2, linestyle='-',
                   alpha=0.6, marker='o', markersize=5, zorder=2, label='cfRRBS median')
        elif len(medians_cf) == 1:
            ax.plot(medians_cf[0][0], np.clip(medians_cf[0][1], -1.25, 1.25),
                   color='C0', marker='o', markersize=5, linestyle='none', alpha=0.7, zorder=3)

        # Connect EPIC medians with straight line segments
        if len(medians_epic) >= 2:
            epic_pos = np.array([m[0] for m in medians_epic])
            epic_med = np.clip(np.array([m[1] for m in medians_epic]), -1.25, 1.25)
            ax.plot(epic_pos, epic_med, color='C1', linewidth=2, linestyle='-',
                   alpha=0.6, marker='s', markersize=5, zorder=2, label='EPIC median')
        elif len(medians_epic) == 1:
            ax.plot(medians_epic[0][0], np.clip(medians_epic[0][1], -1.25, 1.25),
                   color='C1', marker='s', markersize=5, linestyle='none', alpha=0.7, zorder=3)
        
        # Set y-axis limits
        ax.set_ylim([-1.25, 1.25])
        
        ax.set_ylabel("log2 ratio", fontsize=11)
        ax.set_xlabel("Gene / Technology", fontsize=11)
        ax.set_title("Gene-level CNV distribution across all samples (cfRRBS vs EPIC)", fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=10)
        
        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Add sample counts
        n_samples_cf = cfrrbs_data["sample_id"].nunique()
        n_samples_epic = epic_data["sample_id"].nunique()
        ax.text(0.02, 0.98, f"cfRRBS: n={n_samples_cf} | EPIC: n={n_samples_epic}", 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)
        
        # Add legend showing only technology colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', edgecolor='C0', label='cfRRBS'),
            Patch(facecolor='lightsalmon', edgecolor='C1', label='EPIC'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(outdir, "ratio_genes_summary_boxplot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Gene summary boxplot saved to {output_path}")
        
    except Exception as e:
        logging.error(f"Failed to create gene summary boxplot: {e}")
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
    
    # Get gene calling method and compute aberrant threshold from gene call thresholds
    gene_call_method = getattr(args, "gene_call_method", "conumee")
    gene_call_thr_gain = getattr(args, "gene_call_thr_gain", None)
    gene_call_thr_loss = getattr(args, "gene_call_thr_loss", None)
    
    # Derive aberrant_threshold from gain/loss thresholds
    # Use the minimum absolute value of the two thresholds, or 0.1 if neither is set
    if gene_call_thr_gain is not None or gene_call_thr_loss is not None:
        thresholds = []
        if gene_call_thr_gain is not None:
            thresholds.append(abs(float(gene_call_thr_gain)))
        if gene_call_thr_loss is not None:
            thresholds.append(abs(float(gene_call_thr_loss)))
        aberrant_threshold = min(thresholds) if thresholds else 0.1
    else:
        aberrant_threshold = 0.1
    
    logging.info(f"Gene calling method: {gene_call_method}, derived aberrant threshold: {aberrant_threshold} "
                 f"(from gain_thr={gene_call_thr_gain}, loss_thr={gene_call_thr_loss})")
    
    report_rows = []
    all_gene_data = []  # Collect gene-level data across all samples for summary boxplot
    all_gene_call_records = []  # Per-gene call records for summary overview plot
    all_aberrant_pairs = []  # Collect aberrant segment pairs for summary scatter
    total_venn_counts = {"amp_both": 0, "amp_cf_only": 0, "amp_epic_only": 0,
                         "del_both": 0, "del_cf_only": 0, "del_epic_only": 0}

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
            gene_call_method=getattr(args, "gene_call_method", None),
            gene_call_thr_gain=getattr(args, "gene_call_thr_gain", None),
            gene_call_thr_loss=getattr(args, "gene_call_thr_loss", None),
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

            # Genome-track visualisation of segment pairings
            try:
                track_png = os.path.join(sample_pair_dir, "paired_segments_track.png")
                _plot_paired_segments_genome_track(merged_segments, track_png, pair_id=pair_id)
            except Exception as e:
                logging.warning(f"[{pair_id}] Failed to create paired-segments track plot: {e}")

            # --- Aberrant-only segment correlations (for summary-level plots) ---
            ab_stats = {"pearson": float("nan"), "pearson_p": float("nan"), "spearman": float("nan"), "spearman_p": float("nan"), "n_segments": 0}
            ab_pairs = pd.DataFrame()
            try:
                ab_stats, ab_pairs = _compute_segment_correlations_aberrant(cf_segments, epic_segments, hard_thresh=aberrant_threshold)
                # Collect aberrant pairs for later summary plotting (no per-sample scatter)
                if isinstance(ab_pairs, pd.DataFrame) and not ab_pairs.empty:
                    all_aberrant_pairs.append(ab_pairs)
            except Exception as e:
                logging.debug(f"[{pair_id}] aberrant segment correlation failed: {e}")

            # --- Per-sample bin ratio distribution plot (cfRRBS vs EPIC) ---
            try:
                bin_distr_png = os.path.join(sample_pair_dir, "distr_bin_ratio.png")
                _plot_bin_ratio_distribution(cf_bins, epic_bins, bin_distr_png, pair_id=pair_id)
            except Exception as e:
                logging.warning(f"[{pair_id}] Failed to create bin ratio distribution: {e}")

            # --- Per-sample segment ratio distribution plot (cfRRBS vs EPIC) ---
            try:
                distr_png = os.path.join(sample_pair_dir, "distr_segment_ratio.png")
                _plot_segment_ratio_distribution(cf_segments, epic_segments, distr_png, pair_id=pair_id)
            except Exception as e:
                logging.warning(f"[{pair_id}] Failed to create segment ratio distribution: {e}")

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

        # Compute CNV concordance at segment level
        conc_segments = _compute_cnv_concordance(cf_segments, epic_segments, "segments")
        
        # Try to load detail regions for gene-level concordance
        conc_detail = None
        detail_cf_path = temp_outid + "_regions.bed"
        # Always produce a CF detail TSV from the cfRRBS regions file if present
        print(detail_cf_path)
        if os.path.exists(detail_cf_path):
            try:
                cf_detail_df = pd.read_csv(detail_cf_path, sep="\t")
                # If headerless, try to set expected names
                if "name" not in cf_detail_df.columns and cf_detail_df.shape[1] >= 4:
                    cf_detail_df.columns = ["chr", "start", "end", "name"] + list(
                        cf_detail_df.columns[4:]
                    )

                # Ensure ratio column exists in cf detail
                if "ratio" not in cf_detail_df.columns:
                    numeric_cols = [
                        c
                        for c in cf_detail_df.select_dtypes(include=["number"]).columns
                        if c not in ["start", "end"]
                    ]
                    if numeric_cols:
                        cf_detail_df["ratio"] = cf_detail_df[numeric_cols[0]]

                cf_detail_df = cf_detail_df.rename(
                    columns={"seqnames": "chr", "Chromosome": "chr"}, errors="ignore"
                )

                # Build cfRRBS amplified/deleted gene list
                # Choose between conumee-style (with states and ratio thresholds) or segment-wise (if gene in aberrant segment)
                logging.info(f"[{pair_id}] Gene calling: method={gene_call_method}, threshold={aberrant_threshold}, cf_detail rows={len(cf_detail_df)}, cf_segments rows={len(cf_segments)}, cf_detail cols={cf_detail_df.columns.tolist()}, cf_segments cols={cf_segments.columns.tolist()}")
                if gene_call_method == "segment-wise":
                    cf_gene_calls = _call_cfrrbs_gene_events_segment_wise(
                        regions_df=cf_detail_df,
                        segments_df=cf_segments,
                        hard_thresh=aberrant_threshold,
                    )
                    logging.info(f"[{pair_id}] Segment-wise cfRRBS: returned {len(cf_gene_calls)} genes, call distribution: gain={sum(cf_gene_calls['call']=='gain')}, deletion={sum(cf_gene_calls['call']=='deletion')}, neutral={sum(cf_gene_calls['call']=='neutral')}")
                    if not cf_gene_calls.empty:
                        cf_gene_calls = cf_gene_calls.rename(columns={"call": "call"})
                else:
                    # Default: conumee-style with ratio thresholds and aberration support
                    cf_gene_calls = _call_cfrrbs_gene_events(
                        cf_regions_df=cf_detail_df,
                        aberrations_df=cf_aberrations,
                        cf_segments_df=cf_segments,
                        cf_bins_df=cf_bins,
                        conf=0.99,
                        fallback_ratio_threshold=0.3,
                    )
                
                if not cf_gene_calls.empty:
                    amp_genes_df = cf_gene_calls[cf_gene_calls["call"] == "gain"].copy()
                    del_genes_df = cf_gene_calls[cf_gene_calls["call"] == "deletion"].copy()
                    logging.info(f"[{pair_id}] cfRRBS TSV output: gain={len(amp_genes_df)}, deletion={len(del_genes_df)}")

                    amp_genes_path = os.path.join(cfrrbs_dir, f"{cfrrbs_id}_amplified_genes.tsv")
                    del_genes_path = os.path.join(cfrrbs_dir, f"{cfrrbs_id}_deleted_genes.tsv")
                    amp_genes_df.to_csv(amp_genes_path, sep="\t", index=False)
                    del_genes_df.to_csv(del_genes_path, sep="\t", index=False)
                    logging.info(f"[{pair_id}] Wrote TSVs: {amp_genes_path}, {del_genes_path}")
                else:
                    logging.warning(f"[{pair_id}] cf_gene_calls is empty - no TSVs written")

                # Try to find the most comprehensive EPIC detail TSV in epic_dir.
                # Prefer '<epic_id>_detail.tsv' or any '*_detail.tsv' that does not
                # contain 'del' in the filename. Fallback to any 'detail' TSV.
                detail_epic_path = None
                candidates = [f for f in os.listdir(epic_dir) if "detail" in f.lower() and f.lower().endswith(".tsv")]
                # exact match
                exact = f"{epic_id}_detail.tsv"
                if exact in candidates:
                    detail_epic_path = os.path.join(epic_dir, exact)
                else:
                    # prefer non-'del' detail files
                    non_del = [f for f in candidates if 'del' not in f.lower()]
                    if non_del:
                        detail_epic_path = os.path.join(epic_dir, non_del[0])
                    elif candidates:
                        detail_epic_path = os.path.join(epic_dir, candidates[0])

                # If EPIC detail exists, load and merge for plotting/concordance
                if detail_epic_path is None:
                    logging.warning(f"[{pair_id}] No EPIC detail TSV found in {epic_dir}; candidates={candidates}; skipping gene plots")
                elif not os.path.exists(detail_epic_path):
                    logging.warning(f"[{pair_id}] EPIC detail path does not exist: {detail_epic_path}; skipping gene plots")
                if detail_epic_path is not None and os.path.exists(detail_epic_path):
                    logging.info(f"[{pair_id}] Loading EPIC detail TSV: {detail_epic_path}")
                    try:
                        epic_detail_df = pd.read_csv(detail_epic_path, sep="\t")
                        # Use detail-specific column detection (.Chromosome, .Start, .End, .Name, .Value)
                        col_map = _find_epic_detail_columns(list(epic_detail_df.columns))
                        rename_map = {}
                        if 'chr' in col_map:
                            rename_map[col_map['chr']] = 'chr'
                        if 'start' in col_map:
                            rename_map[col_map['start']] = 'start'
                        if 'end' in col_map:
                            rename_map[col_map['end']] = 'end'
                        if 'name' in col_map:
                            rename_map[col_map['name']] = 'name'
                        if 'ratio' in col_map:
                            rename_map[col_map['ratio']] = 'ratio'
                        epic_detail_df = epic_detail_df.rename(columns=rename_map, errors='ignore')

                        # If still missing a ratio column, try numeric fallback
                        if "ratio" not in epic_detail_df.columns:
                            numeric_cols = [
                                c
                                for c in epic_detail_df.select_dtypes(include=["number"]).columns
                                if c not in ["start", "end"]
                            ]
                            if numeric_cols:
                                epic_detail_df["ratio"] = epic_detail_df[numeric_cols[0]]

                        # If EPIC provides a single 'region' column, either parse genomic coords
                        # like 'chr1:100-200' or treat the value as a gene `name` when it does
                        # not contain coordinates (e.g., 'MDM4'). This ensures we can match
                        # EPIC detail rows to CF detail by gene name when available.
                        if "region" in epic_detail_df.columns:
                            # If region values look like genomic ranges (contain ':'), parse
                            if epic_detail_df["region"].astype(str).str.contains(":").any():
                                def _parse_region(val):
                                    try:
                                        s = str(val).strip()
                                        if s.startswith("chr"):
                                            s = s[3:]
                                        parts = s.split(":")
                                        if len(parts) == 2:
                                            chrom = parts[0]
                                            coords = parts[1].split("-")
                                            if len(coords) == 2:
                                                start = int(coords[0].replace(",", ""))
                                                end = int(coords[1].replace(",", ""))
                                                return chrom, start, end
                                    except Exception:
                                        return None, None, None
                                    return None, None, None

                                parsed = epic_detail_df["region"].apply(lambda r: pd.Series(_parse_region(r), index=["chr", "start", "end"]))
                                epic_detail_df = pd.concat([epic_detail_df, parsed], axis=1)
                            else:
                                # region contains non-coordinate names -> treat as gene `name`
                                epic_detail_df["name"] = epic_detail_df["region"].astype(str)

                        epic_has_name = "name" in epic_detail_df.columns

                        # Build merged per-gene table
                        merged_rows = []
                        for _, cf_row in cf_detail_df.iterrows():
                            cf_chr = str(cf_row.get("chr"))
                            cf_start = int(cf_row.get("start"))
                            cf_end = int(cf_row.get("end"))
                            cf_name = (
                                str(cf_row.get("name"))
                                if "name" in cf_detail_df.columns
                                else f"{cf_chr}:{cf_start}-{cf_end}"
                            )
                            cf_ratio = (
                                float(cf_row.get("ratio"))
                                if not pd.isna(cf_row.get("ratio"))
                                else float("nan")
                            )

                            epic_ratio = float("nan")
                            if epic_has_name:
                                matches = epic_detail_df[
                                    epic_detail_df["name"].astype(str).str.lower() == cf_name.lower()
                                ]
                                if not matches.empty and "ratio" in matches:
                                    epic_ratio = float(matches["ratio"].dropna().mean())
                            if pd.isna(epic_ratio):
                                overlaps = epic_detail_df[
                                    (epic_detail_df["chr"].astype(str) == cf_chr)
                                    & (epic_detail_df["start"] < cf_end)
                                    & (epic_detail_df["end"] > cf_start)
                                ]
                                if not overlaps.empty and "ratio" in overlaps:
                                    epic_ratio = float(overlaps["ratio"].dropna().mean())

                            merged_rows.append(
                                {
                                    "gene": cf_name,
                                    "chr": cf_chr,
                                    "start": cf_start,
                                    "end": cf_end,
                                    "cf_ratio": cf_ratio,
                                    "epic_ratio": epic_ratio,
                                }
                            )

                        merged_df = pd.DataFrame(merged_rows)
                        # Order genes by cfRRBS ratio (high to low)
                        merged_df = merged_df.sort_values("cf_ratio", ascending=False, na_position="last").reset_index(drop=True)
                        logging.info(f"[{pair_id}] Merged gene table: {len(merged_df)} rows, cf_ratio notna={merged_df['cf_ratio'].notna().sum()}, epic_ratio notna={merged_df['epic_ratio'].notna().sum()}")
                        logging.info(f"[{pair_id}] Merged columns: {merged_df.columns.tolist()}")
                        if merged_df.empty:
                            logging.warning(f"[{pair_id}] merged_df is empty — no gene plots will be generated")
                        
                        # Generate per-pair gene comparison plot
                        # Build call sets from technology-specific output TSVs
                        cf_gene_calls = set()
                        epic_gene_calls = set()

                        def _extend_calls_from_tsv(tsv_path: str, call_type: str, target_set: Set[str]) -> None:
                            if not os.path.exists(tsv_path):
                                logging.debug(f"TSV not found: {tsv_path}")
                                return
                            try:
                                tmp_df = pd.read_csv(tsv_path, sep="\t")
                                logging.debug(f"Loaded {len(tmp_df)} rows from {tsv_path}, columns: {tmp_df.columns.tolist()}")
                            except Exception as e:
                                logging.warning(f"Failed to read {tsv_path}: {e}")
                                return
                            gene_col = None
                            for candidate in ["gene", "name", "region"]:
                                if candidate in tmp_df.columns:
                                    gene_col = candidate
                                    break
                            if gene_col is None:
                                logging.warning(f"No gene column found in {tsv_path} (columns: {tmp_df.columns.tolist()})")
                                return
                            added = 0
                            for g in tmp_df[gene_col].dropna().astype(str):
                                g_clean = g.strip()
                                if g_clean:
                                    target_set.add(f"{g_clean}:{call_type}")
                                    added += 1
                            logging.info(f"Loaded {added} genes from {tsv_path} as '{call_type}'")

                        cf_amp_tsv = os.path.join(cfrrbs_dir, f"{cfrrbs_id}_amplified_genes.tsv")
                        cf_del_tsv = os.path.join(cfrrbs_dir, f"{cfrrbs_id}_deleted_genes.tsv")
                        _extend_calls_from_tsv(cf_amp_tsv, "gain", cf_gene_calls)
                        _extend_calls_from_tsv(cf_del_tsv, "deletion", cf_gene_calls)
                        logging.info(f"[{pair_id}] cfRRBS set size: {len(cf_gene_calls)} entries")

                        # For EPIC, apply same conditional logic as cfRRBS: segment-wise or conumee-style
                        if gene_call_method == "segment-wise" and epic_detail_df is not None and not epic_detail_df.empty and epic_segments is not None and not epic_segments.empty:
                            # Compute EPIC gene calls using segment-wise approach
                            try:
                                epic_gene_df = _call_cfrrbs_gene_events_segment_wise(
                                    regions_df=epic_detail_df,
                                    segments_df=epic_segments,
                                    hard_thresh=aberrant_threshold,
                                )
                                if epic_gene_df is not None and not epic_gene_df.empty:
                                    for _, row in epic_gene_df.iterrows():
                                        gene_name = str(row.get("gene", "")).strip()
                                        call_type = str(row.get("call", "")).strip()
                                        if gene_name and call_type:
                                            epic_gene_calls.add(f"{gene_name}:{call_type}")
                                    logging.info(f"[{pair_id}] Computed {len(epic_gene_df)} EPIC segment-wise gene calls")
                            except Exception as e:
                                logging.warning(f"[{pair_id}] Failed to compute EPIC segment-wise genes: {e}; falling back to TSV loading")
                                # Fallback to TSV loading
                                epic_amp_candidates = [
                                    os.path.join(epic_dir, f"{epic_id}_amp_detail_regions.tsv"),
                                    os.path.join(epic_dir, f"{epic_id}_amplified_genes.tsv"),
                                ]
                                epic_del_candidates = [
                                    os.path.join(epic_dir, f"{epic_id}_del_detail_regions.tsv"),
                                    os.path.join(epic_dir, f"{epic_id}_deleted_genes.tsv"),
                                ]
                                for p in epic_amp_candidates:
                                    _extend_calls_from_tsv(p, "gain", epic_gene_calls)
                                for p in epic_del_candidates:
                                    _extend_calls_from_tsv(p, "deletion", epic_gene_calls)
                                logging.info(f"[{pair_id}] Fallback EPIC set size: {len(epic_gene_calls)} entries")
                        else:
                            # Default: load EPIC calls from pre-computed TSV files
                            epic_amp_candidates = [
                                os.path.join(epic_dir, f"{epic_id}_amp_detail_regions.tsv"),
                                os.path.join(epic_dir, f"{epic_id}_amplified_genes.tsv"),
                            ]
                            epic_del_candidates = [
                                os.path.join(epic_dir, f"{epic_id}_del_detail_regions.tsv"),
                                os.path.join(epic_dir, f"{epic_id}_deleted_genes.tsv"),
                            ]
                            for p in epic_amp_candidates:
                                _extend_calls_from_tsv(p, "gain", epic_gene_calls)
                            for p in epic_del_candidates:
                                _extend_calls_from_tsv(p, "deletion", epic_gene_calls)
                            logging.info(f"[{pair_id}] EPIC (conumee) set size: {len(epic_gene_calls)} entries")
                        
                        logging.info(f"[{pair_id}] Calling _plot_gene_pair_comparison: merged_df={len(merged_df)} rows, cf_calls={len(cf_gene_calls)}, epic_calls={len(epic_gene_calls)}, outdir={sample_pair_dir}")
                        try:
                            _plot_gene_pair_comparison(merged_df, pair_id, sample_pair_dir, cf_gene_calls, epic_gene_calls)
                            logging.info(f"[{pair_id}] _plot_gene_pair_comparison completed successfully")
                        except Exception as e:
                            logging.error(f"[{pair_id}] _plot_gene_pair_comparison FAILED: {e}", exc_info=True)

                        # Per-pair Venn diagram
                        try:
                            # Build per-pair call lists from the "gene:call" sets
                            def _calls_set_to_list(call_set):
                                out = []
                                for entry in call_set:
                                    parts = entry.rsplit(":", 1)
                                    if len(parts) == 2:
                                        out.append({"name": parts[0], "call": parts[1]})
                                return out

                            pair_cf_list = _calls_set_to_list(cf_gene_calls)
                            pair_epic_list = _calls_set_to_list(epic_gene_calls)
                            pair_venn_counts = _compute_venn_counts(pair_cf_list, pair_epic_list)
                            _plot_venn_diagram(
                                pair_venn_counts,
                                sample_pair_dir,
                                title=f"Gene CNV Reproducibility: {pair_id}",
                                filename="gene_calls_venn.png",
                            )
                            # Accumulate for summary Venn
                            for k in total_venn_counts:
                                total_venn_counts[k] += pair_venn_counts[k]

                            # Accumulate per-gene call records for summary overview
                            cf_by_call = {}  # gene -> set of call types
                            for entry in cf_gene_calls:
                                parts = entry.rsplit(":", 1)
                                if len(parts) == 2:
                                    cf_by_call.setdefault(parts[0], set()).add(parts[1])
                            epic_by_call = {}
                            for entry in epic_gene_calls:
                                parts = entry.rsplit(":", 1)
                                if len(parts) == 2:
                                    epic_by_call.setdefault(parts[0], set()).add(parts[1])
                            all_genes = set(cf_by_call.keys()) | set(epic_by_call.keys())
                            for gene in all_genes:
                                cf_calls_g = cf_by_call.get(gene, set())
                                ep_calls_g = epic_by_call.get(gene, set())
                                for call_type in ("gain", "deletion"):
                                    in_cf = call_type in cf_calls_g
                                    in_ep = call_type in ep_calls_g
                                    if in_cf and in_ep:
                                        source = "both"
                                    elif in_cf:
                                        source = "cf_only"
                                    elif in_ep:
                                        source = "epic_only"
                                    else:
                                        continue
                                    all_gene_call_records.append({
                                        "gene": gene, "call": call_type,
                                        "source": source, "sample_id": pair_id,
                                    })
                        except Exception as e:
                            logging.error(f"[{pair_id}] Per-pair Venn FAILED: {e}", exc_info=True)
                        
                        # Collect gene data for summary boxplot across all samples
                        for _, row in merged_df.iterrows():
                            if pd.notna(row["cf_ratio"]):
                                all_gene_data.append({
                                    "gene": row["gene"],
                                    "technology": "cfRRBS",
                                    "ratio": row["cf_ratio"],
                                    "sample_id": pair_id,
                                })
                            if pd.notna(row["epic_ratio"]):
                                all_gene_data.append({
                                    "gene": row["gene"],
                                    "technology": "EPIC",
                                    "ratio": row["epic_ratio"],
                                    "sample_id": pair_id,
                                })
                        
                        # Plot if EPIC ratios exist
                        try:
                            has_epic = "epic_ratio" in merged_df.columns and merged_df["epic_ratio"].notna().any()
                            logging.info(f"[{pair_id}] ratio_genes_scatter check: has_epic_ratio_col={'epic_ratio' in merged_df.columns}, any_notna={merged_df['epic_ratio'].notna().sum() if 'epic_ratio' in merged_df.columns else 'N/A'}, will_plot={has_epic}")
                            if has_epic:
                                import matplotlib.pyplot as plt

                                # save plot alongside other per-pair files (sample_pair_dir)
                                plot_file = os.path.join(sample_pair_dir, f"ratio_genes_scatter.png")
                                fig, ax = plt.subplots(figsize=(max(10, len(merged_df) * 0.6), 5))
                                x = np.arange(len(merged_df))
                                
                                # Add constitutional reference lines (diploid)
                                ax.axhline(y=np.log2(1/2), color='red', linestyle='--', linewidth=1.5, 
                                          alpha=0.7, zorder=1)
                                ax.axhline(y=0, color='lightgrey', linestyle='--', linewidth=1.5, 
                                          alpha=0.7, zorder=1)
                                ax.axhline(y=np.log2(3/2), color='green', linestyle='--', linewidth=1.5, 
                                          alpha=0.7, zorder=1)
                                
                                # Determine significance (abs(ratio) > 0.3 threshold)
                                cnv_threshold = 0.3
                                cf_significant = merged_df["cf_ratio"].abs() > cnv_threshold
                                epic_significant = merged_df["epic_ratio"].abs() > cnv_threshold
                                
                                # Create color gradient function (red-grey-green based on ratio)
                                def get_gradient_color(ratio_val, y_min, y_max):
                                    """Map ratio value to red-grey-green gradient"""
                                    if pd.isna(ratio_val):
                                        return 'grey'
                                    # Clip to range
                                    clipped = max(min(ratio_val, y_max), y_min)
                                    # Normalize to 0-1
                                    norm_val = (clipped - y_min) / (y_max - y_min)
                                    # Red-Grey-Green gradient
                                    if norm_val < 0.5:  # Red to Grey
                                        # Interpolate from red (1,0,0) to grey (0.8,0.8,0.8)
                                        t = norm_val * 2
                                        r = 1 - 0.2 * t
                                        g = 0.8 * t
                                        b = 0.8 * t
                                    else:  # Grey to Green
                                        # Interpolate from grey (0.8,0.8,0.8) to green (0,0.7,0)
                                        t = (norm_val - 0.5) * 2
                                        r = 0.8 * (1 - t)
                                        g = 0.8 + 0.7 * t * (1 - 0.8)
                                        b = 0.8 * (1 - t)
                                    return (max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b)))
                                
                                # FIXED y-axis scale: always [-1.25, 1.25]
                                y_min, y_max = -1.25, 1.25
                                
                                # Plot cfRRBS points without connecting lines
                                # Plot cfRRBS points and connect with line
                                cf_mask = merged_df["cf_ratio"].notna()
                                cf_plot_x = x[cf_mask]
                                cf_plot_y = np.clip(merged_df.loc[cf_mask, "cf_ratio"].values, y_min, y_max)
                                cf_raw_y = merged_df.loc[cf_mask, "cf_ratio"].values
                                cf_sig = cf_significant[cf_mask].values

                                cfrrbs_plotted = False
                                for i, (xi, pv, rv, is_sig) in enumerate(zip(cf_plot_x, cf_plot_y, cf_raw_y, cf_sig)):
                                    face_color = get_gradient_color(rv, y_min, y_max)
                                    edge_width = 2.5 if is_sig else 1.5
                                    ax.scatter(xi, pv, marker='o', s=100,
                                              facecolors=face_color, edgecolors='C0',
                                              linewidths=edge_width, zorder=3,
                                              label='cfRRBS' if not cfrrbs_plotted else '')
                                    if rv < y_min or rv > y_max:
                                        ax.text(xi, pv, f'{rv:.2f}', fontsize=7, ha='center',
                                               va='bottom' if rv > y_max else 'top', color='C0', fontweight='bold')
                                    cfrrbs_plotted = True

                                if len(cf_plot_x) >= 2:
                                    ax.plot(cf_plot_x, cf_plot_y, color='C0', linewidth=2,
                                           linestyle='-', alpha=0.5, zorder=2)

                                # Plot EPIC points and connect with line
                                epic_mask = merged_df["epic_ratio"].notna()
                                epic_plot_x = x[epic_mask]
                                epic_plot_y = np.clip(merged_df.loc[epic_mask, "epic_ratio"].values, y_min, y_max)
                                epic_raw_y = merged_df.loc[epic_mask, "epic_ratio"].values
                                epic_sig = epic_significant[epic_mask].values

                                epic_plotted = False
                                for i, (xi, pv, rv, is_sig) in enumerate(zip(epic_plot_x, epic_plot_y, epic_raw_y, epic_sig)):
                                    face_color = get_gradient_color(rv, y_min, y_max)
                                    edge_width = 2.5 if is_sig else 1.5
                                    ax.scatter(xi, pv, marker='s', s=100,
                                              facecolors=face_color, edgecolors='C1',
                                              linewidths=edge_width, zorder=3,
                                              label='EPIC' if not epic_plotted else '')
                                    if rv < y_min or rv > y_max:
                                        ax.text(xi, pv, f'{rv:.2f}', fontsize=7, ha='center',
                                               va='bottom' if rv > y_max else 'top', color='C1', fontweight='bold')
                                    epic_plotted = True

                                if len(epic_plot_x) >= 2:
                                    ax.plot(epic_plot_x, epic_plot_y, color='C1', linewidth=2,
                                           linestyle='-', alpha=0.5, zorder=2)
                                
                                # Set y-axis limits (FIXED)
                                ax.set_ylim([y_min, y_max])
                                
                                ax.set_xticks(x)
                                ax.set_xticklabels(merged_df["gene"].tolist(), rotation=45, ha='right', fontsize=8)
                                ax.set_ylabel("log2 ratio", fontsize=10)
                                ax.set_title(f"Detail genes: {pair_id}", fontsize=11)
                                
                                # Add both horizontal and vertical gridlines
                                ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
                                ax.grid(axis='x', alpha=0.2, linestyle=':', linewidth=0.5)
                                
                                # Place legend outside plot area
                                ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, frameon=True)
                                
                                plt.tight_layout()
                                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                                plt.close()
                                logging.info(f"[{pair_id}] ratio_genes_scatter.png saved to {plot_file}")
                        except Exception as e:
                            logging.error(f"[{pair_id}] FAILED to create detail genes scatter plot: {e}", exc_info=True)

                        # Compute concordance if possible
                        try:
                            valid_pairs = merged_df.dropna(subset=["cf_ratio", "epic_ratio"])
                            if len(valid_pairs) >= 3:
                                conc_detail = _compute_cnv_concordance(
                                    valid_pairs.rename(columns={"cf_ratio": "ratio_cf", "epic_ratio": "ratio_epic"})[["ratio_cf", "ratio_epic"]],
                                    valid_pairs.rename(columns={"cf_ratio": "ratio_cf", "epic_ratio": "ratio_epic"})[["ratio_cf", "ratio_epic"]],
                                    "detail_regions",
                                )
                        except Exception:
                            conc_detail = None
                    except Exception as e:
                        logging.debug(f"Could not compute detail concordance: {e}")
            except Exception as e:
                logging.error(f"Failed to process detail file {detail_cf_path}: {e}", exc_info=True)

        report_rows.append(
            {
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
                "segments_agreement": conc_segments.get("agreement", float("nan")),
                "segments_concordance_gain": conc_segments.get("concordance_gain", float("nan")),
                "segments_concordance_deletion": conc_segments.get("concordance_deletion", float("nan")),
                "segments_positive_agreement": conc_segments.get("positive_agreement", float("nan")),
                "segments_negative_agreement": conc_segments.get("negative_agreement", float("nan")),
                "detail_agreement": conc_detail.get("agreement", float("nan")) if conc_detail else float("nan"),
                "detail_concordance_gain": conc_detail.get("concordance_gain", float("nan")) if conc_detail else float("nan"),
                "detail_concordance_deletion": conc_detail.get("concordance_deletion", float("nan")) if conc_detail else float("nan"),
                "detail_positive_agreement": conc_detail.get("positive_agreement", float("nan")) if conc_detail else float("nan"),
                "detail_negative_agreement": conc_detail.get("negative_agreement", float("nan")) if conc_detail else float("nan"),
                "segments_aberrant_pearson": ab_stats.get("pearson", float("nan")),
                "segments_aberrant_pearson_p": ab_stats.get("pearson_p", float("nan")),
                "segments_aberrant_spearman": ab_stats.get("spearman", float("nan")),
                "segments_aberrant_spearman_p": ab_stats.get("spearman_p", float("nan")),
                "segments_aberrant_n": ab_stats.get("n_segments", 0),
                "cf_segment_ratios": cf_segments["ratio"].dropna().tolist() if (isinstance(cf_segments, pd.DataFrame) and "ratio" in cf_segments.columns and not cf_segments.empty) else [],
                "epic_segment_ratios": epic_segments["ratio"].dropna().tolist() if (isinstance(epic_segments, pd.DataFrame) and "ratio" in epic_segments.columns and not epic_segments.empty) else [],
            }
        )

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
    # Final report generation
    # ===================================================================
    logging.info("Phase 4: Writing final reports")
    
    # Summary directory already created above
    
    # Generate summary gene boxplot across all samples
    if all_gene_data:
        try:
            _plot_gene_boxplot_summary(all_gene_data, summary_dir)
            logging.info("Gene summary boxplot written to summary folder")
        except Exception as e:
            logging.warning(f"Failed to create gene summary boxplot: {e}")

    # Generate gene call heatmap (concordance between cfRRBS and EPIC)
    if all_gene_data:
        try:
            _plot_gene_call_heatmap(all_gene_data, all_gene_call_records, summary_dir)
            logging.info("Gene call heatmap written to summary folder")
        except Exception as e:
            logging.warning(f"Failed to create gene call heatmap: {e}")
    
    # Generate compact correlation summary visualization (all)
    if report_rows:
        try:
            _plot_correlation_summary(report_rows, summary_dir)
            logging.info("Correlation summary plot written to summary folder")
        except Exception as e:
            logging.warning(f"Failed to create correlation summary plot: {e}")
    
    # Generate aberrant segments summary scatter
    if all_aberrant_pairs:
        try:
            _plot_aberrant_summary_scatter(all_aberrant_pairs, summary_dir)
            logging.info("Aberrant segments summary scatter written to summary folder")
        except Exception as e:
            logging.warning(f"Failed to create aberrant summary scatter: {e}")
    
    # Generate summary Venn diagram (sum of all per-pair Venn counts)
    logging.info(f"Summary Venn check: total_venn_counts={total_venn_counts}")
    try:
        _plot_venn_diagram(total_venn_counts, summary_dir,
                           title="Gene CNV Call Reproducibility: cfRRBS vs EPIC (all samples summed)")
        logging.info("Summary Venn diagram written to summary folder")
    except Exception as e:
        logging.warning(f"Failed to create summary Venn diagram: {e}")
    
    # Collect all stacked genome-wide plots to dedicated folder
    try:
        _collect_genome_wide_stacked_plots(out_dir, pair_metadata)
        logging.info("Genome-wide CNV plots collected to genome_wide_CNV folder")
    except Exception as e:
        logging.warning(f"Failed to collect genome-wide CNV plots: {e}")
    
    logging.info(f"EPIC vs cfRRBS pipeline completed successfully ({len(report_rows)} pairs)")


