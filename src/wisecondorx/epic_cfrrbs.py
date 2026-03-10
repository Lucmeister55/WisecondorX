import argparse
import csv
import json
import logging
import math
import os
import subprocess
import tempfile
from statistics import NormalDist
from typing import Dict, List, Set, Tuple

import numpy as np
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
    subprocess.check_call(cmd)


def _normalize_chr(chr_value: str) -> str:
    val = str(chr_value).replace("chr", "")
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
        
        # Build dynamic column mapping for R-style names: X{epic_id}.colname
        r_prefix = f"X{epic_id}."
        col_mapping = {
            # Standard names
            "seqnames": "chr", "Chromosome": "chr", "chrom": "chr",
            "Start": "start", "End": "end",
            # R-style names with epic_id prefix
            f"{r_prefix}chrom": "chr",
            f"{r_prefix}start": "start",
            f"{r_prefix}end": "end",
            f"{r_prefix}loc.start": "start",
            f"{r_prefix}loc.end": "end",
        }
        df_bins = df_bins.rename(columns=col_mapping, errors="ignore")
        
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
        
        # Standardize segment column names with dynamic R-style prefix mapping
        df_segments = df_segments.rename(columns=col_mapping, errors="ignore")
        
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
        
        # Find ratio column for segments (seg.mean or seg.median)
        if "ratio" not in df_segments.columns:
            # Try common segment ratio names
            ratio_candidates = [
                f"{r_prefix}seg.mean", f"{r_prefix}seg.median",
                "seg.mean", "seg.median", "seg.mean.log2"
            ]
            for col in ratio_candidates:
                if col in df_segments.columns:
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
    """Pair cfRRBS and EPIC segments by genomic overlap."""
    rows = []
    for chr_name in cf_segments["chr"].unique():
        cf_chr = cf_segments[cf_segments["chr"] == chr_name]
        epic_chr = epic_segments[epic_segments["chr"] == chr_name]
        if cf_chr.empty or epic_chr.empty:
            continue
        for _, cf_row in cf_chr.iterrows():
            overlaps = epic_chr[
                (epic_chr["start"] <= cf_row["end"]) & (epic_chr["end"] >= cf_row["start"])
            ]
            for _, ep_row in overlaps.iterrows():
                if "ratio" in cf_row and "ratio" in ep_row:
                    rows.append({
                        "chr": cf_row.get("chr", chr_name),
                        "start": cf_row.get("start", np.nan),
                        "end": cf_row.get("end", np.nan),
                        "ratio_cf": cf_row["ratio"],
                        "ratio_epic": ep_row["ratio"],
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
    pairs = pairs.dropna(subset=["ratio_cf", "ratio_epic"])
    
    if len(pairs) < 3:
        return {
            "pearson": float("nan"),
            "pearson_p": float("nan"),
            "spearman": float("nan"),
            "spearman_p": float("nan"),
            "n_segments": 0,
        }

    pairs_df = pd.DataFrame(pairs, columns=["ratio_cf", "ratio_epic"]).dropna()
    if pairs_df.empty or len(pairs_df) < 3:
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


def _generate_gene_reproducibility_report(cfrrbs_calls: List[Dict], epic_calls: List[Dict], outdir: str) -> None:
    """
    Generate report showing gene amplification/deletion reproducibility between cfRRBS and EPIC.
    
    Args:
        cfrrbs_calls: List of dicts with cfRRBS gene calls (name, call, sample_id, etc.)
        epic_calls: List of dicts with EPIC gene calls (name, call, sample_id, etc.)
        outdir: Output directory for reproducibility report
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not cfrrbs_calls and not epic_calls:
            logging.warning("No gene calls available for reproducibility report")
            return
        
        cfrrbs_df = pd.DataFrame(cfrrbs_calls) if cfrrbs_calls else pd.DataFrame()
        epic_df = pd.DataFrame(epic_calls) if epic_calls else pd.DataFrame()
        
        # Build gene-call sets per technology
        cfrrbs_amp_genes = set(cfrrbs_df[cfrrbs_df["call"] == "gain"]["name"].unique()) if not cfrrbs_df.empty else set()
        cfrrbs_del_genes = set(cfrrbs_df[cfrrbs_df["call"] == "deletion"]["name"].unique()) if not cfrrbs_df.empty else set()
        epic_amp_genes = set(epic_df[epic_df["call"] == "gain"]["name"].unique()) if not epic_df.empty else set()
        epic_del_genes = set(epic_df[epic_df["call"] == "deletion"]["name"].unique()) if not epic_df.empty else set()
        
        # Compute overlaps and unique genes
        amp_both = cfrrbs_amp_genes & epic_amp_genes
        amp_cfrrbs_only = cfrrbs_amp_genes - epic_amp_genes
        amp_epic_only = epic_amp_genes - cfrrbs_amp_genes
        
        del_both = cfrrbs_del_genes & epic_del_genes
        del_cfrrbs_only = cfrrbs_del_genes - epic_del_genes
        del_epic_only = epic_del_genes - cfrrbs_del_genes
        
        # Create venn diagram visualization (with fallback if matplotlib_venn not available)
        try:
            from matplotlib_venn import venn2
            use_venn2 = True
        except ImportError:
            use_venn2 = False
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        
        if use_venn2:
            # Use matplotlib_venn if available
            # Amplifications venn diagram
            v_amp = venn2(
                subsets=[len(amp_cfrrbs_only), len(amp_epic_only), len(amp_both)],
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
            axes[0].set_title("Amplified Genes Reproducibility", fontsize=13, fontweight="bold")
            axes[0].set_xlim(-1.35, 1.35)
            axes[0].set_ylim(-1.1, 1.2)
            
            # Deletions venn diagram
            v_del = venn2(
                subsets=[len(del_cfrrbs_only), len(del_epic_only), len(del_both)],
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
            axes[1].set_title("Deleted Genes Reproducibility", fontsize=13, fontweight="bold")
            axes[1].set_xlim(-1.35, 1.35)
            axes[1].set_ylim(-1.1, 1.2)
        else:
            # Fallback: draw simple circles manually
            from matplotlib.patches import Circle
            
            # Custom venn-like visualization using circles and text
            def draw_venn_fallback(ax, left_only, right_only, both, left_label, right_label, title):
                ax.set_xlim(-1.7, 1.7)
                ax.set_ylim(-1.1, 1.9)
                ax.set_aspect('equal')
                ax.axis('off')
                
                # Draw circles
                circle_left = Circle((-0.45, 0.35), 0.78, facecolor='#5DADE2', edgecolor='#2E86C1', alpha=0.35, linewidth=2)
                circle_right = Circle((0.45, 0.35), 0.78, facecolor='#F1948A', edgecolor='#CB4335', alpha=0.35, linewidth=2)
                ax.add_patch(circle_left)
                ax.add_patch(circle_right)
                
                # Add labels
                ax.text(-0.95, 1.35, left_label, fontsize=12, fontweight='bold', ha='center', va='center')
                ax.text(0.95, 1.35, right_label, fontsize=12, fontweight='bold', ha='center', va='center')
                
                # Add counts
                ax.text(-0.88, 0.35, f"{left_only}", fontsize=16, fontweight='bold', ha='center', va='center')
                ax.text(0.0, 0.35, f"{both}", fontsize=16, fontweight='bold', ha='center', va='center')
                ax.text(0.88, 0.35, f"{right_only}", fontsize=16, fontweight='bold', ha='center', va='center')
                
                ax.set_title(title, fontsize=13, fontweight='bold')
            
            # Draw amplifications venn
            draw_venn_fallback(axes[0], len(amp_cfrrbs_only), len(amp_epic_only), len(amp_both),
                             'cfRRBS', 'EPIC', "Amplified Genes Reproducibility")
            
            # Draw deletions venn
            draw_venn_fallback(axes[1], len(del_cfrrbs_only), len(del_epic_only), len(del_both),
                             'cfRRBS', 'EPIC', "Deleted Genes Reproducibility")
        
        fig.suptitle("Gene CNV Call Reproducibility: cfRRBS vs EPIC", fontsize=15, fontweight="bold")
        
        repro_plot = os.path.join(outdir, "gene_calls_reproducibility.png")
        plt.savefig(repro_plot, dpi=300, bbox_inches="tight")
        plt.close()
        
        logging.info(f"Gene reproducibility plot saved to {repro_plot}")
        
    except Exception as e:
        logging.error(f"Failed to generate gene reproducibility report: {e}")
        raise


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
        
        # Extract correlations
        bins_pearson = df["bins_pearson"].dropna().values
        segments_pearson = df["segments_pearson"].dropna().values
        
        if len(bins_pearson) == 0 and len(segments_pearson) == 0:
            logging.warning("No valid correlation values for summary plot")
            return
        
        # Create figure with boxplots
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
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
        if len(segments_pearson) > 0:
            bp2 = axes[1].boxplot([segments_pearson], labels=['cfRRBS vs EPIC'], patch_artist=True,
                                   widths=0.5, showmeans=True)
            bp2['boxes'][0].set_facecolor('#e74c3c')
            bp2['boxes'][0].set_alpha(0.7)
            axes[1].scatter([1]*len(segments_pearson), segments_pearson, alpha=0.4, s=30, color='#2c3e50', zorder=3)
            axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            axes[1].set_ylabel('Pearson Correlation', fontsize=12, fontweight='bold')
            axes[1].set_title(f'Segments Correlation (n={len(segments_pearson)})', fontsize=13, fontweight='bold')
            axes[1].set_ylim([-1.1, 1.1])
            axes[1].grid(axis='y', alpha=0.3, linestyle=':')
            axes[1].text(
                0.03, 0.97, _stats_text(segments_pearson),
                transform=axes[1].transAxes,
                va='top', ha='left', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#e74c3c')
            )
        else:
            axes[1].text(0.5, 0.5, 'No segments data', ha='center', va='center', transform=axes[1].transAxes)
        
        plt.suptitle('EPIC vs cfRRBS Correlation Summary', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = os.path.join(outdir, "correlation_summary.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Correlation summary plot saved to {output_path}")
        
    except Exception as e:
        logging.error(f"Failed to create correlation summary plot: {e}")
        raise


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
        
        if merged_df.empty:
            logging.debug(f"No merged gene data for pair {pair_id}")
            return
        
        # Filter to genes with both ratios
        valid_df = merged_df[(merged_df["cf_ratio"].notna()) & (merged_df["epic_ratio"].notna())].copy()
        if valid_df.empty:
            logging.debug(f"No genes with both ratios for pair {pair_id}")
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
        
        # Add reference lines
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.8, zorder=1)
        ax.axhline(y=0.3, color='green', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
        ax.axhline(y=-0.3, color='red', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
        
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
        output_path = os.path.join(outdir, "gene_aberrations.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Gene aberrations plot saved to {output_path}")
        
    except Exception as e:
        logging.debug(f"Failed to create gene aberrations plot for {pair_id}: {e}")


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
        
        # Get unique genes and sort them
        genes = sorted(df["gene"].unique())
        
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
        
        # Draw trendline between cfRRBS medians if multiple genes (smooth spline)
        if len(medians_cf) >= 3:  # Need at least 3 points for smooth spline
            from scipy.interpolate import UnivariateSpline
            cf_pos = np.array([m[0] for m in medians_cf])
            cf_med = np.array([m[1] for m in medians_cf])
            try:
                # Create smooth spline through median points
                cf_spline = UnivariateSpline(cf_pos, cf_med, k=min(3, len(cf_pos)-1), s=0.15)
                # Evaluate spline at fine-grained points
                cf_smooth_x = np.linspace(cf_pos.min(), cf_pos.max(), 100)
                cf_smooth_y = cf_spline(cf_smooth_x)
                cf_smooth_y = np.clip(cf_smooth_y, -1.25, 1.25)
                ax.plot(cf_smooth_x, cf_smooth_y, color='C0', linewidth=2.5, linestyle='-', 
                       alpha=0.6, zorder=2, label='cfRRBS median trend')
                # Mark median points
                ax.plot(cf_pos, np.clip(cf_med, -1.25, 1.25), color='C0', marker='o', 
                       markersize=6, linestyle='none', alpha=0.7, zorder=3)
            except:
                # Fallback to straight line if spline fails
                cf_med_clipped = np.clip(cf_med, -1.25, 1.25)
                ax.plot(cf_pos, cf_med_clipped, color='C0', linewidth=2, linestyle='-', 
                       alpha=0.5, zorder=2, marker='o', markersize=3, label='cfRRBS median trend')
        elif len(medians_cf) >= 2:
            # Straight line for 2 points
            cf_pos = np.array([m[0] for m in medians_cf])
            cf_med = np.array([m[1] for m in medians_cf])
            cf_med_clipped = np.clip(cf_med, -1.25, 1.25)
            ax.plot(cf_pos, cf_med_clipped, color='C0', linewidth=2, linestyle='-', 
                   alpha=0.5, zorder=2, marker='o', markersize=3, label='cfRRBS median trend')
        
        # Draw trendline between EPIC medians if multiple genes (smooth spline)
        if len(medians_epic) >= 3:  # Need at least 3 points for smooth spline
            from scipy.interpolate import UnivariateSpline
            epic_pos = np.array([m[0] for m in medians_epic])
            epic_med = np.array([m[1] for m in medians_epic])
            try:
                # Create smooth spline through median points
                epic_spline = UnivariateSpline(epic_pos, epic_med, k=min(3, len(epic_pos)-1), s=0.15)
                # Evaluate spline at fine-grained points
                epic_smooth_x = np.linspace(epic_pos.min(), epic_pos.max(), 100)
                epic_smooth_y = epic_spline(epic_smooth_x)
                epic_smooth_y = np.clip(epic_smooth_y, -1.25, 1.25)
                ax.plot(epic_smooth_x, epic_smooth_y, color='C1', linewidth=2.5, linestyle='-', 
                       alpha=0.6, zorder=2, label='EPIC median trend')
                # Mark median points
                ax.plot(epic_pos, np.clip(epic_med, -1.25, 1.25), color='C1', marker='s', 
                       markersize=6, linestyle='none', alpha=0.7, zorder=3)
            except:
                # Fallback to straight line if spline fails
                epic_med_clipped = np.clip(epic_med, -1.25, 1.25)
                ax.plot(epic_pos, epic_med_clipped, color='C1', linewidth=2, linestyle='-', 
                       alpha=0.5, zorder=2, marker='s', markersize=3, label='EPIC median trend')
        elif len(medians_epic) >= 2:
            # Straight line for 2 points
            epic_pos = np.array([m[0] for m in medians_epic])
            epic_med = np.array([m[1] for m in medians_epic])
            epic_med_clipped = np.clip(epic_med, -1.25, 1.25)
            ax.plot(epic_pos, epic_med_clipped, color='C1', linewidth=2, linestyle='-', 
                   alpha=0.5, zorder=2, marker='s', markersize=3, label='EPIC median trend')
        
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
        output_path = os.path.join(outdir, "gene_summary_boxplot.png")
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

    target_size = cfrrbs_img.size
    if epic_img.size != target_size:
        epic_img = epic_img.resize(target_size, Image.LANCZOS)

    title_height = 10
    stacked_height = epic_img.height + cfrrbs_img.height + title_height
    stacked_width = target_size[0]
    stacked = Image.new("RGB", (stacked_width, stacked_height), color="white")

    draw = ImageDraw.Draw(stacked)
    font = ImageFont.load_default()
    draw.text((10, 2), title, fill="black", font=font)

    stacked.paste(epic_img, (0, title_height))
    stacked.paste(cfrrbs_img, (0, title_height + epic_img.height))

    stacked.save(out_png)


def tool_epic_cfrrbs(args: argparse.Namespace) -> None:
    logging.info("Starting EPIC vs cfRRBS correlation")

    wd = str(os.path.dirname(os.path.realpath(__file__)))
    pairs = _read_pairs_from_sheet(args.sample_sheet)

    ref_file = np.load(args.reference, encoding="latin1", allow_pickle=True)
    binsize = int(ref_file["binsize"])
    del ref_file

    out_dir = os.path.abspath(args.outdir)
    summary_dir = os.path.join(out_dir, "summary")
    samples_dir = os.path.join(out_dir, "samples")
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
    
    report_rows = []
    all_gene_data = []  # Collect gene-level data across all samples for summary boxplot
    all_cfrrbs_gene_calls = []  # Combined amplified/deleted gene calls across samples (cfRRBS)
    all_epic_gene_calls = []  # Combined amplified/deleted gene calls across samples (EPIC)

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
        plot_path = os.path.join(sample_pair_dir, "bins_scatter.png")
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
            segments_plot_path = os.path.join(sample_pair_dir, "segments_scatter.png")
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

        # Compute CNV concordance at segment level
        conc_segments = _compute_cnv_concordance(cf_segments, epic_segments, "segments")
        
        # Try to load detail regions for gene-level concordance
        conc_detail = None
        detail_cf_path = temp_outid + "_regions.bed"
        # Always produce a CF detail TSV from the cfRRBS regions file if present
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

                # Build cfRRBS amplified/deleted gene list inspired by CNV.focal logic
                # using ratio thresholds and support from WisecondorX aberration overlap.
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

                    amp_genes_path = os.path.join(cfrrbs_dir, f"{cfrrbs_id}_amplified_genes.tsv")
                    del_genes_path = os.path.join(cfrrbs_dir, f"{cfrrbs_id}_deleted_genes.tsv")
                    amp_genes_df.to_csv(amp_genes_path, sep="\t", index=False)
                    del_genes_df.to_csv(del_genes_path, sep="\t", index=False)

                    if not amp_genes_df.empty:
                        tmp = amp_genes_df.copy()
                        tmp["sample_id"] = pair_id
                        tmp.rename(columns={"gene": "name"}, inplace=True)
                        all_cfrrbs_gene_calls.extend(tmp.to_dict("records"))
                    if not del_genes_df.empty:
                        tmp = del_genes_df.copy()
                        tmp["sample_id"] = pair_id
                        tmp.rename(columns={"gene": "name"}, inplace=True)
                        all_cfrrbs_gene_calls.extend(tmp.to_dict("records"))

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
                if os.path.exists(detail_epic_path):
                    try:
                        epic_detail_df = pd.read_csv(detail_epic_path, sep="\t")
                        # Use dynamic column detection to map EPIC columns to standard names
                        col_map = _find_epic_columns(list(epic_detail_df.columns))
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
                        
                        # Generate per-pair gene comparison plot
                        # Build call sets from technology-specific output TSVs
                        cf_gene_calls = set()
                        epic_gene_calls = set()

                        def _extend_calls_from_tsv(tsv_path: str, call_type: str, target_set: Set[str]) -> None:
                            if not os.path.exists(tsv_path):
                                return
                            try:
                                tmp_df = pd.read_csv(tsv_path, sep="\t")
                            except Exception:
                                return
                            gene_col = None
                            for candidate in ["gene", "name", "region"]:
                                if candidate in tmp_df.columns:
                                    gene_col = candidate
                                    break
                            if gene_col is None:
                                return
                            for g in tmp_df[gene_col].dropna().astype(str):
                                g_clean = g.strip()
                                if g_clean:
                                    target_set.add(f"{g_clean}:{call_type}")

                        cf_amp_tsv = os.path.join(cfrrbs_dir, f"{cfrrbs_id}_amplified_genes.tsv")
                        cf_del_tsv = os.path.join(cfrrbs_dir, f"{cfrrbs_id}_deleted_genes.tsv")
                        _extend_calls_from_tsv(cf_amp_tsv, "gain", cf_gene_calls)
                        _extend_calls_from_tsv(cf_del_tsv, "deletion", cf_gene_calls)

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
                        
                        try:
                            _plot_gene_pair_comparison(merged_df, pair_id, sample_pair_dir, cf_gene_calls, epic_gene_calls)
                        except Exception as e:
                            logging.debug(f"Could not create gene pair comparison plot: {e}")
                        
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
                        
                        # Collect EPIC gene calls (significant amplifications/deletions)
                        ratio_threshold = 0.3
                        for _, row in merged_df.iterrows():
                            if pd.notna(row["epic_ratio"]):
                                if row["epic_ratio"] > ratio_threshold:
                                    all_epic_gene_calls.append({
                                        "name": row["gene"],
                                        "chr": row.get("chr", ""),
                                        "start": row.get("start", ""),
                                        "end": row.get("end", ""),
                                        "ratio": row["epic_ratio"],
                                        "call": "gain",
                                        "sample_id": pair_id,
                                    })
                                elif row["epic_ratio"] < -ratio_threshold:
                                    all_epic_gene_calls.append({
                                        "name": row["gene"],
                                        "chr": row.get("chr", ""),
                                        "start": row.get("start", ""),
                                        "end": row.get("end", ""),
                                        "ratio": row["epic_ratio"],
                                        "call": "deletion",
                                        "sample_id": pair_id,
                                    })

                        # Plot if EPIC ratios exist
                        try:
                            if "epic_ratio" in merged_df.columns and merged_df["epic_ratio"].notna().any():
                                import matplotlib.pyplot as plt

                                # save plot alongside other per-pair files (sample_pair_dir)
                                plot_file = os.path.join(sample_pair_dir, f"detail_genes.png")
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
                                # Pre-calculate trendlines to use for gradient coloring (smooth spline)
                                from scipy.interpolate import UnivariateSpline
                                cf_poly = None
                                epic_poly = None
                                
                                # cfRRBS trendline (smooth spline)
                                cf_mask = merged_df["cf_ratio"].notna()
                                if cf_mask.sum() >= 3:  # Need at least 3 points for smooth spline
                                    cf_x = x[cf_mask]
                                    cf_y = merged_df.loc[cf_mask, "cf_ratio"].values
                                    # Use spline with smoothing factor (higher s for smoother curves)
                                    try:
                                        cf_poly = UnivariateSpline(cf_x, cf_y, k=min(3, len(cf_x)-1), s=0.5)
                                    except:
                                        cf_poly = None
                                
                                # EPIC trendline (smooth spline)
                                epic_mask = merged_df["epic_ratio"].notna()
                                if epic_mask.sum() >= 3:  # Need at least 3 points for smooth spline
                                    epic_x = x[epic_mask]
                                    epic_y = merged_df.loc[epic_mask, "epic_ratio"].values
                                    # Use spline with smoothing factor (higher s for smoother curves)
                                    try:
                                        epic_poly = UnivariateSpline(epic_x, epic_y, k=min(3, len(epic_x)-1), s=0.5)
                                    except:
                                        epic_poly = None
                                
                                # Plot cfRRBS points with gradient based on trendline value
                                cfrrbs_plotted = False
                                for i, (cf_val, is_sig) in enumerate(zip(merged_df["cf_ratio"], cf_significant)):
                                    if pd.notna(cf_val):
                                        marker_style = 'o'
                                        # Use trendline value for coloring, or actual value if no trendline
                                        color_val = cf_poly(i) if cf_poly is not None else cf_val
                                        face_color = get_gradient_color(color_val, y_min, y_max)
                                        edge_color = 'C0'  # Blue edge for cfRRBS
                                        edge_width = 2.5 if is_sig else 1.5
                                        # Clip to plot boundaries but remember actual value
                                        plot_val = max(min(cf_val, y_max), y_min)
                                        ax.scatter(i, plot_val, marker=marker_style, s=100, 
                                                  facecolors=face_color, edgecolors=edge_color, 
                                                  linewidths=edge_width, zorder=3,
                                                  label='cfRRBS' if not cfrrbs_plotted else '')
                                        # Add text label for outliers
                                        if cf_val < y_min or cf_val > y_max:
                                            ax.text(i, plot_val, f'{cf_val:.2f}', fontsize=7, ha='center', 
                                                   va='bottom' if cf_val > y_max else 'top', color='C0', fontweight='bold')
                                        cfrrbs_plotted = True
                                
                                # Plot EPIC points with gradient based on trendline value
                                epic_plotted = False
                                for i, (epic_val, is_sig) in enumerate(zip(merged_df["epic_ratio"], epic_significant)):
                                    if pd.notna(epic_val):
                                        marker_style = 's'
                                        # Use trendline value for coloring, or actual value if no trendline
                                        color_val = epic_poly(i) if epic_poly is not None else epic_val
                                        face_color = get_gradient_color(color_val, y_min, y_max)
                                        edge_color = 'C1'  # Orange edge for EPIC
                                        edge_width = 2.5 if is_sig else 1.5
                                        # Clip to plot boundaries but remember actual value
                                        plot_val = max(min(epic_val, y_max), y_min)
                                        ax.scatter(i, plot_val, marker=marker_style, s=100, 
                                                  facecolors=face_color, edgecolors=edge_color, 
                                                  linewidths=edge_width, zorder=3,
                                                  label='EPIC' if not epic_plotted else '')
                                        # Add text label for outliers
                                        if epic_val < y_min or epic_val > y_max:
                                            ax.text(i, plot_val, f'{epic_val:.2f}', fontsize=7, ha='center', 
                                                   va='bottom' if epic_val > y_max else 'top', color='C1', fontweight='bold')
                                        epic_plotted = True
                                
                                # Draw trendlines through each technology's points
                                # cfRRBS trendline (blue) - using pre-calculated spline
                                if cf_poly is not None and cf_mask.sum() >= 2:
                                    cf_x = x[cf_mask]
                                    cf_line_x = np.linspace(cf_x.min(), cf_x.max(), 100)
                                    cf_line_y = np.clip(cf_poly(cf_line_x), y_min, y_max)
                                    ax.plot(cf_line_x, cf_line_y, color='C0', linewidth=2.5, 
                                           linestyle='-', alpha=0.6, zorder=2)
                                
                                # EPIC trendline (orange) - using pre-calculated spline
                                if epic_poly is not None and epic_mask.sum() >= 2:
                                    epic_x = x[epic_mask]
                                    epic_line_x = np.linspace(epic_x.min(), epic_x.max(), 100)
                                    epic_line_y = np.clip(epic_poly(epic_line_x), y_min, y_max)
                                    ax.plot(epic_line_x, epic_line_y, color='C1', linewidth=2.5, 
                                           linestyle='-', alpha=0.6, zorder=2)
                                
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
                        except Exception as e:
                            logging.debug(f"Failed to create detail genes plot: {e}")

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
                logging.debug(f"Failed to load cf detail file {detail_cf_path}: {e}")

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
            }
        )

        # Generate stacked pair plots (after EPIC genomeplot is created in phase 2)
        cfrrbs_subdirs = [f for f in os.listdir(cfrrbs_dir) if f.endswith(".plots")]
        if cfrrbs_subdirs:
            cfrrbs_plot_path = os.path.join(cfrrbs_dir, cfrrbs_subdirs[0], "genome_wide.png")
            epic_plot_path = os.path.join(epic_dir, "CNV_genomeplot.png")
            
            # For 2-tech: EPIC + cfRRBS
            stacked_path_2tech = os.path.join(sample_pair_dir, "genomewide_stacked_2tech.png")
            if os.path.exists(epic_plot_path) and os.path.exists(cfrrbs_plot_path):
                try:
                    _stack_pair_plots(
                        epic_plot_path,
                        cfrrbs_plot_path,
                        stacked_path_2tech,
                        title=f"EPIC vs cfRRBS - {pair_id}",
                    )
                except Exception as e:
                    logging.warning(f"Failed to create 2-tech stacked plot for {pair_id}: {e}")
            
            # Keep current "genomewide_stacked.png" as the 2-tech version for backward compatibility
            stacked_path = os.path.join(sample_pair_dir, "genomewide_stacked.png")
            if os.path.exists(stacked_path_2tech) and not os.path.exists(stacked_path):
                try:
                    import shutil
                    shutil.copy(stacked_path_2tech, stacked_path)
                except Exception as e:
                    logging.debug(f"Could not copy 2-tech plot to stacked.png: {e}")

    # ===================================================================
    # Final report generation
    # ===================================================================
    logging.info("Phase 4: Writing final reports")
    
    # Create sub-summary directory for processed pairs
    summary_2tech_dir = os.path.join(summary_dir, "2tech_only")
    if report_rows:
        os.makedirs(summary_2tech_dir, exist_ok=True)
    
    # Generate summary gene boxplot across all samples
    if all_gene_data:
        try:
            _plot_gene_boxplot_summary(all_gene_data, summary_dir)
            logging.info("Gene summary boxplot written to summary folder")
        except Exception as e:
            logging.warning(f"Failed to create gene summary boxplot: {e}")
    
    # Generate compact correlation summary visualization (all)
    if report_rows:
        try:
            _plot_correlation_summary(report_rows, summary_dir)
            logging.info("Correlation summary plot written to summary folder")
        except Exception as e:
            logging.warning(f"Failed to create correlation summary plot: {e}")
    
    # Generate correlation summary for all processed pairs
    if report_rows:
        try:
            _plot_correlation_summary(report_rows, summary_2tech_dir)
            logging.info("Correlation summary plot written to 2tech_only subfolder")
        except Exception as e:
            logging.warning(f"Failed to create correlation summary plot: {e}")
    
    # Generate gene reproducibility report (cfRRBS vs EPIC)
    if all_cfrrbs_gene_calls or all_epic_gene_calls:
        try:
            _generate_gene_reproducibility_report(all_cfrrbs_gene_calls, all_epic_gene_calls, summary_dir)
            logging.info("Gene reproducibility report written to summary folder")
        except Exception as e:
            logging.warning(f"Failed to create gene reproducibility report: {e}")
    
    logging.info(f"EPIC vs cfRRBS pipeline completed successfully ({len(report_rows)} pairs)")

