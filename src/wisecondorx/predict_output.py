# WisecondorX

import os
import re
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wisecondorx.overall_tools import (
    exec_R,
    get_z_score,
    get_median_segment_variance,
    get_cpa,
)

"""
Writes plots.
"""


def exec_write_plots(rem_input, results, conumee=False):
    json_plot_dir = os.path.abspath(rem_input["args"].outid + "_plot_tmp")

    # Choose R script depending on conumee flag
    r_script_file = "plotter_conumee.R" if conumee else "plotter.R"

    json_dict = {
        "R_script": str("{}/include/{}".format(rem_input["wd"], r_script_file)),
        "ref_gender": str(rem_input["ref_gender"]),
        "beta": str(rem_input["args"].beta),
        "zscore": str(rem_input["args"].zscore),
        "binsize": str(rem_input["binsize"]),
        "n_reads": str(rem_input["n_reads"]),
        "cairo": str(rem_input["args"].cairo),
        "results_r": results["results_r"],
        "results_w": results["results_w"],
        "results_c": results["results_c"],
        "results_variance": results["results_variance"],
        "ref_sizes": rem_input["ref_sizes"].tolist(),
        "min_coverage_refsize": str(rem_input["args"].min_coverage_refsize if rem_input["args"].min_coverage_refsize else "NULL"),
        "min_confidence_score": str(rem_input["args"].min_confidence_score if rem_input["args"].min_confidence_score else "NULL"),
        "plot_loci_bed": str(rem_input["args"].plot_loci_bed if rem_input["args"].plot_loci_bed else "NULL"),
        "ylim": str(rem_input["args"].ylim),
        "regions": str(rem_input["args"].regions),
        "gene_call_method": str(getattr(rem_input["args"], "gene_call_method", "conumee") or "conumee"),
        "gene_call_thr_gain": str(rem_input["args"].gene_call_thr_gain if rem_input["args"].gene_call_thr_gain is not None else "NULL"),
        "gene_call_thr_loss": str(rem_input["args"].gene_call_thr_loss if rem_input["args"].gene_call_thr_loss is not None else "NULL"),
        "infile": str("{}.json".format(json_plot_dir)),
        "out_dir": str("{}.plots".format(rem_input["args"].outid)),
    }

    # Attempt to detect a centromere file: prefer a dedicated arg, else use blacklist if it looks like a centromere file
    centromeres_path = None
    try:
        if hasattr(rem_input["args"], "centromeres") and rem_input["args"].centromeres:
            centromeres_path = rem_input["args"].centromeres
        elif hasattr(rem_input["args"], "blacklist") and rem_input["args"].blacklist:
            bl = os.path.basename(rem_input["args"].blacklist).lower()
            if "centrom" in bl:
                centromeres_path = rem_input["args"].blacklist
    except Exception:
        centromeres_path = None

    json_dict["centromeres"] = str(centromeres_path if centromeres_path else "NULL")

    if rem_input["args"].add_plot_title:
        json_dict["plot_title"] = str(os.path.basename(rem_input["args"].outid))

    # Persist a copy so the plotter can be re-run after gene calling (focal/broad coloring).
    try:
        import json as _json
        plots_dir = "{}.plots".format(rem_input["args"].outid)
        os.makedirs(plots_dir, exist_ok=True)
        _json.dump(json_dict, open(os.path.join(plots_dir, "plot_input.json"), "w"))
    except Exception:
        pass

    exec_R(json_dict)


"""
Calculates zz-scores, marks aberrations and
writes tables.
"""


def generate_output_tables(rem_input, results):
    _generate_bins_bed(rem_input, results)
    _generate_segments_and_aberrations_bed(rem_input, results)
    _generate_chr_statistics_file(rem_input, results)
    
    if rem_input["args"].regions is not None:
        _generate_regions_bed(rem_input, results)
        # Generate gene calls and plots based on selected method
        try:
            _generate_gene_calls_and_plots(rem_input, results)
        except Exception:
            pass


def generate_plot_bins_stats(rem_input, results):
    out_path = "{}_plot_bins_stats.tsv".format(rem_input["args"].outid)
    binsize = rem_input["binsize"]
    min_refsize = rem_input["args"].min_coverage_refsize
    min_confidence = rem_input["args"].min_confidence_score
    loci_bed = rem_input["args"].plot_loci_bed
    sample_counts = rem_input.get("sample_counts", None)

    results_r = results["results_r"]
    results_z = results["results_z"]
    results_w = results["results_w"]
    results_v = results["results_variance"]
    results_rs = results["results_refsizes"]

    # Build loci mask if requested (autosomes only, like plotter)
    loci_mask = None
    if loci_bed is not None and os.path.exists(loci_bed):
        loci_mask = []
        for chr_idx in range(22):
            loci_mask.append([False] * len(results_r[chr_idx]))
        try:
            with open(loci_bed, "r") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue
                    chr_name = parts[0].replace("chr", "")
                    if chr_name in ["X", "Y"]:
                        continue
                    try:
                        chr_idx = int(chr_name) - 1
                        if chr_idx < 0 or chr_idx >= 22:
                            continue
                        start = int(parts[1])
                        end = int(parts[2])
                    except ValueError:
                        continue
                    bin_start = max(0, start // binsize)
                    bin_end = max(0, (end - 1) // binsize)
                    bin_end = min(bin_end, len(results_r[chr_idx]) - 1)
                    if bin_start > bin_end:
                        continue
                    for b in range(bin_start, bin_end + 1):
                        loci_mask[chr_idx][b] = True
        except Exception:
            loci_mask = None

    with open(out_path, "w") as out:
        out.write(
            "chr\tstart\tend\treads\tratio\tzscore\tweight\tvariance\trefsize\tconfidence\tplotted\n"
        )
        for chr_idx in range(22):
            chr_name = str(chr_idx + 1)
            feat = 1
            for i in range(len(results_r[chr_idx])):
                r = results_r[chr_idx][i]
                z = results_z[chr_idx][i]
                w = results_w[chr_idx][i]
                v = results_v[chr_idx][i]
                rs = results_rs[chr_idx][i]

                # Get original read count from sample
                reads = 0
                if sample_counts is not None and chr_name in sample_counts:
                    if i < len(sample_counts[chr_name]):
                        reads = int(sample_counts[chr_name][i])

                ratio_val = float("nan") if r == 0 else r
                plotted = True

                if r == 0 or np.isnan(ratio_val):
                    plotted = False

                if min_refsize is not None and rs < min_refsize:
                    plotted = False

                confidence = 0
                if v and v > 0:
                    confidence = rs / v
                if min_confidence is not None and confidence < min_confidence:
                    plotted = False

                if loci_mask is not None and not loci_mask[chr_idx][i]:
                    plotted = False

                out.write(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                        chr_name,
                        feat,
                        feat + binsize - 1,
                        reads,
                        ratio_val,
                        "nan" if z == 0 else z,
                        "nan" if w == 0 else w,
                        "nan" if v == 0 else v,
                        "nan" if rs == 0 else rs,
                        confidence,
                        1 if plotted else 0,
                    )
                )
                feat += binsize

def _generate_bins_bed(rem_input, results):
    bins_file = open("{}_bins.bed".format(rem_input["args"].outid), "w")
    bins_file.write("chr\tstart\tend\tid\tratio\tzscore\n")
    results_r = results["results_r"]
    results_z = results["results_z"]
    binsize = rem_input["binsize"]

    for chr in range(len(results_r)):
        chr_name = str(chr + 1)
        if chr_name == "23":
            chr_name = "X"
        if chr_name == "24":
            chr_name = "Y"
        feat = 1
        for i in range(len(results_r[chr])):
            r = results_r[chr][i]
            z = results_z[chr][i]
            if r == 0:
                r = "nan"
            if z == 0:
                z = "nan"
            feat_str = "{}:{}-{}".format(chr_name, str(feat), str(feat + binsize - 1))
            row = [chr_name, feat, feat + binsize - 1, feat_str, r, z]
            bins_file.write("{}\n".format("\t".join([str(x) for x in row])))
            feat += binsize
    bins_file.close()


def _generate_regions_bed(rem_input, results):
    regions_file = open("{}_regions.bed".format(rem_input["args"].outid), "w")
    regions_file.write("chr\tstart\tend\tname\tratio\tzscore\n")

    with open(rem_input["args"].regions, "r") as regions_file_handle:
        regions = [line.strip().split("\t") for line in regions_file_handle if line.strip() != ""]

        for region in regions:
            assert len(region) >= 4, "Regions file must have at least 4 columns: chr, start, end, name"
            chr_name, start, end, name  = region[0], region[1], region[2], region[3]

            # Convert chromosome name to zero-based index
            if chr_name == "chrX" or chr_name == "X":
                chr = 21
            if chr_name == "chrY" or chr_name == "Y":
                chr = 22
            chr = int(re.sub("chr", "", chr_name)) - 1
            start_bin = int(start) // rem_input["binsize"]
            end_bin = int(end) // rem_input["binsize"]
            if end_bin >= rem_input["bins_per_chr"][chr]:
                end_bin = rem_input["bins_per_chr"][chr] - 1


            if start_bin < 0 or end_bin < 0 or start_bin > end_bin:
                regions_file.write("Skipping invalid region: {}\n".format("\t".join(region)))
                continue
            
            # Extract ratios, weights, and z-scores for the region
            region_ratios = results["results_r"][chr][start_bin : end_bin + 1]
            region_weights = results["results_w"][chr][start_bin : end_bin + 1]
            region_zscores = results["results_z"][chr][start_bin : end_bin + 1]

            if len(region_ratios) == 0:
                regions_file.write("Skipping region with no bins: {}\n".format("\t".join(region)))
                continue
            
            # Calculate weighted means
            ratio_mean = np.ma.average(region_ratios, weights=region_weights)
            zscore_mean = np.ma.average(region_zscores, weights=region_weights)
            
            if ratio_mean == 0:
                ratio_mean = "nan"
            if zscore_mean == 0:
                zscore_mean = "nan"

            row = [chr_name, start, end, name, ratio_mean, zscore_mean]
            regions_file.write("{}\n".format("\t".join([str(x) for x in row])))

    regions_file.close()

def _generate_segments_and_aberrations_bed(rem_input, results):
    segments_file = open("{}_segments.bed".format(rem_input["args"].outid), "w")
    aberrations_file = open("{}_aberrations.bed".format(rem_input["args"].outid), "w")
    segments_file.write("chr\tstart\tend\tratio\tzscore\n")
    aberrations_file.write("chr\tstart\tend\tratio\tzscore\ttype\tgenes\tpval\n")

    regions_by_chr = {}
    regions_path = rem_input["args"].regions
    if regions_path is not None and os.path.exists(regions_path):
        try:
            with open(regions_path, "r") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 4:
                        continue
                    chr_name = parts[0].replace("chr", "")
                    try:
                        start = int(parts[1])
                        end = int(parts[2])
                    except ValueError:
                        continue
                    gene = parts[3]
                    if chr_name in ["X", "Y"]:
                        chr_key = chr_name
                    else:
                        try:
                            chr_key = str(int(chr_name))
                        except ValueError:
                            continue
                    if chr_key not in regions_by_chr:
                        regions_by_chr[chr_key] = []
                    regions_by_chr[chr_key].append((start, end, gene))
        except Exception:
            regions_by_chr = {}

    def get_overlapping_genes(chr_name, seg_start, seg_end):
        if chr_name not in regions_by_chr:
            return "."
        genes = []
        for start, end, gene in regions_by_chr[chr_name]:
            if end >= seg_start and start <= seg_end:
                genes.append(gene)
        if not genes:
            return "."
        return ",".join(sorted(set(genes)))

    def get_z_pval(zscore):
        try:
            z = float(zscore)
        except (TypeError, ValueError):
            return "."
        if not math.isfinite(z):
            return "."
        return math.erfc(abs(z) / math.sqrt(2.0))

    for segment in results["results_c"]:
        chr_name = str(segment[0] + 1)
        if chr_name == "23":
            chr_name = "X"
        if chr_name == "24":
            chr_name = "Y"
        row = [
            chr_name,
            int(segment[1] * rem_input["binsize"] + 1),
            int(segment[2] * rem_input["binsize"]),
            segment[4],
            segment[3],
        ]
        segments_file.write("{}\n".format("\t".join([str(x) for x in row])))

        ploidy = 2
        if (chr_name == "X" or chr_name == "Y") and rem_input["ref_gender"] == "M":
            ploidy = 1
        if rem_input["args"].beta is not None:
            if (
                float(segment[4])
                > __get_aberration_cutoff(rem_input["args"].beta, ploidy)[1]
            ):
                genes = get_overlapping_genes(row[0], row[1], row[2])
                pval = get_z_pval(row[4])
                aberrations_file.write(
                    "{}\tgain\t{}\t{}\n".format("\t".join([str(x) for x in row]), genes, pval)
                )
            elif (
                float(segment[4])
                < __get_aberration_cutoff(rem_input["args"].beta, ploidy)[0]
            ):
                genes = get_overlapping_genes(row[0], row[1], row[2])
                pval = get_z_pval(row[4])
                aberrations_file.write(
                    "{}\tloss\t{}\t{}\n".format("\t".join([str(x) for x in row]), genes, pval)
                )
        elif isinstance(segment[3], str):
            continue
        else:
            if float(segment[3]) > rem_input["args"].zscore:
                genes = get_overlapping_genes(row[0], row[1], row[2])
                pval = get_z_pval(row[4])
                aberrations_file.write(
                    "{}\tgain\t{}\t{}\n".format("\t".join([str(x) for x in row]), genes, pval)
                )
            elif float(segment[3]) < -rem_input["args"].zscore:
                genes = get_overlapping_genes(row[0], row[1], row[2])
                pval = get_z_pval(row[4])
                aberrations_file.write(
                    "{}\tloss\t{}\t{}\n".format("\t".join([str(x) for x in row]), genes, pval)
                )

    segments_file.close()
    aberrations_file.close()


def _build_conumee_state_thresholds(results, conf=0.99, outlier_thresh=0.8, min_state_bins=5):
    """
    K-means (k=3) on autosomal bin ratios → state-specific analytical normal thresholds.

    States are ordered by centroid: 1=deletion, 2=neutral, 3=gain.
    Returns (thresholds, centroids):
      thresholds – dict {1:(low,high), 2:(low,high), 3:(low,high)}
      centroids  – length-3 ndarray sorted ascending
    """
    from scipy.cluster.vq import kmeans2
    from scipy.stats import norm as _norm

    all_r, all_w = [], []
    for ci in range(22):
        r = np.array(results["results_r"][ci], dtype=float)
        w = np.array(results["results_w"][ci], dtype=float)
        valid = np.isfinite(r) & np.isfinite(w) & (w > 0) & (np.abs(r) <= outlier_thresh)
        all_r.extend(r[valid].tolist())
        all_w.extend(w[valid].tolist())

    r_arr = np.array(all_r, dtype=float)
    w_arr = np.array(all_w, dtype=float)

    zcrit = _norm.ppf(1 - (1 - conf) / 2)
    centroids = np.array([-0.3, 0.0, 0.3])

    global_mu = float(np.nanmean(r_arr)) if len(r_arr) else 0.0
    global_sigma = max(float(np.nanstd(r_arr, ddof=1)) if len(r_arr) > 2 else 0.1, 0.05)
    fallback_thr = (global_mu - zcrit * global_sigma, global_mu + zcrit * global_sigma)
    thresholds = {1: fallback_thr, 2: fallback_thr, 3: fallback_thr}

    if len(r_arr) < 15:
        return thresholds, centroids

    init_centers = np.array([
        np.percentile(r_arr, 15),
        np.percentile(r_arr, 50),
        np.percentile(r_arr, 85),
    ])
    try:
        raw_centroids, labels = kmeans2(r_arr, init_centers, iter=20, minit='matrix')
    except Exception:
        return thresholds, centroids

    order = np.argsort(raw_centroids)
    centroids = raw_centroids[order]
    label_remap = {int(old): i for i, old in enumerate(order)}
    labels_sorted = np.array([label_remap[int(l)] for l in labels])

    for i in range(3):
        state_id = i + 1
        mask = labels_sorted == i
        state_r = r_arr[mask]
        state_w = w_arr[mask]
        if len(state_r) < min_state_bins:
            thresholds[state_id] = fallback_thr
            continue
        mu = float(np.average(state_r, weights=state_w))
        var = float(np.average((state_r - mu) ** 2, weights=state_w))
        sigma = max(float(np.sqrt(var)), 0.03)
        thresholds[state_id] = (mu - zcrit * sigma, mu + zcrit * sigma)

    return thresholds, centroids


def _generate_gene_calls_and_plots(rem_input, results):
    from scipy.stats import norm as _norm

    regions_path = rem_input["args"].regions
    if regions_path is None or not os.path.exists(regions_path):
        return

    regions = []
    with open(regions_path, "r") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            chr_raw = parts[0].replace("chr", "")
            try:
                start, end = int(parts[1]), int(parts[2])
            except ValueError:
                continue
            regions.append((chr_raw, start, end, parts[3]))

    if not regions:
        return

    outdir  = os.path.abspath(rem_input["args"].outid + ".plots")
    os.makedirs(outdir, exist_ok=True)
    outid   = rem_input["args"].outid
    binsize = rem_input["binsize"]
    bins_per_chr = rem_input.get("bins_per_chr", [])

    conf        = 0.99
    noise_floor = 0.05
    zcrit       = _norm.ppf(1 - (1 - conf) / 2)

    def _chr_to_idx(c):
        if c in ("X", "chrX"): return 22
        if c in ("Y", "chrY"): return 23
        try: return int(re.sub("chr", "", c)) - 1
        except (ValueError, TypeError): return -1

    # Extract per-gene bin metrics
    all_genes = []
    for chr_raw, start, end, name in regions:
        chr_idx   = _chr_to_idx(chr_raw)
        if chr_idx < 0 or chr_idx >= len(results["results_r"]):
            continue
        s_bin = max(0, start // binsize)
        e_bin = max(0, (end - 1) // binsize)
        if chr_idx < len(bins_per_chr):
            e_bin = min(e_bin, bins_per_chr[chr_idx] - 1)
        if s_bin > e_bin:
            continue
        r_b = np.array(results["results_r"][chr_idx][s_bin:e_bin + 1], dtype=float)
        w_b = np.array(results["results_w"][chr_idx][s_bin:e_bin + 1], dtype=float)
        z_b = np.array(results.get("results_z", [[]] * 24)[chr_idx][s_bin:e_bin + 1], dtype=float)
        ok  = np.isfinite(r_b) & np.isfinite(w_b) & (w_b > 0)
        if not np.any(ok):
            continue
        all_genes.append({
            "name":   name,
            "chr":    chr_raw,
            "start":  start,
            "end":    end,
            "ratio":  float(np.average(r_b[ok], weights=w_b[ok])),
            "zscore": float(np.average(z_b[ok], weights=w_b[ok])),
        })

    if not all_genes:
        return

    # K-means neutral-state thresholds; neutral_sigma used as focal z-score denominator
    thresholds, _ = _build_conumee_state_thresholds(results, conf=conf)
    neutral_low, neutral_high = thresholds[2]
    neutral_sigma = max((neutral_high - neutral_low) / (2 * zcrit), noise_floor)

    # ── Tumor purity estimation (ML on copy number space) ─────────────────────
    try:
        from scipy.optimize import minimize as _minimize
        seg_ratios_all = [float(seg[4]) for seg in results["results_c"]]
        seg_zscores_all = []
        for seg in results["results_c"]:
            ci = int(seg[0])
            s_b, e_b = int(seg[1]), int(seg[2])
            r_b = np.array(results["results_r"][ci][s_b:e_b + 1], dtype=float)
            w_b = np.array(results["results_w"][ci][s_b:e_b + 1], dtype=float)
            z_b = np.array(results.get("results_z", [[]] * 24)[ci][s_b:e_b + 1], dtype=float)
            ok = np.isfinite(r_b) & np.isfinite(w_b) & (w_b > 0)
            seg_zscores_all.append(float(np.average(z_b[ok], weights=w_b[ok])) if np.any(ok) else 0.0)

        if len(seg_ratios_all) >= 3:
            abs_z_arr = np.abs(seg_zscores_all)
            q1, q3 = np.percentile(abs_z_arr, [25, 75])
            purity_spread = (q3 - q1) / (2 * _norm.ppf(0.75)) if (q3 - q1) > 0 else 0.1
            ploidy = 2.0
            purities = np.linspace(0.1, 0.99, 30)
            best_p, best_ll = 0.5, -np.inf
            for pur in purities:
                ll = 0.0
                for r in seg_ratios_all:
                    ll += max(-0.5 * ((r - np.log2((pur * cn + (1 - pur) * ploidy) / ploidy)) / purity_spread) ** 2
                              for cn in [0, 1, 2, 3, 4])
                if ll > best_ll:
                    best_ll, best_p = ll, pur

            def _neg_ll(p):
                if p < 0.05 or p > 0.99: return 1e10
                return -sum(max(-0.5 * ((r - np.log2((p * cn + (1 - p) * ploidy) / ploidy)) / purity_spread) ** 2
                                for cn in [0, 1, 2, 3, 4]) for r in seg_ratios_all)

            res = _minimize(_neg_ll, best_p, method="L-BFGS-B", bounds=[(0.05, 0.99)])
            final_purity = float(res.x[0]) if res.success else best_p
            purity_file = f"{outid}_purity_estimate.txt"
            with open(purity_file, "w") as fh:
                fh.write("Tumor Purity Estimate (maximum likelihood on copy number space)\n")
                fh.write(f"Purity: {final_purity:.3f}\n")
                fh.write(f"95% CI: ({max(0.0, final_purity - 0.1):.3f}-{min(1.0, final_purity + 0.1):.3f})\n")
                fh.write(f"Ploidy: {ploidy:.1f}\n")
                fh.write(f"Segments used: {len(seg_ratios_all)}\n")
    except Exception as _e:
        print(f"[WisecondorX] Purity estimation failed: {_e}", file=sys.stderr)

    # ── Gold-standard focal calling: segment-based, |log2|>=0.25 AND |z|>=2.5, <5Mb ─
    FOCAL_MAX_BP   = 5_000_000
    FOCAL_LOG2_THR = 0.25
    FOCAL_Z_THR    = 2.5

    def _seg_zscore(seg_ratio):
        return seg_ratio / neutral_sigma if neutral_sigma > 0 else 0.0

    # Pre-compute set of focal-significant segments
    focal_sig_segs = set()
    for seg in results["results_c"]:
        ci     = int(seg[0])
        ck     = str(ci + 1) if ci + 1 not in (23, 24) else ("X" if ci + 1 == 23 else "Y")
        s_bp   = int(seg[1]) * binsize
        e_bp   = int(seg[2]) * binsize + binsize - 1
        seg_r  = float(seg[4])
        seg_z  = _seg_zscore(seg_r)
        seg_sz = e_bp - s_bp + 1
        if (abs(seg_r) >= FOCAL_LOG2_THR and abs(seg_z) >= FOCAL_Z_THR and seg_sz <= FOCAL_MAX_BP):
            focal_sig_segs.add((ck, s_bp, e_bp, seg_r))

    focal_amp, focal_del = [], []
    cols_focal = ["gene", "chr", "start", "end", "ratio", "zscore",
                  "seg_ratio", "seg_zscore", "focal_call", "pval"]

    for gene in all_genes:
        ratio, zscore, ck = gene["ratio"], gene["zscore"], gene["chr"]

        base = {"gene": gene["name"], "chr": ck,
                "start": gene["start"], "end": gene["end"],
                "ratio": ratio, "zscore": zscore}
        gene["call"] = "neutral"
        if ratio > neutral_high:
            gene["call"] = "gain"
        elif ratio < neutral_low:
            gene["call"] = "deletion"

        # Gene called focal only if its overlapping CBS segment passes all thresholds
        hit = next(
            ((ck2, s, e, sr) for (ck2, s, e, sr) in focal_sig_segs
             if ck2 == ck and gene["end"] >= s and gene["start"] <= e),
            None,
        )
        if hit is None:
            continue
        _, _, _, seg_r = hit
        seg_z  = _seg_zscore(seg_r)
        fcall  = "gain" if seg_r > 0 else "deletion"
        fpval  = float(2 * (1 - _norm.cdf(abs(seg_z))))
        row = {**base, "seg_ratio": seg_r, "seg_zscore": seg_z,
               "focal_call": fcall, "pval": fpval}
        (focal_amp if fcall == "gain" else focal_del).append(row)

    for path, rows, cols in [
        (f"{outid}_focal_amplified_genes.tsv", focal_amp, cols_focal),
        (f"{outid}_focal_deleted_genes.tsv",   focal_del, cols_focal),
    ]:
        with open(path, "w") as fh:
            fh.write("\t".join(cols) + "\n")
            for row in rows:
                fh.write("\t".join(str(row[c]) for c in cols) + "\n")

    # Focal segments TSV
    cols_seg = ["chr", "start", "end", "ratio", "call", "zscore", "size_bp"]
    with open(f"{outid}_focal_segments.tsv", "w") as fh:
        fh.write("\t".join(cols_seg) + "\n")
        for ck, s_bp, e_bp, seg_r in sorted(focal_sig_segs, key=lambda x: (x[0], x[1])):
            seg_z = _seg_zscore(seg_r)
            fh.write("\t".join(str(v) for v in [
                ck, s_bp, e_bp, round(seg_r, 4),
                "gain" if seg_r > 0 else "deletion",
                round(seg_z, 4), e_bp - s_bp + 1,
            ]) + "\n")

    try:
        focal_gene_names = {row["gene"] for row in focal_amp + focal_del}
        def _gene_color(g):
            return "#8e44ad" if g["name"] in focal_gene_names else "lightgrey"

        import matplotlib.patches as _mpatches
        x = list(range(len(all_genes)))
        fig, ax = plt.subplots(figsize=(max(8, len(all_genes) * 0.35), 4))
        ax.scatter(x, [g["ratio"] for g in all_genes],
                   c=[_gene_color(g) for g in all_genes],
                   s=50, edgecolor="black", linewidths=0.5)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([g["name"] for g in all_genes], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("log2 ratio")
        ax.set_title("Gene Aberrations: {}".format(os.path.basename(outid)))
        legend_handles = [
            _mpatches.Patch(color="#8e44ad", label="Focal altered"),
            _mpatches.Patch(color="lightgrey", label="Neutral"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7.5, framealpha=0.85)
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, "ratio_genes_bar.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass


def __get_aberration_cutoff(beta, ploidy):
    loss_cutoff = np.log2((ploidy - (beta / 2)) / ploidy)
    gain_cutoff = np.log2((ploidy + (beta / 2)) / ploidy)
    return loss_cutoff, gain_cutoff


def _generate_chr_statistics_file(rem_input, results):
    stats_file = open("{}_statistics.txt".format(rem_input["args"].outid), "w")
    stats_file.write("chr\tratio.mean\tratio.median\tzscore\n")
    chr_ratio_means = [
        np.ma.average(results["results_r"][chr], weights=results["results_w"][chr])
        for chr in range(len(results["results_r"]))
    ]
    chr_ratio_medians = [
        np.median([x for x in results["results_r"][chr] if x != 0])
        for chr in range(len(results["results_r"]))
    ]

    results_c_chr = [
        [x, 0, rem_input["bins_per_chr"][x] - 1, chr_ratio_means[x]]
        for x in range(len(results["results_r"]))
    ]

    msv = round(
        get_median_segment_variance(results["results_c"], results["results_r"]), 5
    )
    cpa = round(get_cpa(results["results_c"], rem_input["binsize"]), 5)
    chr_z_scores = get_z_score(results_c_chr, results)

    for chr in range(len(results["results_r"])):

        chr_name = str(chr + 1)
        if chr_name == "23":
            chr_name = "X"
        if chr_name == "24":
            chr_name = "Y"

        row = [
            chr_name,
            chr_ratio_means[chr],
            chr_ratio_medians[chr],
            chr_z_scores[chr],
        ]

        stats_file.write("\t".join([str(x) for x in row]) + "\n")

    stats_file.write(
        "Gender based on --yfrac (or manually overridden by --gender): {}\n".format(
            str(rem_input["gender"])
        )
    )

    stats_file.write("Number of reads: {}\n".format(str(rem_input["n_reads"])))

    stats_file.write(
        "Standard deviation of the ratios per chromosome: {}\n".format(
            str(round(float(np.nanstd(chr_ratio_means)), 5))
        )
    )

    stats_file.write(
        "Median segment variance per bin (doi: 10.1093/nar/gky1263): {}\n".format(
            str(msv)
        )
    )

    stats_file.write(
        "Copy number profile abnormality (CPA) score (doi: 10.1186/s13073-020-00735-4): {}\n".format(
            str(cpa)
        )
    )

    stats_file.close()
