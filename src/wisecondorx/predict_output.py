# WisecondorX

import os
import re
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


def _generate_gene_calls_and_plots(rem_input, results):
    """Generate amplified/deleted gene TSVs and detail plots using gene bin values.

    Uses the regions file (chr,start,end,name) to extract bin values for each gene.
    Calls gains/losses based on the selected method:
    - "conumee": uses beta/zscore thresholds (default)
    - "segment-wise": checks if gene overlaps a deviant segment
    
    Always generates plots showing all genes with colors based on call status.
    """
    regions_path = rem_input["args"].regions
    if regions_path is None or not os.path.exists(regions_path):
        return

    import math as _math

    # Load regions (genes)
    regions = []
    with open(regions_path, "r") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            chr_name = parts[0].replace("chr", "")
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                continue
            name = parts[3]
            regions.append((chr_name, start, end, name))

    if not regions:
        return

    outdir = os.path.abspath(rem_input["args"].outid + ".plots")
    os.makedirs(outdir, exist_ok=True)

    # Helper function to get pval from zscore
    def _get_pval_from_z(z):
        try:
            zf = float(z)
        except Exception:
            return _math.nan
        if not _math.isfinite(zf):
            return _math.nan
        return math.erfc(abs(zf) / math.sqrt(2.0))

    # Extract gene bin values and determine calls
    all_genes = []
    for chr_name, start, end, name in regions:
        # Determine chr index
        chr_idx = None
        if chr_name in ["X", "chrX"]:
            chr_idx = 21
        elif chr_name in ["Y", "chrY"]:
            chr_idx = 22
        else:
            try:
                chr_idx = int(re.sub("chr", "", chr_name)) - 1
            except Exception:
                continue
        if chr_idx < 0 or chr_idx >= len(results["results_r"]):
            continue

        binsize = rem_input["binsize"]
        start_bin = max(0, start // binsize)
        end_bin = max(0, (end - 1) // binsize)
        if end_bin >= rem_input.get("bins_per_chr", [])[chr_idx]:
            end_bin = rem_input.get("bins_per_chr", [])[chr_idx] - 1
        if start_bin > end_bin:
            continue

        # Extract bin values for this gene
        region_ratios = np.array(results["results_r"][chr_idx][start_bin : end_bin + 1], dtype=float)
        region_weights = np.array(results["results_w"][chr_idx][start_bin : end_bin + 1], dtype=float)
        region_z = np.array(results.get("results_z", [])[chr_idx][start_bin : end_bin + 1], dtype=float)

        # Compute weighted mean values for the gene
        try:
            ratio_mean = float(np.ma.average(region_ratios, weights=region_weights))
        except Exception:
            ratio_mean = float("nan")
        try:
            z_mean = float(np.ma.average(region_z, weights=region_weights))
        except Exception:
            z_mean = float("nan")

        pval = _get_pval_from_z(z_mean)

        all_genes.append({
            "name": name,
            "chr": chr_name,
            "start": start,
            "end": end,
            "ratio": ratio_mean,
            "zscore": z_mean,
            "pval": pval,
        })

    if not all_genes:
        return

    # Determine calling method
    gene_call_method = getattr(rem_input["args"], "gene_call_method", None) or "conumee"

    # Build segment-based index for segment-wise method
    deviant_segments_by_chr = {}
    if gene_call_method == "segment-wise":
        gain_thr = rem_input["args"].gene_call_thr_gain
        loss_thr = rem_input["args"].gene_call_thr_loss
        if gain_thr is not None or loss_thr is not None:
            for segment in results["results_c"]:
                chr_idx = int(segment[0])
                chr_name = str(chr_idx + 1)
                if chr_name == "23":
                    chr_name = "X"
                if chr_name == "24":
                    chr_name = "Y"
                ratio = float(segment[4])
                
                is_deviant = False
                dev_type = None
                if gain_thr is not None and ratio >= gain_thr:
                    is_deviant = True
                    dev_type = "gain"
                elif loss_thr is not None and ratio <= loss_thr:
                    is_deviant = True
                    dev_type = "loss"
                
                if is_deviant:
                    start_pos = int(segment[1] * rem_input["binsize"] + 1)
                    end_pos = int(segment[2] * rem_input["binsize"])
                    if chr_name not in deviant_segments_by_chr:
                        deviant_segments_by_chr[chr_name] = []
                    deviant_segments_by_chr[chr_name].append((start_pos, end_pos, dev_type))

    # Apply calling logic based on method
    amp_rows = []
    del_rows = []

    for gene in all_genes:
        call = "neutral"
        call_source = ""

        if gene_call_method == "segment-wise":
            # Check if gene overlaps a deviant segment
            chr_name = gene["chr"]
            if chr_name in deviant_segments_by_chr:
                for seg_start, seg_end, dev_type in deviant_segments_by_chr[chr_name]:
                    if gene["end"] >= seg_start and gene["start"] <= seg_end:
                        if dev_type == "gain":
                            call = "gain"
                            call_source = "segment-wise"
                            break
                        elif dev_type == "loss":
                            call = "deletion"
                            call_source = "segment-wise"
                            break
        else:
            # Conumee method: use beta/zscore thresholds
            chr_name = gene["chr"].replace("chr", "")
            ploidy = 2
            if chr_name in ["X", "Y"] and rem_input["ref_gender"] == "M":
                ploidy = 1

            if rem_input["args"].beta is not None:
                loss_cutoff, gain_cutoff = __get_aberration_cutoff(rem_input["args"].beta, ploidy)
                if gene["ratio"] is not None and not np.isnan(gene["ratio"]):
                    if gene["ratio"] > gain_cutoff:
                        call = "gain"
                        call_source = "beta_ratio"
                    elif gene["ratio"] < loss_cutoff:
                        call = "deletion"
                        call_source = "beta_ratio"
            else:
                # Use zscore threshold primarily
                try:
                    if not np.isnan(gene["zscore"]) and gene["zscore"] > rem_input["args"].zscore:
                        call = "gain"
                        call_source = "zscore"
                    elif not np.isnan(gene["zscore"]) and gene["zscore"] < -rem_input["args"].zscore:
                        call = "deletion"
                        call_source = "zscore"
                    else:
                        # fallback to ratio magnitude
                        if not np.isnan(gene["ratio"]) and abs(gene["ratio"]) >= 0.3:
                            call = "gain" if gene["ratio"] > 0 else "deletion"
                            call_source = "ratio_fallback"
                except Exception:
                    pass

        # Update gene with call status
        gene["call"] = call
        gene["call_source"] = call_source

        outrow = {
            "name": gene["name"],
            "chr": gene["chr"],
            "start": gene["start"],
            "end": gene["end"],
            "ratio": gene["ratio"],
            "zscore": gene["zscore"],
            "call": call,
            "call_source": call_source,
            "pval": gene["pval"],
        }

        if call == "gain":
            amp_rows.append(outrow)
        elif call == "deletion":
            del_rows.append(outrow)

    # Write TSVs
    def _write_tsv(path, rows_list):
        with open(path, "w") as fo:
            fo.write("name\tchr\tstart\tend\tratio\tzscore\tcall\tcall_source\tpval\n")
            for r in rows_list:
                pval_str = "{:.6e}".format(r["pval"]) if _math.isfinite(r["pval"]) else "nan"
                row_copy = r.copy()
                del row_copy["pval"]  # remove pval from dict to avoid conflict with explicit pval parameter
                fo.write("{name}\t{chr}\t{start}\t{end}\t{ratio:.4f}\t{zscore:.4f}\t{call}\t{call_source}\t{pval}\n".format(
                    pval=pval_str, **row_copy
                ))

    if amp_rows:
        amp_path = "{}_amplified_genes.tsv".format(rem_input["args"].outid)
        _write_tsv(amp_path, amp_rows)
    if del_rows:
        del_path = "{}_deleted_genes.tsv".format(rem_input["args"].outid)
        _write_tsv(del_path, del_rows)

    # Create plots: always generate regardless of method
    try:
        genes_df = all_genes
        # detail genes barplot
        fig, ax = plt.subplots(figsize=(max(8, len(genes_df) * 0.35), 6))
        names = [g["name"] for g in genes_df]
        ratios = [0.0 if (g["ratio"] is None or np.isnan(g["ratio"])) else g["ratio"] for g in genes_df]
        
        # Build call->color mapping for each gene
        call_colors = {"gain": "#27ae60", "deletion": "#c0392b", "neutral": "#bdc3c7"}
        colors = [call_colors.get(g["call"], "#bdc3c7") for g in genes_df]
        
        x = range(len(names))
        ax.bar(x, ratios, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("log2 ratio")
        ax.set_ylim(-1.25, 1.25)
        ax.set_title("Detail Genes: {}".format(os.path.basename(rem_input["args"].outid)))
        plt.tight_layout()
        fig_path = os.path.join(outdir, "ratio_genes_scatter.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # gene aberrations scatter (ratio vs index with color)
        fig, ax = plt.subplots(figsize=(max(8, len(genes_df) * 0.35), 4))
        ax.scatter(x, ratios, c=colors, s=50, edgecolor='black')
        ax.axhline(0, color='grey', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("log2 ratio")
        ax.set_ylim(-1.25, 1.25)
        ax.set_title("Gene Aberrations: {}".format(os.path.basename(rem_input["args"].outid)))
        plt.tight_layout()
        fig_path2 = os.path.join(outdir, "ratio_genes_bar.png")
        fig.savefig(fig_path2, dpi=150, bbox_inches="tight")
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
