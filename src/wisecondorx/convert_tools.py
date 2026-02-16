# WisecondorX

import csv
import logging

import numpy as np
import pysam
import sys
import os

from wisecondorx.overall_tools import exec_R

"""
Converts aligned reads file to numpy array by transforming
individual reads to counts per bin.
"""


def convert_reads(args):
    bins_per_chr = dict()
    for chr in range(1, 25):
        bins_per_chr[str(chr)] = None

    logging.info("Importing data ...")

    if args.infile.endswith(".bam"):
        reads_file = pysam.AlignmentFile(args.infile, "rb")
    elif args.infile.endswith(".cram"):
        if args.reference is not None:
            reads_file = pysam.AlignmentFile(
                args.infile, "rc", reference_filename=args.reference
            )
        else:
            logging.error(
                "Cram support requires a reference file, please use the --reference argument"
            )
            sys.exit(1)
    else:
        logging.error(
            "Unsupported input file type. Make sure your input filename has a correct extension ( bam or cram)"
        )
        sys.exit(1)

    reads_seen = 0
    reads_kept = 0
    reads_mapq = 0
    reads_rmdup = 0
    reads_pairf = 0
    larp = -1
    larp2 = -1

    logging.info("Converting aligned reads ... This might take a while ...")

    for index, chr in enumerate(reads_file.references):

        chr_name = chr
        if chr_name[:3].lower() == "chr":
            chr_name = chr_name[3:]
        if chr_name not in bins_per_chr and chr_name != "X" and chr_name != "Y":
            continue

        logging.info(
            "Working at {}; processing {} bins".format(
                chr, int(reads_file.lengths[index] / float(args.binsize) + 1)
            )
        )
        counts = np.zeros(
            int(reads_file.lengths[index] / float(args.binsize) + 1), dtype=np.int32
        )
        bam_chr = reads_file.fetch(chr)

        if chr_name == "X":
            chr_name = "23"
        if chr_name == "Y":
            chr_name = "24"

        for read in bam_chr:
            if read.is_paired:
                if not read.is_proper_pair:
                    reads_pairf += 1
                    continue
                if (
                    not args.normdup
                    and larp == read.pos
                    and larp2 == read.next_reference_start
                ):
                    reads_rmdup += 1
                else:
                    if read.mapping_quality >= 1:
                        location = read.pos / args.binsize
                        counts[int(location)] += 1
                    else:
                        reads_mapq += 1

                larp2 = read.next_reference_start
                reads_seen += 1
                larp = read.pos
            else:
                if not args.normdup and larp == read.pos:
                    reads_rmdup += 1
                else:
                    if read.mapping_quality >= 1:
                        location = read.pos / args.binsize
                        counts[int(location)] += 1
                    else:
                        reads_mapq += 1

                reads_seen += 1
                larp = read.pos

        bins_per_chr[chr_name] = counts
        reads_kept += sum(counts)

    qual_info = {
        "mapped": reads_file.mapped,
        "unmapped": reads_file.unmapped,
        "no_coordinate": reads_file.nocoordinate,
        "filter_rmdup": reads_rmdup,
        "filter_mapq": reads_mapq,
        "pre_retro": reads_seen,
        "post_retro": reads_kept,
        "pair_fail": reads_pairf,
    }
    return bins_per_chr, qual_info


def convert_idat(args):
    infile = args.infile
    idat_basenames = []
    out_bins = None
    out_segments = None
    out_dir = None

    if os.path.isdir(infile):
        out_dir = args.outfile
        for root, _, files in os.walk(infile):
            for name in files:
                if name.endswith("_Grn.idat"):
                    idat_basenames.append(os.path.join(root, name[:-9]))
        if not idat_basenames:
            for root, _, files in os.walk(infile):
                for name in files:
                    if name.endswith("_Red.idat"):
                        idat_basenames.append(os.path.join(root, name[:-9]))
    else:
        basename = infile
        if infile.endswith("_Grn.idat"):
            basename = infile[:-9]
        elif infile.endswith("_Red.idat"):
            basename = infile[:-9]
        elif infile.endswith(".idat"):
            basename = infile[:-5]
        idat_basenames = [basename]

        out_bins = args.outfile
        out_segments = args.epic_segments_out
        if not out_segments:
            out_segments = "{}.segments.bed".format(out_bins)

    json_plot = "{}.epic_convert.json".format(args.outfile)
    json_dict = {
        "R_script": str("{}/include/epic_convert.R".format(args.wd)),
        "infile": str(json_plot),
        "idat_basenames": idat_basenames,
        "binsize": int(args.binsize),
        "out_bins": str(out_bins) if out_bins else "",
        "out_segments": str(out_segments) if out_segments else "",
        "out_dir": str(out_dir) if out_dir else "",
        "conumee_anno": str(args.conumee_anno),
        "conumee_ref_m": str(args.conumee_ref_m),
        "conumee_ref_f": str(args.conumee_ref_f),
        "epic_gender": str(args.epic_gender) if args.epic_gender else "",
    }

    exec_R(json_dict)

    def write_epic_npz(bins_path: str, npz_path: str) -> None:
        rows = []
        with open(bins_path, "r", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                rows.append(row)

        if not rows:
            return

        chr_vals = []
        start_vals = []
        end_vals = []
        ratio_vals = []
        n_probes_vals = []
        for row in rows:
            chr_vals.append(str(row.get("chr", "")))
            start_vals.append(int(float(row.get("bin_start", 0))))
            end_vals.append(int(float(row.get("bin_end", 0))))
            ratio_vals.append(float(row.get("ratio", "nan")))
            n_probes_vals.append(int(float(row.get("n_probes", 0))))

        np.savez_compressed(
            npz_path,
            chr=np.array(chr_vals, dtype=object),
            start=np.array(start_vals, dtype=np.int64),
            end=np.array(end_vals, dtype=np.int64),
            ratio=np.array(ratio_vals, dtype=np.float64),
            n_probes=np.array(n_probes_vals, dtype=np.int64),
        )

    if out_bins:
        npz_path = os.path.splitext(out_bins)[0] + ".npz"
        if os.path.exists(out_bins):
            write_epic_npz(out_bins, npz_path)
    elif out_dir:
        for idat_basename in idat_basenames:
            sample_id = os.path.basename(idat_basename)
            bins_path = os.path.join(out_dir, "{}.tsv".format(sample_id))
            npz_path = os.path.join(out_dir, "{}.npz".format(sample_id))
            if os.path.exists(bins_path):
                write_epic_npz(bins_path, npz_path)
