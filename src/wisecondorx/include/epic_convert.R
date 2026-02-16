options(warn=1)

suppressMessages(library("jsonlite"))
suppressMessages(library("minfi"))
suppressMessages(library("conumee"))

args <- commandArgs(T)
in.file <- paste0(args[which(args == "--infile") + 1])
input <- read_json(in.file, na="string")

idat_basenames <- unlist(input$idat_basenames)
binsize <- as.integer(input$binsize)
out_bins <- input$out_bins
out_segments <- input$out_segments
out_dir <- input$out_dir

conumee_anno <- input$conumee_anno
conumee_ref_m <- input$conumee_ref_m
conumee_ref_f <- input$conumee_ref_f

is_missing_path <- function(x) {
  is.null(x) || !nzchar(x) || x %in% c("NULL", "None", "NA")
}

has_external <- !is_missing_path(conumee_anno) && !is_missing_path(conumee_ref_m) &&
  !is_missing_path(conumee_ref_f) && file.exists(conumee_anno) &&
  file.exists(conumee_ref_m) && file.exists(conumee_ref_f)

if (has_external) {
  load(conumee_anno)
  load(conumee_ref_m)
  load(conumee_ref_f)
} else {
  message("Conumee reference files not provided or not found. Attempting to load packaged conumeeData.")
  suppressMessages(library("conumeeData"))
  tryCatch({
    data("anno", package = "conumeeData", envir = environment())
  }, error = function(e) {})
  tryCatch({
    data("ref", package = "conumeeData", envir = environment())
  }, error = function(e) {})
  tryCatch({
    data("annoXY", package = "conumeeData", envir = environment())
  }, error = function(e) {})
  tryCatch({
    data("refM.data", package = "conumeeData", envir = environment())
  }, error = function(e) {})
  tryCatch({
    data("refF.data", package = "conumeeData", envir = environment())
  }, error = function(e) {})

  if (!exists("annoXY") && exists("anno")) {
    annoXY <- anno
  }
  if (!exists("refM.data") && exists("ref")) {
    refM.data <- ref
  }
  if (!exists("refF.data") && exists("ref")) {
    refF.data <- ref
  }
}

if (!exists("annoXY")) {
  stop("annoXY not available. Provide conumee annotation RData or install conumeeData with EPIC annotation.")
}
if (!exists("refM.data")) {
  stop("refM.data not available. Provide conumee male reference RData or install conumeeData.")
}
if (!exists("refF.data")) {
  stop("refF.data not available. Provide conumee female reference RData or install conumeeData.")
}

if (length(idat_basenames) == 0) {
  stop("No IDAT basenames provided")
}

get_detail_df <- function(x) {
  if (!is.null(x@detail)) return(as.data.frame(x@detail))
  if (!is.null(x$detail)) return(as.data.frame(x$detail))
  return(NULL)
}

get_segments_df <- function(x) {
  if (!is.null(x@segments)) return(as.data.frame(x@segments))
  if (!is.null(x$segments)) return(as.data.frame(x$segments))
  return(NULL)
}

process_sample <- function(idat_basename, bins_out, segments_out) {
  rgset <- read.metharray(basenames=idat_basename, extended=TRUE)
  mset <- preprocessRaw(rgset)

  cndata <- CNV.load(mset)

  ref_data <- refM.data
  if (!is.null(input$epic_gender) && toupper(as.character(input$epic_gender)) == "F") {
    ref_data <- refF.data
  }

  x <- CNV.fit(cndata, ref_data, annoXY)
  x <- CNV.bin(x)
  x <- CNV.detail(x)
  x <- CNV.segment(x)

  detail_df <- get_detail_df(x)
  if (is.null(detail_df)) {
    stop("No detail data available for IDAT")
  }

chr_col <- intersect(c("chr", "chromosome", "chrom", "CHR"), colnames(detail_df))
pos_col <- intersect(c("pos", "position", "start", "probe_pos"), colnames(detail_df))
ratio_col <- intersect(c("ratio", "log2", "CN", "value", "log2ratio"), colnames(detail_df))

if (length(chr_col) == 0 || length(pos_col) == 0 || length(ratio_col) == 0) {
  stop("Unable to detect chr/pos/ratio columns in conumee detail data")
}

chr_col <- chr_col[1]
pos_col <- pos_col[1]
ratio_col <- ratio_col[1]

detail_df <- detail_df[, c(chr_col, pos_col, ratio_col)]
colnames(detail_df) <- c("chr", "pos", "ratio")
detail_df$chr <- gsub("chr", "", as.character(detail_df$chr))
detail_df$pos <- as.integer(detail_df$pos)
detail_df$ratio <- as.numeric(detail_df$ratio)

detail_df <- detail_df[!is.na(detail_df$pos) & !is.na(detail_df$ratio), ]

detail_df$bin_start <- floor(detail_df$pos / binsize) * binsize + 1
detail_df$bin_end <- detail_df$bin_start + binsize - 1

bins_df <- aggregate(ratio ~ chr + bin_start + bin_end, data=detail_df, FUN=mean)
counts_df <- aggregate(ratio ~ chr + bin_start + bin_end, data=detail_df, FUN=length)
colnames(counts_df)[4] <- "n_probes"
bins_df <- merge(bins_df, counts_df, by=c("chr", "bin_start", "bin_end"))

  write.table(bins_df, file=bins_out, sep="\t", row.names=FALSE, quote=FALSE)

  seg_df <- get_segments_df(x)
  if (!is.null(seg_df)) {
    chr_s <- intersect(c("chr", "chromosome", "chrom", "CHR"), colnames(seg_df))
    start_s <- intersect(c("start", "loc.start", "loc.start.pos"), colnames(seg_df))
    end_s <- intersect(c("end", "loc.end", "loc.end.pos"), colnames(seg_df))
    ratio_s <- intersect(c("seg.mean", "ratio", "mean", "log2"), colnames(seg_df))

    if (length(chr_s) > 0 && length(start_s) > 0 && length(end_s) > 0) {
      chr_s <- chr_s[1]
      start_s <- start_s[1]
      end_s <- end_s[1]
      ratio_s <- if (length(ratio_s) > 0) ratio_s[1] else NULL

      seg_out_df <- seg_df[, c(chr_s, start_s, end_s)]
      colnames(seg_out_df) <- c("chr", "start", "end")
      if (!is.null(ratio_s)) {
        seg_out_df$ratio <- seg_df[[ratio_s]]
      }
      seg_out_df$chr <- gsub("chr", "", as.character(seg_out_df$chr))
      write.table(seg_out_df, file=segments_out, sep="\t", row.names=FALSE, quote=FALSE)
    }
  }
}

if (!is.null(out_bins) && nzchar(out_bins)) {
  process_sample(idat_basenames[1], out_bins, out_segments)
} else if (!is.null(out_dir) && nzchar(out_dir)) {
  dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)
  for (idat_basename in idat_basenames) {
    sample_id <- basename(idat_basename)
    bins_out <- file.path(out_dir, paste0(sample_id, ".tsv"))
    segments_out <- file.path(out_dir, paste0(sample_id, ".segments.bed"))
    process_sample(idat_basename, bins_out, segments_out)
  }
} else {
  stop("No output path provided")
}
