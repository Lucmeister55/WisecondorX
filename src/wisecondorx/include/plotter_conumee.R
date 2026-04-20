options(warn=1)

# -----
# arg
# -----

args <- commandArgs(T)
in.file <- paste0(args[which(args == "--infile")+1])

genome_only <- TRUE
legend <- FALSE

# -----
# lib
# -----

suppressMessages(library("jsonlite"))

# -----
# main
# -----

input <- read_json(in.file, na="string")
binsize <- as.integer(input$binsize)
out.dir <- input$out_dir

dir.create(out.dir, showWarnings=F)

# param

gender = input$ref_gender
beta = as.numeric(input$beta)
zcutoff = as.numeric(input$zscore)
ylim = input$ylim
gene_call_method = if (!is.null(input$gene_call_method) && input$gene_call_method != "NULL") input$gene_call_method else "conumee"
gene_call_thr_gain = if (!is.null(input$gene_call_thr_gain) && input$gene_call_thr_gain != "NULL") as.numeric(input$gene_call_thr_gain) else NULL
gene_call_thr_loss = if (!is.null(input$gene_call_thr_loss) && input$gene_call_thr_loss != "NULL") as.numeric(input$gene_call_thr_loss) else NULL

if (!is.null(input$regions) && file.exists(input$regions)) {
  regions <- tryCatch({
    read.delim(input$regions, header=F, sep="\t", stringsAsFactors=F)
  }, error = function(e) {
    warning("Failed to read regions file. Proceeding without regions.")
    data.frame()  # empty data frame
  })
} else {
  warning("No regions file provided. Proceeding without regions.")
  regions <- data.frame()  # empty data frame
}


plot.title = input$plot_title

if (input$cairo) options(bitmaptype='cairo')

# aberration_cutoff

get.aberration.cutoff <- function(beta, ploidy){
    loss.cutoff = log2((ploidy - (beta / 2)) / ploidy)
    gain.cutoff = log2((ploidy + (beta / 2)) / ploidy)
    return(c(loss.cutoff, gain.cutoff))
}

# get n.reads (readable)

n.reads <- input$n_reads
first.part <- substr(n.reads, 1, nchar(n.reads) %% 3)
second.part <- substr(n.reads, nchar(n.reads) %% 3 + 1, nchar(n.reads))
n.reads <- c(first.part,  regmatches(second.part, gregexpr(".{3}", second.part))[[1]])
n.reads <- n.reads[n.reads != ""]
n.reads <- paste0(n.reads, collapse=".")

# get ratios

ratio <- unlist(input$results_r)
ratio[which(ratio == 0)] <- NA
weights <- unlist(input$results_w)
weights[which(weights == 0)] <- NA
variance <- unlist(input$results_variance)
variance[which(variance == 0)] <- NA
ref_sizes <- unlist(input$ref_sizes)
min_coverage_refsize <- as.numeric(input$min_coverage_refsize)
min_confidence_score <- as.numeric(input$min_confidence_score)
plot_loci_bed <- input$plot_loci_bed

chrs = 1:22

bins.per.chr <- sapply(chrs, FUN=function(x) length(unlist(input$results_r[x])))

labels = as.vector(sapply(chrs, FUN=function(x) paste0("chr", x)))

# define chromosome positions

chr.ends <- c(1, cumsum(bins.per.chr))
chr.mids <- chr.ends[2:length(chr.ends)] - bins.per.chr/2

ratio <- ratio[1:chr.ends[length(chrs) + 1]]
weights <- weights[1:chr.ends[length(chrs) + 1]]
variance <- variance[1:chr.ends[length(chrs) + 1]]
ref_sizes <- ref_sizes[1:chr.ends[length(chrs) + 1]]

# Apply coverage filter if specified
if (!is.na(min_coverage_refsize)) {
  ratio[ref_sizes < min_coverage_refsize] <- NA
  message(paste("Filtering: Masking", sum(ref_sizes < min_coverage_refsize, na.rm=TRUE), 
                "bins with ref_sizes <", min_coverage_refsize))
}

# Apply confidence filter if specified (confidence = ref_sizes / variance)
if (!is.na(min_confidence_score)) {
  confidence <- ref_sizes / variance
  ratio[confidence < min_confidence_score] <- NA
  message(paste("Filtering: Masking", sum(confidence < min_confidence_score, na.rm=TRUE),
                "bins with confidence <", min_confidence_score))
}

# Apply loci mask if specified (BED file)
if (!is.null(plot_loci_bed) && plot_loci_bed != "NULL" && file.exists(plot_loci_bed)) {
  loci <- tryCatch({
    read.delim(plot_loci_bed, header=FALSE, sep="\t", stringsAsFactors=FALSE)
  }, error = function(e) {
    warning("Failed to read plot loci BED. Proceeding without loci masking.")
    data.frame()
  })

  if (nrow(loci) > 0) {
    keep_mask <- rep(FALSE, length(ratio))
    for (i in seq_len(nrow(loci))) {
      chr_name <- as.character(loci[i, 1])
      start_pos <- as.numeric(loci[i, 2])
      end_pos <- as.numeric(loci[i, 3])

      if (is.na(start_pos) || is.na(end_pos)) next

      chr_name <- gsub("chr", "", chr_name, ignore.case=TRUE)
      if (chr_name == "X" || chr_name == "Y") next
      chr <- suppressWarnings(as.integer(chr_name))
      if (is.na(chr) || chr < 1 || chr > 22) next

      bin_start <- floor(start_pos / binsize)
      bin_end <- floor((end_pos - 1) / binsize)
      if (bin_end < 0) next

      chr_offset <- chr.ends[chr]
      chr_bins <- bins.per.chr[chr]
      bin_start <- max(bin_start, 0)
      bin_end <- min(bin_end, chr_bins - 1)
      if (bin_start > bin_end) next

      idx_start <- chr_offset + bin_start + 1
      idx_end <- chr_offset + bin_end + 1
      keep_mask[idx_start:idx_end] <- TRUE
    }

    ratio[!keep_mask] <- NA
    message(paste("Loci mask: kept", sum(keep_mask, na.rm=TRUE), "bins from BED"))
  }
}

# Summary of plotted vs not plotted bins (after all masking)
total_bins <- length(ratio)
kept_bins <- sum(!is.na(ratio))
not_plotted_bins <- total_bins - kept_bins
message(paste("Plot bins: kept", kept_bins, "/", total_bins, "; not plotted", not_plotted_bins))

# get margins

box.list <- list()
l.whis.per.chr <- c()
h.whis.per.chr <- c()

for (chr in chrs){
  box.list[[chr]] <- ratio[chr.ends[chr]:chr.ends[chr + 1]]
  whis = boxplot(box.list[[chr]], plot=F)$stats[c(1,5),]
  l.whis.per.chr = c(l.whis.per.chr, whis[1])
  h.whis.per.chr = c(h.whis.per.chr, whis[2])
}

# Reduce vertical space (conumee-style default)
expand.factor <- 1.0

chr.wide.upper.limit <- max(0.65, max(h.whis.per.chr, na.rm=T)) * expand.factor
chr.wide.lower.limit <- min(-0.95, min(l.whis.per.chr, na.rm=T)) * expand.factor

if (ylim != 'def'){
  ylim=gsub('[', '', ylim, fixed=T) ; ylim=gsub(']', '', ylim, fixed=T)
  ylim=as.numeric(strsplit(ylim, ',', fixed=T)[[1]])
  chr.wide.lower.limit = ylim[1]
  chr.wide.upper.limit = ylim[2]
}

# Force ylim and y.ticks for conumee-style
y.ticks <- round(seq(-1.2, 1.2, by=0.4), 1)
chr.wide.lower.limit <- -1.25
chr.wide.upper.limit <- 1.25

# plot chromosome wide plot

black = "#3f3f3f"

color.segmentLine = "#e0e0e0"

# Chromosome separator and centromere colors/styles
chr_sep_col <- "#bdbdbd"  # solid grey for chromosome separators
cent_col <- "#9e9e9e"     # dashed grey for centromeres

# Conumee2-style colors
color.A  = "lightgrey"   # neutral
color.B  = "red"         # loss
color.C  = "green"       # gain
color.D  = "darkblue"    # labels etc
color.X <- c(rgb(141, 209, 198, maxColorValue=255), rgb(84, 84, 84, maxColorValue=255), rgb(227, 200, 138, maxColorValue=255))

# Transparent versions for segments
color.AA = adjustcolor(color.A, alpha.f=0.3)
color.BB = adjustcolor(color.B, alpha.f=0.3)
color.CC = adjustcolor(color.C, alpha.f=0.3)
color.XX = c(color.CC, color.AA, color.BB)

png(paste0(out.dir, "/genome_wide.png"), width=12, height=6, units="in", res=720, pointsize=12)

# Instead, just plot the top panel for genome-wide plot:
layout(matrix(1, nrow=1, ncol=1))  # single panel
par(mar=c(4,4,4,4), oma=c(0,0,0,0), mgp=c(2,0.7,0))
par(cex=0.8, cex.axis=1.0, cex.lab=1.0)
par(xaxs="i")

plot(1, main="", axes=F, # plots nothing -- enables segments function
     xlab="", ylab="", xlim=c(chr.ends[1], chr.ends[length(chr.ends)]),
     cex=0, ylim=c(chr.wide.lower.limit,chr.wide.upper.limit))

plot.constitutionals <- function(ploidy, start, end){
  segments(start, log2(2/ploidy), end, log2(2/ploidy), col=color.A, lwd=2, lty=3)
}

genome.len <- chr.ends[length(chr.ends)]
autosome.len <- chr.ends[22]
if (gender == "F"){
  plot.constitutionals(2, -genome.len * 0.025, genome.len * 1.025)
} else {
  plot.constitutionals(2, -genome.len * 0.025, autosome.len)
  plot.constitutionals(1, autosome.len, genome.len * 1.025)
}

for (undetectable.index in which(is.na(ratio))){
  segments(undetectable.index, par("usr")[3], undetectable.index, par("usr")[4], col="#e0e0e0", lwd=0.1, lty=1)
}

# Draw chromosome separator lines behind all overlays so they do not cover labels/segments
# Use plot extremes so lines reach top and bottom edges
y_bot <- par("usr")[3]
y_top <- par("usr")[4]
for (x in chr.ends){
  segments(x, y_bot, x, y_top, col=chr_sep_col, lwd=1.0, lty=1)
}

# -----------------------------
# Set dot colors (conumee-style gradient)
# -----------------------------
n.colors <- 1000
gradient.colors <- colorRampPalette(c("red", "red", "lightgrey", "green", "green"))(n.colors)

# Set limits for gradient mapping based on ylim
max_ratio <- max(abs(c(chr.wide.lower.limit, chr.wide.upper.limit)))
ratio.clipped <- pmax(pmin(ratio, max_ratio), -max_ratio)

# Map ratio values to gradient
dot.cols <- sapply(ratio.clipped, function(x) {
  if (is.na(x)) return("grey")  # missing values
  index <- round((x + max_ratio) / (2 * max_ratio) * (n.colors - 1)) + 1
  gradient.colors[index]
})

# Calculate dot sizes based on distance to each segment median
# Near segment median (blue line) = bigger dots, far = smaller dots
seg_median_by_bin <- rep(NA_real_, length(ratio))
for (ab in input$results_c){
  info = unlist(ab)
  chr = as.integer(info[1]) + 1
  if (is.na(chr) || chr < 1 || chr > length(chr.ends) - 1) next
  start = as.integer(info[2]) + chr.ends[chr] + 1
  end = as.integer(info[3]) + chr.ends[chr]
  if (is.na(start) || is.na(end) || start >= end) next
  seg_median <- median(ratio[start:end], na.rm=TRUE)
  seg_median_by_bin[start:end] <- seg_median
}

global_median <- median(ratio, na.rm=TRUE)
seg_median_by_bin[is.na(seg_median_by_bin)] <- global_median

ratio_dist <- abs(ratio - seg_median_by_bin)
min_dist <- quantile(ratio_dist, 0.01, na.rm=TRUE)
max_dist <- quantile(ratio_dist, 0.99, na.rm=TRUE)

if (!is.finite(min_dist) || !is.finite(max_dist) || min_dist == max_dist) {
  dot.cex <- rep(0.7, length(ratio))
} else {
  ratio_dist_clipped <- pmax(pmin(ratio_dist, max_dist), min_dist)
  # Relative distance (sample-specific) and absolute distance (axis-referenced)
  # are blended so high-spread samples shrink dots more aggressively.
  dist_scaled <- 1 - (ratio_dist_clipped - min_dist) / (max_dist - min_dist)
  spread_range <- max_dist - min_dist
  spread_norm <- pmin(pmax(spread_range / max_ratio, 0), 2)
  w_abs <- pmin(0.85, 0.20 + 0.35 * spread_norm)
  abs_scaled <- 1 - pmin(ratio_dist_clipped / max_ratio, 1)
  combined_scaled <- (1 - w_abs) * dist_scaled + w_abs * abs_scaled
  combined_scaled[is.na(combined_scaled)] <- 0.5

  # Increase non-linearity with spread to accelerate shrinkage in noisy samples.
  power <- 1.6 + 1.2 * spread_norm
  dot.cex <- 0.15 + 0.95 * (combined_scaled ^ power)
}


# create labels dataframe
gene_labels <- data.frame(start_bin=integer(), end_bin=integer(), label=character(), label_position=double(), label_adj=integer(), dot_x=double(), dot_y=double())

# Load amplified and deleted genes from TSV files if they exist
amplified_genes <- character(0)
deleted_genes <- character(0)
outid_base <- gsub("\\.plots$", "", out.dir)
amp_file <- paste0(outid_base, "_amplified_genes.tsv")
del_file <- paste0(outid_base, "_deleted_genes.tsv")
if (file.exists(amp_file)) {
  tryCatch({
    amp_df <- read.delim(amp_file, header=TRUE, sep="\t", stringsAsFactors=FALSE)
    if (nrow(amp_df) > 0 && "name" %in% colnames(amp_df)) {
      amplified_genes <- unique(trimws(amp_df$name))
    }
  }, error = function(e) {})
}
if (file.exists(del_file)) {
  tryCatch({
    del_df <- read.delim(del_file, header=TRUE, sep="\t", stringsAsFactors=FALSE)
    if (nrow(del_df) > 0 && "name" %in% colnames(del_df)) {
      deleted_genes <- unique(trimws(del_df$name))
    }
  }, error = function(e) {})
}

append_labels_from_regions <- function(regions_df) {
  if (nrow(regions_df) == 0) return(invisible(NULL))

  for (i in seq_len(nrow(regions_df))) {
    region <- regions_df[i, ]
    chr <- gsub("chr", "", as.character(region[1]))
    if (chr %in% c("X", "Y")) {
      warning(paste("Skipping region on chrX/chrY:", paste(region[1:4], collapse="\t")))
      next
    }
    chr <- suppressWarnings(as.integer(chr))
    if (is.na(chr) || chr < 1 || chr > 22) {
      warning(paste("Skipping region with invalid chromosome:", region[1]))
      next
    }
    start_pos <- suppressWarnings(as.integer(region[2]))
    end_pos <- suppressWarnings(as.integer(region[3]))
    if (is.na(start_pos) || is.na(end_pos)) {
      warning(paste("Skipping region with invalid start/end:", region[2], region[3]))
      next
    }

    # Figure out start and end bins based on coordinates
    start_bin <- floor(start_pos / binsize) + chr.ends[chr] + 1
    end_bin <- floor((end_pos - 1) / binsize) + chr.ends[chr] + 1
    if (start_bin > end_bin) {
      warning(paste("Skipping region with start > end:", start_pos, end_pos))
      next
    }

    label_value <- if (ncol(regions_df) >= 4 && !is.na(region[4]) && nzchar(as.character(region[4]))) {
      as.character(region[4])
    } else {
      paste0("chr", chr, ":", start_pos, "-", end_pos)
    }

    if (all(is.na(ratio[start_bin:end_bin]))) {
      warning(paste("Skipping region with no plotted bins (all NA):", paste(region[1:4], collapse="\t")))
      next
    }

    # Use a single representative dot per region
    dot_x <- start_bin + (end_bin - start_bin) / 2
    dot_y <- median(ratio[start_bin:end_bin], na.rm=TRUE)
    if (!is.finite(dot_y)) {
      warning(paste("Skipping region with non-finite ratio:", paste(region[1:4], collapse="\t")))
      next
    }

    # Place label near the dot, on the same side of the segment median line
    label_offset <- 0.02 * (chr.wide.upper.limit - chr.wide.lower.limit)
    dot_idx <- min(max(round(dot_x), 1), length(seg_median_by_bin))
    median_line <- seg_median_by_bin[dot_idx]
    if (is.na(median_line)) {
      median_line <- global_median
    }
    if (dot_y >= median_line) {
      label_position <- dot_y + label_offset
      label_adj <- 0
    } else {
      label_position <- dot_y - label_offset
      label_adj <- 1
    }

    gene_labels <<- rbind(gene_labels,
                data.frame(start_bin=start_bin, end_bin=end_bin,
                     label=label_value,
                     label_position=label_position,
                     label_adj=label_adj,
                     dot_x=dot_x,
                     dot_y=dot_y))
  }
}

append_labels_from_regions(regions)

colnames(gene_labels) <- c("start_bin", "end_bin", "label",
                           "label_position", "label_adj", "dot_x", "dot_y")

# Draw gene-call threshold lines BEFORE points if provided and using segment-wise method (behind points)
if (gene_call_method == "segment-wise") {
  if (!is.null(gene_call_thr_gain) && is.finite(gene_call_thr_gain)) {
    segments(chr.ends[1], gene_call_thr_gain, chr.ends[length(chr.ends)], gene_call_thr_gain,
             col=adjustcolor("black", alpha.f=0.9), lwd=1.5, lty=2)  # green for gain
  }
  if (!is.null(gene_call_thr_loss) && is.finite(gene_call_thr_loss)) {
    segments(chr.ends[1], gene_call_thr_loss, chr.ends[length(chr.ends)], gene_call_thr_loss,
             col=adjustcolor("black", alpha.f=0.9), lwd=1.5, lty=2)  # red for loss
  }
}

par(new=T)
plot(ratio, main="", axes=F,
     xlab="", ylab="", col=dot.cols, pch=16,
     ylim=c(chr.wide.lower.limit,chr.wide.upper.limit), cex=dot.cex)

# Determine aberration cutoffs for highlighting labels
cutoffs <- get.aberration.cutoff(beta, 2)
loss_cutoff <- cutoffs[1]
gain_cutoff <- cutoffs[2]

# Plot gene labels
for (i in seq_len(nrow(gene_labels))){
  start_bin = gene_labels$start_bin[i]
  end_bin = gene_labels$end_bin[i]
  label = gene_labels$label[i]
  label_position = gene_labels$label_position[i]
  label_adj = gene_labels$label_adj[i]
  
  # Get actual dot position and value
  dot_x <- gene_labels$dot_x[i]
  dot_y <- gene_labels$dot_y[i]
  
  # Check if dot is beyond chr.wide scale limits
  if (dot_y < chr.wide.lower.limit) {
    # Dot is below lower limit - clamp to lower edge
    clamped_dot_y <- chr.wide.lower.limit
    label_y <- chr.wide.lower.limit - 0.08
    is_clamped <- TRUE
  } else if (dot_y > chr.wide.upper.limit) {
    # Dot is above upper limit - clamp to upper edge
    clamped_dot_y <- chr.wide.upper.limit
    label_y <- chr.wide.upper.limit + 0.08
    is_clamped <- TRUE
  } else {
    # Dot is within scale limits
    clamped_dot_y <- dot_y
    label_y <- label_position
    is_clamped <- FALSE
  }
  
  # Overlay a single representative point for the gene region
  points(dot_x, clamped_dot_y,
    col="black", pch=16, cex=0.85, lwd=1)
  
  # Add the label: color based on gene call status (green=amplified, red=deleted, black=neutral)
  gene_name <- gene_labels$label[i]
  if (gene_name %in% amplified_genes) {
    lab_col <- "darkgreen"
  } else if (gene_name %in% deleted_genes) {
    lab_col <- adjustcolor("#B30000", alpha.f=0.95)  # darker red, more opaque
  } else {
    lab_col <- "black"
  }
  
  text(dot_x, label_y,
    labels=label, col=lab_col, cex=1.1, srt=90, adj=0.5, font=1)
  
  # Add true log2 ratio as small text if clamped
  if (is_clamped) {
    ratio_text <- sprintf("%.2f", dot_y)
    par(xpd=NA)
    text(dot_x + 1.5, label_y,
      labels=ratio_text, col=lab_col, cex=0.7, font=1)
    par(xpd=F)
  }
}

# Draw x and y axes (conumee-style)
par(xpd=NA)
# Calculate offset as 5% of y-axis range
y_offset <- 0.02 * (par("usr")[4] - par("usr")[3])

# Draw x-axis labels with offset and right-justification
text(x = chr.mids, 
  y = par("usr")[3] - y_offset,  # move labels below the ticks
  labels = labels, 
  srt = 90, 
  adj = 1,   # right-justified for vertical text
  xpd = NA,
  cex = 1.0)
axis(1, at=chr.mids, labels=FALSE, tick=TRUE, tcl=-0.3, las=2, cex.axis=1.0)  # x-axis with ticks outward
axis(2, at=y.ticks, tick=TRUE, tcl=-0.3, las=1, cex.axis=1.0)                  # y-axis with ticks outward
box()
par(xpd=F)

## Chromosome separators drawn earlier so they are behind overlays

# Legends

if (legend) {
  par(xpd=NA)
  legend(x=chr.ends[length(chr.ends)] * 0.3,
        y=chr.wide.upper.limit + (abs(chr.wide.upper.limit) + abs(chr.wide.lower.limit)) * 0.23,
        legend=c("Constitutional 3n", "Constitutional 2n", "Constitutional 1n"),
        text.col=c(color.C, color.A, color.B), cex=1.3, bty="n", lty=c(3,3,3), lwd=1.5, col=c(color.C, color.A, color.B))
  legend(x=0,
        y=chr.wide.upper.limit + (abs(chr.wide.upper.limit) + abs(chr.wide.lower.limit)) * 0.23,
        legend=c("Gain", "Loss", paste0("Number of reads: ", n.reads)), text.col=c(color.C, color.B, black),
        cex=1.3, bty="n", pch=c(16,16), col=c(color.C, color.B, "white"))
  par(xpd=F)
}

if (!is.null(plot.title)) {
    par(xpd=NA)
    title(plot.title, line=1.3, adj=0.8, col.main=color.A)
    par(xpd=F)
}

# plot segmentation (conumee-style median lines)
for (ab in input$results_c){
  info = unlist(ab)
  chr = as.integer(info[1]) + 1
  
  # skip X/Y chromosomes
  if (chr > 22) next
  
  start = as.integer(info[2]) + chr.ends[chr] + 1
  end   = as.integer(info[3]) + chr.ends[chr]
  height = as.double(info[5])
  
  rect(start, height, end, 0, col=color.XX[dot.cols[start] == color.X], 
       border=color.XX[dot.cols[start] == color.X], lwd=0.1)

  # calculate median for this segment
  seg_median <- median(ratio[start:end], na.rm=TRUE)
  # Draw horizontal line at median
  segments(start, seg_median, end, seg_median, col="darkblue", 
           lwd=4, lty=1)
}

# write image

invisible(dev.off())

# boxplots

# Define ylim
ymin <- min(l.whis.per.chr[1:22], na.rm = TRUE)
ymax <- max(h.whis.per.chr[1:22], na.rm = TRUE)

# Filter box.list to remove outliers outside plotting range
filtered.box.list <- lapply(box.list[1:22], function(x) x[x >= ymin & x <= ymax])

# Save autosome boxplot to a separate file
png(paste0(out.dir, "/autosome_boxplot.png"), width=12, height=6, units="in", res=720, pointsize=12)

# Reduce margins: bottom, left, top, right
par(mar = c(3, 4, 1, 1), mgp = c(2.5, 0.5, 0))  

# Plot with filtered points
boxplot(filtered.box.list, 
        ylim = c(ymin, ymax), 
        bg = black,
        axes = FALSE, 
        outpch = 16, 
        ylab = expression('log'[2]*'(ratio)'))

par(xpd = NA)
text(1:22, par("usr")[3], labels = labels[1:22], srt = 45, pos = 1)

axis(2, tick = TRUE, cex.lab = 2, col = black, las = 1, tcl = 0.5)
par(xpd = F)

plot.constitutionals(2, 0, 23)

dev.off()

# create chr specific plots

if (!genome_only){
  for (c in chrs){

    margins <- c(chr.ends[c], chr.ends[c+1])
    len <- chr.ends[c+1] - chr.ends[c]
    x.labels <- seq(0, bins.per.chr[c] * binsize, bins.per.chr[c] * binsize / 10)
    x.labels.at <- seq(0, bins.per.chr[c], bins.per.chr[c] / 10) + chr.ends[c]
    x.labels <- x.labels[2:(length(x.labels) - 1)]
    x.labels.at <- x.labels.at[2:(length(x.labels.at) - 1)]

    whis = boxplot(box.list[[c]], plot=F)$stats[c(1,5),]

    if (any(is.na(whis))){
      next
    }

    png(paste0(out.dir, "/", labels[c],".png"), width=12,height=6,units="in",res=720,pointsize=12)

    upper.limit <- 0.6 + whis[2]
    lower.limit <- -1.05 + whis[1]
    upper.limit <- max(upper.limit, max(ratio[margins[1]:margins[2]], na.rm = T))
    lower.limit <- min(lower.limit, min(ratio[margins[1]:margins[2]], na.rm = T))
    if (ylim != 'def'){
      lower.limit = ylim[1] ; upper.limit = ylim[2]
    }
    par(mar=c(2.5,4,1,0), mgp=c(2.4,0.5,0))

    plot(1, main="", axes=F, # plots nothing -- enables segments function
        xlab="", ylab="", cex=0, ylim=c(lower.limit,upper.limit), xlim=margins)

    if (gender == "F"){
      plot.constitutionals(2, chr.ends[c] - bins.per.chr[c] * 0.02, chr.ends[c+1] + bins.per.chr[c] * 0.02)
    } else {
      if (c == 23 | c == 24){
        plot.constitutionals(1, chr.ends[c] - bins.per.chr[c] * 0.02, chr.ends[c+1] + bins.per.chr[c] * 0.02)
      } else {
        plot.constitutionals(2, chr.ends[c] - bins.per.chr[c] * 0.02, chr.ends[c+1] + bins.per.chr[c] * 0.02)
      }
    }

    for (undetectable.index in which(is.na(ratio))){
      segments(undetectable.index, par("usr")[3], undetectable.index, par("usr")[4],
              col=color.A, lwd=1/len * 200, lty=1)
    }

    par(new=T)
    plot(ratio, main=labels[c], axes=F,
        xlab="", ylab=expression('log'[2]*'(ratio)'), col=dot.cols, pch=16,
        cex=dot.cex, ylim=c(lower.limit,upper.limit),
        xlim=margins)

    for (ab in input$results_c){
      info = unlist(ab)
      chr = as.integer(info[1]) + 1
      start = as.integer(info[2]) + chr.ends[chr] + 1
      end = as.integer(info[3]) + chr.ends[chr]
      height = as.double(info[5])
      rect(start, height, end, 0, col=color.XX[dot.cols[start] == color.X],border=color.XX[dot.cols[start] == color.X], lwd=0.1)
      # Conumee-style median line for segment
      seg_median <- median(ratio[start:end], na.rm=TRUE)
       segments(start, seg_median, end, seg_median, col="darkblue",
         lwd=3, lty=1)
    }

    rect(0, lower.limit - 10, chr.ends[c], upper.limit + 10, col="white", border=NA)
    rect(chr.ends[c+1], lower.limit - 10, chr.ends[length(chr.ends)], upper.limit + 10, col="white", border=NA)

    for (i in seq_len(nrow(gene_labels))){
      start_bin = gene_labels$start_bin[i]
      end_bin = gene_labels$end_bin[i]
      label = gene_labels$label[i]
      label_position = gene_labels$label_position[i]
      label_adj = gene_labels$label_adj[i]
      
      # Get actual dot position and value
      dot_x <- gene_labels$dot_x[i]
      dot_y <- gene_labels$dot_y[i]
      
      # Check if dot is beyond chromosome-specific scale limits
      if (dot_y < lower.limit) {
        # Dot is below lower limit - clamp to lower edge
        clamped_dot_y <- lower.limit
        label_y <- lower.limit - 0.08 * (upper.limit - lower.limit)
        is_clamped <- TRUE
      } else if (dot_y > upper.limit) {
        # Dot is above upper limit - clamp to upper edge
        clamped_dot_y <- upper.limit
        label_y <- upper.limit + 0.08 * (upper.limit - lower.limit)
        is_clamped <- TRUE
      } else {
        # Dot is within scale limits
        clamped_dot_y <- dot_y
        label_y <- label_position
        is_clamped <- FALSE
      }
      
      # Overlay the representative point for the gene region
      points(dot_x, clamped_dot_y, col=color.D, pch=16, cex=1.1, lwd=1)
      
      # Add the label
      par(xpd=NA)
      text(dot_x, label_y,
          labels=label, col=color.D, cex=1.05, srt=90, adj=0.5)
      
      # Add true log2 ratio as small text if clamped
      if (is_clamped) {
        ratio_text <- sprintf("%.2f", dot_y)
        text(dot_x + 1.5, label_y,
          labels=ratio_text, col=color.D, cex=0.65)
      }
      par(xpd=F)
  } 

    par(xpd=NA)
    text(x.labels.at, par("usr")[3], labels=x.labels, srt=45, pos=1)
    axis(2, at = y.ticks, tick=T, cex.lab=2, col=black, las=1, tcl=0.5)
    par(xpd=F)

    for (x in chr.ends){
      segments(x, lower.limit * 1.03, x, upper.limit * 1.03, col=black, lwd=2, lty=3)
    }
    for (x in x.labels.at){
      segments(x, lower.limit * 1.02, x, upper.limit * 1.02, col=black, lwd=1, lty=3)
    }
    invisible(dev.off())
  }

  q(save="no")
}