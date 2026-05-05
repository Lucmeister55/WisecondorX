#!/usr/bin/env Rscript
# Re-draw CNV_genomeplot.png for each EPIC sample, overlaying broad gene calls in purple.
# Called from epic_cfrrbs.py after broad gene calls are computed (Phase 3).
#
# Input JSON: { samples: [{rds_path, out_dir, epic_id, broad_amp_genes, broad_del_genes}] }
# broad_amp_genes / broad_del_genes: arrays of gene name strings

options(warn=1)

suppressMessages(library("jsonlite"))
suppressMessages(library("conumee2"))

args <- commandArgs(T)
in_file <- paste0(args[which(args == "--infile")+1])
input <- read_json(in_file, na = "string")

samples <- input$samples
if (is.null(samples) || length(samples) == 0) {
  cat("No samples to process.\n")
  quit(save = "no", status = 0)
}

for (s in samples) {
  rds_path       <- s$rds_path
  out_dir        <- s$out_dir
  epic_id        <- s$epic_id
  broad_amp_genes <- unique(as.character(unlist(s$broad_amp_genes)))
  broad_del_genes <- unique(as.character(unlist(s$broad_del_genes)))
  broad_genes     <- unique(c(broad_amp_genes, broad_del_genes))
  broad_genes     <- broad_genes[nzchar(broad_genes)]

  if (!file.exists(rds_path)) {
    cat(paste("Skipping", epic_id, "— RDS not found:", rds_path, "\n"))
    next
  }

  tryCatch({
    x <- readRDS(rds_path)
    chr_sel <- paste0("chr", 1:22)
    png_path <- file.path(out_dir, "CNV_genomeplot.png")

    png(png_path, width = 12, height = 6, units = "in", res = 720, pointsize = 12)
    par(mar = c(4, 4, 4, 4), oma = c(0, 0, 0, 0), mgp = c(2, 0.7, 0))
    par(xaxs = "i", cex = 0.8, cex.axis = 1.0, cex.lab = 1.0)

    CNV.genomeplot(
      x,
      chr      = chr_sel,
      cols     = c("red", "red", "lightgrey", "green", "green"),
      main     = "",
      bins_cex = 0.75,
      set_par  = FALSE
    )

    # Overlay broad gene annotations in purple
    if (length(broad_genes) > 0) {
      tryCatch({
        anno      <- x@anno
        all_bins  <- anno@bins
        sel_idx   <- which(as.character(GenomeInfoDb::seqnames(all_bins)) %in% chr_sel)
        sel_bins  <- all_bins[sel_idx]

        detail_ratio   <- x@detail$ratio[[1]]
        detail_regions <- anno@detail

        for (gene in broad_genes) {
          if (!(gene %in% names(detail_ratio)))   next
          if (!(gene %in% names(detail_regions))) next

          gene_gr   <- detail_regions[gene]
          gene_chr  <- as.character(GenomeInfoDb::seqnames(gene_gr))
          if (!(gene_chr %in% chr_sel)) next

          ov <- GenomicRanges::findOverlaps(gene_gr, sel_bins)
          if (length(ov) == 0) next
          x_pos <- mean(S4Vectors::subjectHits(ov))

          y_pos <- as.numeric(detail_ratio[[gene]])
          if (!is.finite(y_pos)) next

          usr <- par("usr")
          y_clamp <- max(usr[3], min(usr[4], y_pos))
          par(xpd = NA)
          points(x_pos, y_clamp, col = "#8e44ad", pch = 19, cex = 1.0)
          text(x_pos, y_clamp + 0.04 * (usr[4] - usr[3]),
               labels = gene, col = "#8e44ad",
               cex = 0.75, srt = 90, adj = 0, font = 2)
          par(xpd = FALSE)
        }
      }, error = function(e) {
        cat(paste("  Warning: broad annotation failed for", epic_id, ":", e$message, "\n"))
      })
    }

    dev.off()
    cat(paste("Regenerated genomeplot for", epic_id, "\n"))

  }, error = function(e) {
    try(dev.off(), silent = TRUE)
    cat(paste("Error processing", epic_id, ":", e$message, "\n"))
  })
}

cat("Done.\n")
