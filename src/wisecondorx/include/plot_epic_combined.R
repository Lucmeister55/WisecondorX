#!/usr/bin/env Rscript
# Plot EPIC RDS data using conumee2: genomeplot for each sample + summaryplot + heatmap for all samples
# Loads all packages and RDS files at once to avoid repetitive overhead

options(warn=1)

# Parse JSON input
args <- commandArgs(T)
in_file <- paste0(args[which(args == "--infile")+1])

suppressMessages(library("jsonlite"))
suppressMessages(library("conumee2"))

# Read input JSON
input <- read_json(in_file, na="string")
rds_files <- unlist(input$rds_files)
out_dirs <- unlist(input$out_dirs)
epic_ids <- unlist(input$epic_ids)
summary_dir <- input$summary_dir

# Validate inputs
if (length(rds_files) != length(out_dirs) || length(rds_files) != length(epic_ids)) {
  stop("Mismatch: rds_files, out_dirs, and epic_ids must have same length")
}

if (!dir.exists(summary_dir)) {
  dir.create(summary_dir, recursive=TRUE, showWarnings=FALSE)
}

# Load all RDS files
# Keep strict positional indexing so duplicate epic_ids do not overwrite entries.
conumee_list <- vector("list", length(rds_files))
sample_keys <- make.unique(epic_ids, sep = "__dup")
names(conumee_list) <- sample_keys

# Persist a pre-loop audit table so each iteration mapping can be inspected later.
iter_map <- data.frame(
  iter_idx = seq_along(rds_files),
  epic_id = as.character(epic_ids),
  sample_key = as.character(sample_keys),
  rds_file = as.character(rds_files),
  out_dir = as.character(out_dirs),
  stringsAsFactors = FALSE
)
iter_map$rds_exists <- file.exists(iter_map$rds_file)
iter_map$out_dir_exists_pre <- dir.exists(iter_map$out_dir)
iter_map$epic_id_duplicate <- duplicated(iter_map$epic_id) | duplicated(iter_map$epic_id, fromLast = TRUE)
iter_map$sample_key_duplicate <- duplicated(iter_map$sample_key) | duplicated(iter_map$sample_key, fromLast = TRUE)
iter_map$rds_basename <- basename(iter_map$rds_file)
iter_map$out_dir_basename <- basename(iter_map$out_dir)

iter_map_path <- file.path(summary_dir, "epic_iteration_input_map.tsv")
write.table(iter_map, file = iter_map_path, sep = "\t", row.names = FALSE, quote = FALSE)
cat(paste("Saved EPIC iteration input map to", iter_map_path, "\n"))

dup_count <- sum(iter_map$epic_id_duplicate, na.rm = TRUE)
if (dup_count > 0) {
  cat(paste("Warning:", dup_count, "rows have duplicated epic_id values; inspect epic_iteration_input_map.tsv\n"))
}

tryCatch({
  for (i in seq_along(rds_files)) {
    rds_path <- rds_files[i]
    epic_id <- epic_ids[i]
    
    if (!file.exists(rds_path)) {
      stop(paste("RDS file not found:", rds_path))
    }
    
    x <- readRDS(rds_path)
    conumee_list[[i]] <- x
    cat(paste("Loaded RDS:", epic_id, "\n"))
  }
  
  # Generate genomeplot for each sample
  cat("\nGenerating individual genomeplot images and per-sample outputs...\n")
  for (i in seq_along(rds_files)) {
    tryCatch({
      x <- conumee_list[[i]]
      out_dir <- out_dirs[i]
      epic_id <- epic_ids[i]
      
      if (!dir.exists(out_dir)) {
        dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)
      }
      
      chr_sel <- paste0("chr", c(1:22))
      
      png_path <- file.path(out_dir, "CNV_genomeplot.png")
      png(png_path, width = 12, height = 6, units = "in", res = 720, pointsize = 12)
      par(mar = c(4, 4, 4, 4), oma = c(0, 0, 0, 0), mgp = c(2, 0.7, 0))
      par(xaxs = "i", cex = 0.8, cex.axis = 1.0, cex.lab = 1.0)
      
      plot_res <- try(
        CNV.genomeplot(
          x,
          chr = chr_sel,
          cols = c("red", "red", "lightgrey", "green", "green"),
          main = "",
          bins_cex = 0.75,
          set_par = FALSE
        ),
        silent = TRUE
      )
      if (inherits(plot_res, "try-error")) {
        CNV.genomeplot(
          x,
          chr = chr_sel,
          cols = c("red", "red", "lightgrey", "green", "green"),
          main = "",
          bins_cex = 0.75,
          set_par = FALSE
        )
        title(main = "")
      }
      dev.off()
      
      cat(paste("  Saved genomeplot for", epic_id, "to", png_path, "\n"))

      # Save wrapped detail plot per sample
      tryCatch({
        detail_png_path <- file.path(out_dir, "CNV_detailplot_wrap.png")
        png(detail_png_path, width = 14, height = 10, units = "in", res = 300, pointsize = 11)
        CNV.detailplot_wrap(x)
        dev.off()
        cat(paste("  Saved detailplot_wrap for", epic_id, "to", detail_png_path, "\n"))
      }, error = function(e) {
        # Ensure device is closed if plotting failed
        try(dev.off(), silent = TRUE)
        cat(paste("    Warning: Failed to generate detailplot_wrap:", e$message, "\n"))
      })
      
      # Write per-sample outputs
      tryCatch({
        # Bins output
        bins_out <- file.path(out_dir, paste0(epic_id, "_bins.tsv"))
        CNV.write(x, file = bins_out, what = "bins")
        cat(paste("    Wrote bins to", bins_out, "\n"))
      }, error = function(e) {
        cat(paste("    Warning: Failed to write bins:", e$message, "\n"))
      })
      
      tryCatch({
        # Segments output
        segs_out <- file.path(out_dir, paste0(epic_id, "_segments.tsv"))
        CNV.write(x, file = segs_out, what = "segments")
        cat(paste("    Wrote segments to", segs_out, "\n"))
      }, error = function(e) {
        cat(paste("    Warning: Failed to write segments:", e$message, "\n"))
      })
      
      tryCatch({
        # GISTIC output (for downstream processing)
        gistic_out <- file.path(out_dir, paste0(epic_id, "_gistic.tsv"))
        CNV.write(x, file = gistic_out, what = "gistic")
        cat(paste("    Wrote GISTIC to", gistic_out, "\n"))
      }, error = function(e) {
        cat(paste("    Warning: Failed to write GISTIC:", e$message, "\n"))
      })
      
      tryCatch({
        # Detail output (contains focal information)
        detail_out <- file.path(out_dir, paste0(epic_id, "_detail.tsv"))
        CNV.write(x, file = detail_out, what = "detail")
        cat(paste("    Wrote detail to", detail_out, "\n"))
      }, error = function(e) {
        cat(paste("    Warning: Failed to write detail (CNV.write):", e$message, "\n"))
      })
      
      tryCatch({
        # Extract and write individual detail components
        detail <- x@detail
        
        # Write cancer gene ratios (copy number log2 ratios for focal genes)
        if (!is.null(detail$ratio) && length(detail$ratio) > 0) {
          ratio_data <- data.frame(gene = names(detail$ratio[[1]]), ratio = as.numeric(detail$ratio[[1]]))
          ratio_out <- file.path(out_dir, paste0(epic_id, "_focal_ratio.tsv"))
          write.table(ratio_data, file = ratio_out, sep = "\t", row.names = FALSE, quote = FALSE)
          cat(paste("    Wrote focal ratios to", ratio_out, "\n"))
        }
        
        # Write probes per gene
        if (!is.null(detail$probes) && length(detail$probes) > 0) {
          probes_data <- data.frame(gene = names(detail$probes), probes = as.numeric(detail$probes))
          probes_out <- file.path(out_dir, paste0(epic_id, "_focal_probes.tsv"))
          write.table(probes_data, file = probes_out, sep = "\t", row.names = FALSE, quote = FALSE)
          cat(paste("    Wrote focal probes to", probes_out, "\n"))
        }
        
        # Write amplified bins
        if (!is.null(detail$amp.bins[[1]]) && length(detail$amp.bins[[1]]) > 0) {
          amp_data <- data.frame(bin = names(detail$amp.bins[[1]]), ratio = as.numeric(detail$amp.bins[[1]]))
          amp_out <- file.path(out_dir, paste0(epic_id, "_amp_bins.tsv"))
          write.table(amp_data, file = amp_out, sep = "\t", row.names = FALSE, quote = FALSE)
          cat(paste("    Wrote amplified bins to", amp_out, "\n"))
        }
        
        # Write deleted bins
        if (!is.null(detail$del.bins[[1]]) && length(detail$del.bins[[1]]) > 0) {
          del_data <- data.frame(bin = names(detail$del.bins[[1]]), ratio = as.numeric(detail$del.bins[[1]]))
          del_out <- file.path(out_dir, paste0(epic_id, "_del_bins.tsv"))
          write.table(del_data, file = del_out, sep = "\t", row.names = FALSE, quote = FALSE)
          cat(paste("    Wrote deleted bins to", del_out, "\n"))
        }
        
        # Write amplified detail regions
        if (!is.null(detail$amp.detail.regions[[1]]) && length(detail$amp.detail.regions[[1]]) > 0) {
          amp_detail_data <- data.frame(region = names(detail$amp.detail.regions[[1]]), ratio = as.numeric(detail$amp.detail.regions[[1]]))
          amp_detail_out <- file.path(out_dir, paste0(epic_id, "_amp_detail_regions.tsv"))
          write.table(amp_detail_data, file = amp_detail_out, sep = "\t", row.names = FALSE, quote = FALSE)
          cat(paste("    Wrote amplified detail regions to", amp_detail_out, "\n"))
        }
        
        # Write deleted detail regions
        if (!is.null(detail$del.detail.regions[[1]]) && length(detail$del.detail.regions[[1]]) > 0) {
          del_detail_data <- data.frame(region = names(detail$del.detail.regions[[1]]), ratio = as.numeric(detail$del.detail.regions[[1]]))
          del_detail_out <- file.path(out_dir, paste0(epic_id, "_del_detail_regions.tsv"))
          write.table(del_detail_data, file = del_detail_out, sep = "\t", row.names = FALSE, quote = FALSE)
          cat(paste("    Wrote deleted detail regions to", del_detail_out, "\n"))
        }
        
      }, error = function(e) {
        cat(paste("    Warning: Failed to write detail components:", e$message, "\n"))
      })
      
    }, error = function(e) {
      cat(paste("  Error processing sample", epic_ids[i], ":", e$message, "\n"))
    })
  }
  
  # Generate summary plot and heatmap from all samples (if >1 sample)
  if (length(conumee_list) > 1) {
    cat("\nGenerating summary plots from all samples...\n")
    
    # Combine individual CNV.analysis objects step by step
    tryCatch({
      combined <- conumee_list[[1]]
      for (i in 2:length(conumee_list)) {
        combined <- CNV.combine(combined, conumee_list[[i]])
        cat(paste("  Combined", i, "of", length(conumee_list), "samples\n"))
      }
      cat(paste("Combined all", length(conumee_list), "CNV.analysis objects\n"))
      
      # Debug: check combined object structure
      cat(paste("DEBUG: combined object has", length(combined@seg$summary), "segment summaries\n"))
      cat(paste("DEBUG: combined @fit$ratio has", ncol(combined@fit$ratio), "samples\n"))
      cat(paste("DEBUG: seg$summary names:", paste(names(combined@seg$summary), collapse=", "), "\n"))
      
      # Fix: manually combine seg$summary which CNV.combine() doesn't handle
      # Rebuild seg$summary by combining all individual samples
      all_summaries <- list()
      for (j in seq_along(conumee_list)) {
        sample_name <- sample_keys[j]
        if (!is.null(conumee_list[[j]]@seg$summary[[1]])) {
          all_summaries[[sample_name]] <- conumee_list[[j]]@seg$summary[[1]]
        }
      }
      combined@seg$summary <- all_summaries
      cat(paste("DEBUG: After fix, combined@seg$summary has", length(combined@seg$summary), "entries\n"))
      cat(paste("DEBUG: Fixed seg$summary names:", paste(names(combined@seg$summary), collapse=", "), "\n"))
      
      # Fix: manually combine seg$p which CNV.combine() doesn't handle
      if (!is.null(combined@seg$p)) {
        all_p <- list()
        for (j in seq_along(conumee_list)) {
          sample_name <- sample_keys[j]
          if (!is.null(conumee_list[[j]]@seg$p[[1]])) {
            all_p[[sample_name]] <- conumee_list[[j]]@seg$p[[1]]
          }
        }
        combined@seg$p <- all_p
        cat(paste("DEBUG: Fixed seg$p with", length(combined@seg$p), "entries\n"))
      }
      
      # Fix: manually combine @bin slot which CNV.combine() doesn't handle
      # The @bin slot contains per-bin ratio/variance values needed by CNV.heatmap
      all_bin_ratios <- list()
      all_bin_variance <- list()
      all_shifts <- c()
      for (j in seq_along(conumee_list)) {
        sample_name <- sample_keys[j]
        if (!is.null(conumee_list[[j]]@bin$ratio[[1]])) {
          all_bin_ratios[[sample_name]] <- conumee_list[[j]]@bin$ratio[[1]]
          all_bin_variance[[sample_name]] <- conumee_list[[j]]@bin$variance[[1]]
          all_shifts[sample_name] <- conumee_list[[j]]@bin$shift[1]
        }
      }
      combined@bin$ratio <- all_bin_ratios
      combined@bin$variance <- all_bin_variance
      combined@bin$shift <- all_shifts
      cat(paste("DEBUG: Fixed @bin with", length(combined@bin$ratio), "sample ratios\n"))
      
      # Create epic subfolder in summary directory
      epic_summary_dir <- file.path(summary_dir, "epic")
      if (!dir.exists(epic_summary_dir)) {
        dir.create(epic_summary_dir, recursive = TRUE)
      }
      
      # CNV.summaryplot: shows consensus of alterations across all samples
      tryCatch({
        png_summary <- file.path(epic_summary_dir, "CNV_summaryplot.png")
        cat(paste("DEBUG: Before summaryplot - ncol(ratio)=", ncol(combined@fit$ratio), ", nrow(ratio)=", nrow(combined@fit$ratio), "\n"))
        cat(paste("DEBUG: Before summaryplot - Number of samples in seg$summary:", length(combined@seg$summary), "\n"))
        cat(paste("Generating summaryplot to", png_summary, "...\n"))
        png(png_summary, width = 12, height = 8, units = "in", res = 720, pointsize = 12)
        par(mar = c(4, 4, 4, 4), oma = c(0, 0, 0, 0))
        par(cex = 0.8, cex.axis = 1.0, cex.lab = 1.0)
        
        CNV.summaryplot(combined)
        
        dev.off()
        if (file.exists(png_summary)) {
          cat(paste("✓ Saved CNV summaryplot to", png_summary, "\n"))
        } else {
          cat(paste("✗ Failed to save summaryplot (file not created)\n"))
        }
        
      }, error = function(e) {
        cat(paste("✗ Error generating summaryplot:", e$message, "\n"))
      })
      
      # CNV.heatmap: shows copy-number heatmap across all samples
      tryCatch({
        png_heatmap <- file.path(epic_summary_dir, "CNV_heatmap.png")
        cat(paste("DEBUG: Before heatmap - Number of samples in combined@fit$ratio:", ncol(combined@fit$ratio), "\n"))
        cat(paste("Generating heatmap to", png_heatmap, "...\n"))
        png(png_heatmap, width = 14, height = 8, units = "in", res = 720, pointsize = 12)
        par(mar = c(4, 8, 4, 4), oma = c(0, 0, 0, 0))
        par(cex = 0.8, cex.axis = 1.0, cex.lab = 1.0)
        
        CNV.heatmap(combined)
        
        dev.off()
        if (file.exists(png_heatmap)) {
          cat(paste("✓ Saved CNV heatmap to", png_heatmap, "\n"))
        } else {
          cat(paste("✗ Failed to save heatmap (file not created)\n"))
        }
        
      }, error = function(e) {
        cat(paste("✗ Error generating heatmap:", e$message, "\n"))
      })
      
    }, error = function(e) {
      cat(paste("  Error combining CNV.analysis objects:", e$message, "\n"))
    })
  } else if (length(conumee_list) == 1) {
    cat("  Skipping summaryplot and heatmap (only 1 sample; need >1 for meaningful summary)\n")
  }
  
  cat("\nAll plots and per-sample outputs generated successfully.\n")
  
}, error = function(e) {
  cat(paste("Fatal error:", e$message, "\n"))
  quit(status=1)
})
