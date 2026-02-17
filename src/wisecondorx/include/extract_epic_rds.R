options(warn=1)

suppressMessages(library("jsonlite"))
suppressMessages(library("conumee2"))

args <- commandArgs(T)
in.file <- paste0(args[which(args == "--infile") + 1])
input <- read_json(in.file, na="string")

rds_path <- input$rds_path
out_bins_path <- input$out_bins_path
out_segments_path <- input$out_segments_path

# Read RDS file
x <- readRDS(rds_path)

# Extract bins
bins <- CNV.write(x[1], what = "bins")
write.table(bins, 
            file = out_bins_path,
            sep = "\t", 
            row.names = FALSE, 
            quote = FALSE)

# Extract segments
segments <- CNV.write(x[1], what = "segments")
write.table(segments, 
            file = out_segments_path,
            sep = "\t", 
            row.names = FALSE, 
            quote = FALSE)

message("Extracted bins and segments from RDS file")
