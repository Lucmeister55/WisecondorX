options(warn=1)

suppressMessages(library("jsonlite"))
suppressMessages(library("minfi"))
suppressMessages(library("conumee2"))

args <- commandArgs(T)
in.file <- paste0(args[which(args == "--infile") + 1])
input <- read_json(in.file, na="string")

idat_basenames <- unlist(input$idat_basenames)
binsize <- as.integer(input$binsize)
out_bins <- input$out_bins
out_segments <- input$out_segments
out_dir <- input$out_dir

query_dir <- input$query_dir
ref_dir <- input$ref_dir
detail_regions_file <- input$detail_regions_file
exclude_regions_file <- input$exclude_regions_file
max_query <- input$max_query
max_ref <- input$max_ref
genome <- input$genome
if (is.null(genome) || !genome %in% c("hg19", "hg38")) genome <- "hg19"

is_missing_path <- function(x) {
  is.null(x) || !nzchar(x) || x %in% c("NULL", "None", "NA", "null")
}

# Resolve basenames function
resolve_idat_basenames <- function(input_dir, max_samples = NULL) {
  if (is_missing_path(input_dir) || !dir.exists(input_dir)) {
    stop("Provide a valid input_dir: ", input_dir)
  }

  grn <- list.files(input_dir, pattern = "_Grn\\.idat(\\.gz)?$", full.names = TRUE, recursive = TRUE)
  red <- list.files(input_dir, pattern = "_Red\\.idat(\\.gz)?$", full.names = TRUE, recursive = TRUE)
  basenames <- unique(c(sub("_Grn\\.idat(\\.gz)?$", "", grn), sub("_Red\\.idat(\\.gz)?$", "", red)))
  basenames <- basenames[nzchar(basenames)]

  if (length(basenames) == 0) {
    stop("No IDAT files found in input_dir: ", input_dir)
  }

  if (!is.null(max_samples) && !is.na(max_samples) && max_samples > 0 && length(basenames) > max_samples) {
    basenames <- basenames[seq_len(max_samples)]
  }

  return(basenames)
}

# Load query samples from query_dir
if (is_missing_path(query_dir)) {
  stop("Query directory is required")
}
query_basenames <- resolve_idat_basenames(query_dir, max_query)
message("Found ", length(query_basenames), " query samples in ", query_dir)

# Load reference samples from ref_dir
if (is_missing_path(ref_dir)) {
  stop("Reference directory is required")
}
ref_basenames <- resolve_idat_basenames(ref_dir, max_ref)
message("Loading ", length(ref_basenames), " reference samples from ", ref_dir)

# Load reference data using minfi
RGset.ref <- read.metharray(ref_basenames, verbose = FALSE)
Mset.ref <- preprocessRaw(RGset.ref)
data.ref <- CNV.load(Mset.ref)
message("Reference data loaded successfully")

# Load detail regions
if (!is_missing_path(detail_regions_file) && file.exists(detail_regions_file)) {
  load(detail_regions_file)
  if (exists("annoXY")) {
    detail_regions <- annoXY@detail
    message("Loaded detail regions from annoXY object")
  } else if (exists("detail_regions")) {
    message("Loaded detail_regions object directly")
  } else {
    stop("Neither annoXY nor detail_regions found in detail_regions_file")
  }
} else {
  stop("Detail regions file is required and must exist")
}

# Load exclude regions
if (!is_missing_path(exclude_regions_file) && file.exists(exclude_regions_file)) {
  load(exclude_regions_file)
  if (!exists("exclude_regions")) {
    stop("exclude_regions not found in exclude_regions_file")
  }
  message("Loaded exclude regions")
} else {
  stop("Exclude regions file is required and must exist")
}

# Create annotation
anno_new <- CNV.create_anno(array_type = c("450k", "EPIC"), exclude_regions = exclude_regions, detail_regions = detail_regions, genome = genome)
message("Annotation created successfully")

# Process samples
process_sample <- function(idat_basename, out_dir) {
  message("Processing: ", basename(idat_basename))
  
  # Load query sample
  rgset <- read.metharray(basenames = idat_basename, verbose = FALSE)
  mset <- preprocessRaw(rgset)
  data.q <- CNV.load(mset)

  # Run CNV analysis
  x <- CNV.fit(query = data.q, ref = data.ref, anno = anno_new)
  x <- CNV.bin(x)
  x <- CNV.detail(x)
  x <- CNV.segment(x)
  x <- CNV.focal(x)

  # Save RDS file
  sample_id <- basename(idat_basename)
  rds_path <- file.path(out_dir, paste0(sample_id, ".rds"))
  saveRDS(x, file = rds_path)
  
  message("Completed: ", basename(idat_basename), " -> ", rds_path)
}

if (length(query_basenames) == 0) {
  stop("No query samples to process")
}

if (!is.null(out_dir) && nzchar(out_dir)) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  for (idat_basename in query_basenames) {
    process_sample(idat_basename, out_dir)
  }
} else {
  stop("No output directory provided")
}

message("All samples processed successfully")
