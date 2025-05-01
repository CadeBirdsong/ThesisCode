import glob, os, shutil

# Directory containing the uploaded .log files
DATA_DIR = r"C:\Users\Cade\RealFusionProject"

# Pattern for the fusion log files
log_pattern = os.path.join(DATA_DIR, "FusionTerms-2024-*.log")

# Gather and sort log file paths
log_files = sorted(glob.glob(log_pattern))

if not log_files:
    raise FileNotFoundError("No FusionTerms-2024-*.log files found in /mnt/data")

# Destination combined file
combined_path = os.path.join(DATA_DIR, "FusionTerms-2024.txt")

# Write/append each log's full content into the combined file
with open(combined_path, "w", encoding="utf-8") as out_f:
    for idx, log_file in enumerate(log_files, 1):
        with open(log_file, "r", encoding="utf-8") as in_f:
            out_f.write(in_f.read())
        # Separate logs with a newline marker for readability
        out_f.write(f"\n\n--- End of {os.path.basename(log_file)} ---\n\n")

combined_path
