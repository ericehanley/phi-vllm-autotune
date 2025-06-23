import os
import re
import csv
import argparse
import glob

def parse_filename(filename):
    """
    Parses benchmark parameters from a log file's name, handling multiple formats.
    """
    base_name = os.path.basename(filename)
    
    # --- Pattern 1: New, more detailed format ---
    # Example: bm_log_16_4096_1_0.92_0.10.txt
    pattern_long = r"bm_log_(\d+)_(\d+)_([\d\w]+)_([\d.]+)_([\d.]+)\.txt"
    match = re.match(pattern_long, base_name)
    if match:
        return {
            "max_num_seqs": int(match.group(1)),
            "max_num_batched_tokens": int(match.group(2)),
            "request_rate": match.group(3), # Can be 'inf'
            "gpu_memory_utilization": float(match.group(4)),
            "min_cache_hit_pct": int(float(match.group(5)) * 100),
        }

    # --- Pattern 2: Older, simpler format ---
    # Example: bm_log_64_4096_1.txt
    pattern_short = r"bm_log_(\d+)_(\d+)_([\d\w]+)\.txt"
    match = re.match(pattern_short, base_name)
    if match:
        return {
            "max_num_seqs": int(match.group(1)),
            "max_num_batched_tokens": int(match.group(2)),
            "request_rate": match.group(3),
            "gpu_memory_utilization": 0.90,  # Assign default value
            "min_cache_hit_pct": 0,       # Assign default value
        }
        
    # If neither pattern matches, return None
    return None

def parse_log_content(filepath):
    """
    Parses performance metrics from the content of a log file.
    """
    metrics = {}
    
    patterns = {
        "request_throughput_req_s": r"Request throughput \(req/s\):\s*([\d\.]+)",
        "output_token_throughput_tok_s": r"Output token throughput \(tok/s\):\s*([\d\.]+)",
        "p99_ttft_ms": r"P99 TTFT \(ms\):\s*([\d\.]+)",
        "median_ttft_ms": r"Median TTFT \(ms\):\s*([\d\.]+)",
        "p99_tpot_ms": r"P99 TPOT \(ms\):\s*([\d\.]+)",
        "median_tpot_ms": r"Median TPOT \(ms\):\s*([\d\.]+)",
        "p99_e2el_ms": r"P99 E2EL \(ms\):\s*([\d\.]+)",
        "median_e2el_ms": r"Median E2EL \(ms\):\s*([\d\.]+)"
    }
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    metrics[key] = float(match.group(1))
                else:
                    metrics[key] = "N/A" # Use "N/A" for missing values
    except Exception as e:
        print(f"Error reading or parsing file {filepath}: {e}")
        return None
        
    return metrics

def main():
    """
    Main function to find log files, parse them, and write to a CSV.
    """
    parser = argparse.ArgumentParser(description="Parse vLLM benchmark logs and create a CSV summary.")
    parser.add_argument("log_directory", type=str, help="The directory containing the benchmark log files.")
    args = parser.parse_args()

    log_directory = args.log_directory
    if not os.path.isdir(log_directory):
        print(f"Error: Directory not found at '{log_directory}'")
        return

    log_files = glob.glob(os.path.join(log_directory, "bm_log_*.txt"))
    if not log_files:
        print(f"No benchmark log files found in '{log_directory}'.")
        return

    all_results = []
    print(f"Found {len(log_files)} log files to process...")

    for log_file in log_files:
        params = parse_filename(log_file)
        if not params:
            print(f"Skipping file with unexpected name format: {log_file}")
            continue

        metrics = parse_log_content(log_file)
        if not metrics:
            continue
        
        combined_data = {**params, **metrics}
        all_results.append(combined_data)

    if not all_results:
        print("No valid data was parsed.")
        return

    csv_output_path = os.path.join(log_directory, "benchmark_summary.csv")
    
    headers = [
        "max_num_seqs", "max_num_batched_tokens", "request_rate", 
        "gpu_memory_utilization", "min_cache_hit_pct", "request_throughput_req_s", 
        "output_token_throughput_tok_s", "p99_e2el_ms", "median_e2el_ms",
        "p99_ttft_ms", "median_ttft_ms", "p99_tpot_ms", "median_tpot_ms"
    ]

    try:
        with open(csv_output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nSuccessfully created summary file: {csv_output_path}")
    except Exception as e:
        print(f"Error writing to CSV file: {e}")

if __name__ == "__main__":
    main()
