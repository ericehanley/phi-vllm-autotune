#!/bin/bash
set -e

# Install required packages
echo "--- Installing required packages (bc, git, datamash) ---"
apt-get update && apt-get install -y bc git datamash
echo "--- Dependencies installed ---"

# --- Script Parameters ---
TAG=$(date +"%Y_%m_%d_%H_%M")
BASE="."
MODEL="microsoft/Phi-3-mini-128k-instruct"
TP=8
DOWNLOAD_DIR="model_dir"
INPUT_LEN=15000
OUTPUT_LEN=2048
NUM_MEASUREMENT_RUNS=20

# --- Parameter lists to iterate through ---
MIN_CACHE_LIST="0 0.10 0.25 0.501"
NUM_BATCHED_TOKENS_LIST="8192 16384"
GPU_UTIL_LIST="0.98"
NUM_SEQS_LIST="1"
REQUEST_RATE_LIST="1"
# ===============================================================

LOG_FOLDER="$BASE/auto-benchmark/$TAG"
RESULT="$LOG_FOLDER/result.txt"

echo "--- auto_tune_latency.sh ---"
echo "Result file will be located at: $RESULT"
echo "Model: $MODEL"

rm -rf $LOG_FOLDER
mkdir -p $LOG_FOLDER

chmod +x /app/measure_latency.py

best_latency_e2el=999999999
best_latency_config=""

# --- start_server() function (no changes) ---
start_server() {
    local gpu_memory_utilization=$1
    local max_num_seqs=$2
    local max_num_batched_tokens=$3
    local vllm_log=$4
    pkill -f vllm || true
    sleep 2
    VLLM_USE_V1=1 VLLM_SERVER_DEV_MODE=1 vllm serve $MODEL \
        --disable-log-requests --port 8004 --gpu-memory-utilization $gpu_memory_utilization \
        --max-num-seqs $max_num_seqs --max-num-batched-tokens $max_num_batched_tokens \
        --tensor-parallel-size $TP --enable-prefix-caching --load-format dummy \
        --download-dir "$DOWNLOAD_DIR" --max-model-len $(( INPUT_LEN+OUTPUT_LEN )) > "$vllm_log" 2>&1 &
    server_pid=$!
    echo "vLLM server started with PID $server_pid"
    for i in {1..60}; do
        RESPONSE=$(curl -s -X GET "http://0.0.0.0:8004/health" -w "%{http_code}" -o /dev/stdout || true)
        STATUS_CODE=$(echo "$RESPONSE" | tail -n 1)
        if [[ "$STATUS_CODE" -eq 200 ]]; then
            echo "Server is healthy."
            return 0
        else
            echo "Waiting for server... attempt $i/60"
            sleep 10
        fi
    done
    echo "Server did not start within 10 minutes. Check log at $vllm_log".
    return 1
}

# --- run_benchmark() function (Corrected Statistics Calculation) ---
run_benchmark() {
    local max_num_seqs=$1
    local max_num_batched_tokens=$2
    local gpu_memory_utilization=$3
    local request_rate=$4
    local min_cache_rate=$5

    echo
    printf '=%.0s' $(seq 1 80)
    echo -e "\nRunning test for: tokens=$max_num_batched_tokens, util=$gpu_memory_utilization, cache_rate=${min_cache_rate}"
    
    local vllm_log="$LOG_FOLDER/vllm_log_${max_num_batched_tokens}_${gpu_memory_utilization}_${min_cache_rate}.txt"
    
    echo "Starting server..."
    if ! start_server $gpu_memory_utilization $max_num_seqs $max_num_batched_tokens "$vllm_log"; then return 1; fi
    sleep 5

    prefix_len=$(printf "%.0f" $(echo "$INPUT_LEN * $min_cache_rate" | bc))
    
    if (( $(echo "$min_cache_rate == 0" | bc -l) )); then
        warmup_prompt=$(cat /dev/urandom | tr -dc 'A-Za-z0-9' | head -c $INPUT_LEN)
    else
        warmup_prompt=$(cat /dev/urandom | tr -dc 'A-Za-z0-9' | head -c $prefix_len)
    fi

    echo "Warming up cache with a prompt of length ${#warmup_prompt}..."
    python3 /app/measure_latency.py "$warmup_prompt" > /dev/null

    echo "Measuring latency with ${NUM_MEASUREMENT_RUNS} independent requests..."
    
    local latencies=()
    for (( i=1; i<=$NUM_MEASUREMENT_RUNS; i++ )); do
        if (( $(echo "$min_cache_rate == 0" | bc -l) )); then
             main_prompt=$(cat /dev/urandom | tr -dc 'A-Za-z0-9' | head -c $INPUT_LEN)
        else
            local suffix_len=$((INPUT_LEN - prefix_len))
            local suffix=$(cat /dev/urandom | tr -dc 'A-Za-z0-9' | head -c $suffix_len)
            main_prompt="${warmup_prompt}${suffix}"
        fi
        
        latency_ms=$(python3 /app/measure_latency.py "$main_prompt")
        echo "Run $i: $latency_ms ms"
        latencies+=($latency_ms)
    done

    # --- CORRECTED STATISTICS CALCULATION ---
    all_latencies=$(printf "%s\n" "${latencies[@]}")
    median_latency_ms=$(echo "$all_latencies" | datamash median 1)
    p99_latency_ms=$(echo "$all_latencies" | datamash perc 99 1)

    result_line="tokens: $max_num_batched_tokens, cache_rate: $min_cache_rate, median_latency: $median_latency_ms ms, p99_latency: $p99_latency_ms ms"
    echo "RESULT: $result_line"
    echo "$result_line" >> "$RESULT"

    if (( $(echo "$median_latency_ms < $best_latency_e2el" | bc -l) )); then
        echo "New best latency found!"
        best_latency_e2el=$median_latency_ms
        best_latency_config="$result_line"
    fi

    pkill -f vllm || true
    sleep 5
}

# --- Main Execution Logic (no changes) ---
read -r -a num_seqs_list <<< "$NUM_SEQS_LIST"
read -r -a num_batched_tokens_list <<< "$NUM_BATCHED_TOKENS_LIST"
read -r -a request_rate_list <<< "$REQUEST_RATE_LIST"
read -r -a gpu_util_list <<< "$GPU_UTIL_LIST"
read -r -a min_cache_list <<< "$MIN_CACHE_LIST"

for num_seqs in "${num_seqs_list[@]}"; do
    for num_batched_tokens in "${num_batched_tokens_list[@]}"; do
        for rate in "${request_rate_list[@]}"; do
            for gpu_memory_utilization in "${gpu_util_list[@]}"; do
                for min_cache in "${min_cache_list[@]}"; do
                    run_benchmark $num_seqs $num_batched_tokens $gpu_memory_utilization $rate $min_cache
                done
            done
        done
    done
done

echo
printf '*%.0s' $(seq 1 80)
echo -e "\n--- SCRIPT FINISHED ---"
echo
echo "Best Latency Configuration:"
echo "$best_latency_config"
printf '*%.0s' $(seq 1 80)
echo

echo "Full results available in: $RESULT"
