!/bin/bash
set -e

# --- Pre-download model files to avoid network timeouts ---
echo "--- Pre-downloading model configuration and tokenizer files... ---"
huggingface-cli download microsoft/Phi-3-mini-128k-instruct \
    --local-dir model_dir --local-dir-use-symlinks False
echo "--- Model files downloaded successfully. ---"

# Install required packages
echo "--- Installing required packages (bc, git, datasets) ---"
apt-get update && apt-get install -y bc git
pip install -q datasets
echo "--- Dependencies installed ---"

# --- Script Parameters ---
TAG=$(date +"%Y_%m_%d_%H_%M")
BASE="."
MODEL="microsoft/Phi-3-mini-128k-instruct"
TP=8
DOWNLOAD_DIR="model_dir"
INPUT_LEN=15000
OUTPUT_LEN=2048
MIN_CACHE_HIT_PCT=0
MAX_LATENCY_ALLOWED_MS=6000

# Parameter lists to iterate through
NUM_SEQS_LIST="8 12 16 32"
NUM_BATCHED_TOKENS_LIST="8192 16384"
GPU_UTIL_LIST="0.98"
MIN_CACHE_LIST="0.501"
REQUEST_RATE_LIST="1"
# ===============================================================

LOG_FOLDER="$BASE/auto-benchmark/$TAG"
RESULT="$LOG_FOLDER/result.txt"

echo "result file: $RESULT"
echo "model: $MODEL"

rm -rf $LOG_FOLDER
mkdir -p $LOG_FOLDER

cd /app

current_hash=$(git rev-parse HEAD 2>/dev/null) || current_hash="N/A"
echo "hash:$current_hash" >> "$RESULT"
echo "current_hash: $current_hash"

# Variables to track the best results
best_throughput=0
best_throughput_config=""
best_latency_e2el=999999999
best_latency_config=""

start_server() {
    local gpu_memory_utilization=$1
    local max_num_seqs=$2
    local max_num_batched_tokens=$3
    local vllm_log=$4

    pkill -f vllm || true

    VLLM_USE_V1=1 VLLM_SERVER_DEV_MODE=1 vllm serve $MODEL \
        --disable-log-requests --port 8004 --gpu-memory-utilization $gpu_memory_utilization \
        --max-num-seqs $max_num_seqs --max-num-batched-tokens $max_num_batched_tokens \
        --tensor-parallel-size $TP --enable-prefix-caching --load-format dummy \
        --download-dir "$DOWNLOAD_DIR" --max-model-len $(( INPUT_LEN+OUTPUT_LEN )) > "$vllm_log" 2>&1 &

    server_started=0
    for i in {1..60}; do
        RESPONSE=$(curl -s -X GET "http://0.0.0.0:8004/health" -w "%{http_code}" -o /dev/stdout || true)
        STATUS_CODE=$(echo "$RESPONSE" | tail -n 1)
        if [[ "$STATUS_CODE" -eq 200 ]]; then
            server_started=1
            break
        else
            echo "Waiting for server... attempt $i/60"
            sleep 10
        fi
    done
    if (( ! server_started )); then
        echo "Server did not start within 10 minutes. Check log at $vllm_log".
        return 1
    else
        return 0
    fi
}

run_benchmark() {
    local max_num_seqs=$1
    local max_num_batched_tokens=$2
    local gpu_memory_utilization=$3
    local request_rate=$4
    local min_cache_rate=$5

    echo
    printf '=%.0s' $(seq 1 80)
    echo -e "\nRunning test for: max_num_seqs=$max_num_seqs, max_num_batched_tokens=$max_num_batched_tokens, request_rate=$request_rate req/s, gpu_memory_utilization=$gpu_memory_utilization, min_cache_rate=$min_cache_rate"
    
    local vllm_log="$LOG_FOLDER/vllm_log_${max_num_seqs}_${max_num_batched_tokens}_${request_rate}_${gpu_memory_utilization}_${min_cache_rate}.txt"
    local bm_log="$LOG_FOLDER/bm_log_${max_num_seqs}_${max_num_batched_tokens}_${request_rate}_${gpu_memory_utilization}_${min_cache_rate}.txt"
    
    echo "starting server..."
    if ! start_server $gpu_memory_utilization $max_num_seqs $max_num_batched_tokens "$vllm_log"; then
        echo "Server failed to start. Skipping this combination."
        return 1
    fi
    echo "server started."

    echo "Running benchmark..."
    prefix_len=$(printf "%.0f" $(echo "$INPUT_LEN * $min_cache_rate" | bc))
    python3 /vllm-workspace/benchmarks/benchmark_serving.py \
        --backend vllm --model $MODEL --dataset-name random \
        --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
        --ignore-eos --disable-tqdm --request-rate $request_rate \
        --percentile-metrics ttft,tpot,itl,e2el --num-prompts 200 \
        --random-prefix-len $prefix_len --port 8004 &> "$bm_log"

    # Parse all the relevant metrics
    throughput=$(grep "Request throughput (req/s):" "$bm_log" | sed 's/[^0-9.]//g')
    p99_e2el=$(grep "P99 E2EL (ms):" "$bm_log" | awk '{print $NF}')
    median_e2el=$(grep "Median E2EL (ms):" "$bm_log" | awk '{print $NF}') # New metric

    # Check if any results were produced
    if [[ -z "$throughput" || -z "$p99_e2el" || -z "$median_e2el" ]]; then
        echo "Benchmark did not produce valid results. See log: $bm_log"
        pkill -f vllm || true
        return 1
    fi

    # Record results
    result_line="seqs: $max_num_seqs, tokens: $max_num_batched_tokens, rate: $request_rate, throughput: $throughput, P99 E2EL: $p99_e2el, Median E2EL: $median_e2el"
    echo "RESULT: $result_line"
    echo "$result_line" >> "$RESULT"

    # Check for best throughput (if latency constraint is met)
    if (( $(echo "$p99_e2el <= $MAX_LATENCY_ALLOWED_MS" | bc -l) )); then
        if (( $(echo "$throughput > $best_throughput" | bc -l) )); then
            echo "New best throughput found!"
            best_throughput=$throughput
            best_throughput_config="$result_line"
        fi
    else
        echo "Latency requirement not met for this run."
    fi

    # Check for best latency
    if (( $(echo "$median_e2el < $best_latency_e2el" | bc -l) )); then
        echo "New best latency found!"
        best_latency_e2el=$median_e2el
        best_latency_config="$result_line"
    fi

    pkill -f vllm || true
    sleep 5
}

# --- Main Execution Logic ---
# gpu_memory_utilization=0.90

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
echo "Best Throughput Configuration (under ${MAX_LATENCY_ALLOWED_MS}ms P99 E2E Latency):"
echo "$best_throughput_config"
echo
echo "Best Latency Configuration (lowest Median E2E Latency):"
echo "$best_latency_config"
printf '*%.0s' $(seq 1 80)
echo

echo "Full results available in: $RESULT"
