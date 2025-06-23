import requests
import time
import json
import sys

# --- Configuration ---
URL = "http://127.0.0.1:8004/v1/completions"
OUTPUT_LEN = 2048

# --- Validate and Get Prompt from Command-Line Argument ---
if len(sys.argv) < 2:
    print("-1.00", file=sys.stderr)
    sys.exit(1)

prompt = sys.argv[1]

# --- Request Payload ---
payload = {
    "prompt": prompt,
    "max_tokens": OUTPUT_LEN,
    "temperature": 0,
    "stream": False
}

headers = {"Content-Type": "application/json"}

# --- Send Request and Measure Time ---
try:
    start_time = time.perf_counter()
    response = requests.post(URL, json=payload, headers=headers)
    end_time = time.perf_counter()

    if response.status_code == 200:
        latency_ms = (end_time - start_time) * 1000
        print(f"{latency_ms:.2f}")
    else:
        print("-1.00")

except requests.exceptions.RequestException:
    print("-1.00")
