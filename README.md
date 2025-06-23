### Benchmarking phi-3-mini-128k-instruct on vLLM
Set up infrastructure on Google Cloud with environment_setup.sh
Update auto_tune with desired parameter space -- (this file has been substantially updated from the original in vllm repo: https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune.sh).
Update auto_tune_latency.sh to determine latency floor for model.
