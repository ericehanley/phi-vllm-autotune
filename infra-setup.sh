# --- Configuration ---
# IMPORTANT: Replace 'your-gcp-project-id' with your actual Google Cloud Project ID.
export PROJECT_ID="northam-ce-mlai-tpu"
export INSTANCE_NAME="vllm-b200-spot"
export ZONE="us-central1-b" # Or any zone with A3 instance availability

# --- Create the VM Instance ---
# A3 High - 8 GPU
gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --image-family="common-cu124-ubuntu-2204" \
    --image-project=deeplearning-platform-release \
    --machine-type="a3-highgpu-8g" \
    --accelerator-type="nvidia-h100-80g,count=8"
    --maintenance-policy=TERMINATE \
    --boot-disk-size="200GB" \
    --metadata="install-nvidia-driver=True" \
    --provisioning-model=SPOT

# OR

# A3 Ultra - 8 GPU
gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --image-family="common-cu124-ubuntu-2204" \
    --image-project=deeplearning-platform-release \
    --machine-type="a3-ultragpu-8g" \
    --maintenance-policy=TERMINATE \
    --boot-disk-size="200GB" \
    --metadata="install-nvidia-driver=True" \
    --provisioning-model=SPOT

# Benchmarking after SSH into VM
docker run --gpus all -it --rm \
  -v "$(pwd)":/app \
  --entrypoint "" \
  vllm/vllm-openai:latest \
  /bin/bash

huggingface-cli download microsoft/Phi-3-mini-128k-instruct \
    --local-dir /app/model_dir --local-dir-use-symlinks False

docker run \
    --gpus all \
    --shm-size=16g \
    -d \
    --name vllm-benchmark \
    --workdir /app \
    -v "$(pwd)":/app \
    --entrypoint "" \
    vllm/vllm-openai:latest \
    bash /app/auto_tune_latency.sh