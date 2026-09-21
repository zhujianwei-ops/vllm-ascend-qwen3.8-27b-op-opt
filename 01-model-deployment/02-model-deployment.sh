#!/usr/bin/env bash

# Load model from ModelScope to speed up download
export ASCEND_RT_VISIBLE_DEVICES=14,15
export MODEL_PATH=/home1/model/Qwen3.8-27B-w8a8/
export VLLM_USE_MODELSCOPE=True
export HCCL_BUFFSIZE=512
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_CASE="${PROFILE_CASE:-baseline_single_request}"
PROFILE_DIR="${SCRIPT_DIR}/../04-operator-profiling-data-analysis/vllm_profile/${PROFILE_CASE}"
mkdir -p "$PROFILE_DIR"

# Size of the shared buffer (in MB) used by HCCL for NPU-to-NPU collective communication
# To reduce memory fragmentation and avoid out of memory

# Model weight path; can be a ModelScope model id (e.g., Eco-Tech/Qwen3.8-27B-w8a8) or a local directory path
# Ensure the model path matches the directory recorded during download

vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --data-parallel-size 1 \
    --tensor-parallel-size 2 \
    --quantization ascend \
    --served-model-name qwen3.8 \
    --max-num-seqs 32 \
    --max-model-len 131072 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.85 \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "'"$PROFILE_DIR"'", "torch_profiler_with_stack": false}' \
    --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true}'
