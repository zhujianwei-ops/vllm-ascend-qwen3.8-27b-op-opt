#!/usr/bin/env bash

set -Eeuo pipefail

# Start the profiler before sending exactly one batch of 32 requests.
# The server must be started with PROFILE_CASE=bs32_prefill.

SERVER_URL="http://127.0.0.1:8000"
MODEL_NAME="qwen3.8"

echo "Starting profiling for bs32 prefill..."
curl --fail --silent --show-error --request POST "${SERVER_URL}/start_profile"
printf '\n'

echo "Sending 32 requests: input=8192 tokens, output=1 token..."
vllm bench serve \
    --backend openai \
    --base-url "${SERVER_URL}" \
    --endpoint /v1/completions \
    --model /home1/model/Qwen3.8-27B-w8a8/ \
    --served-model-name "${MODEL_NAME}" \
    --dataset-name random \
    --random-input-len 8192 \
    --random-output-len 1 \
    --random-range-ratio 0 \
    --ignore-eos \
    --num-prompts 32 \
    --num-warmups 0 \
    --max-concurrency 32 \
    --request-rate inf \
    --seed 20260921 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99

echo "Stopping profiling..."
curl --fail --silent --show-error --request POST "${SERVER_URL}/stop_profile"
printf '\n'

echo "Profiling complete. Data is saved under 04-operator-profiling-data-analysis/vllm_profile/bs32_prefill."
