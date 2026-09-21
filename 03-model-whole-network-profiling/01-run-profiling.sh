#!/usr/bin/env bash

set -Eeuo pipefail

SERVER_URL="http://localhost:8000"
MODEL_NAME="qwen3.8"
PROFILE_CASE="${PROFILE_CASE:-baseline_single_request}"

echo "Starting profiling..."
curl --fail --silent --show-error --request POST "${SERVER_URL}/start_profile"
printf '\n'

echo "Sending request..."
curl --fail --silent --show-error "${SERVER_URL}/v1/completions" \
    --header "Content-Type: application/json" \
    --data @- <<EOF
{
  "model": "${MODEL_NAME}",
  "prompt": "San Francisco is a",
  "max_tokens": 7,
  "temperature": 0
}
EOF
printf '\n'

echo "Stopping profiling..."
curl --fail --silent --show-error --request POST "${SERVER_URL}/stop_profile"
printf '\n'

echo "Profiling complete. Data is saved under 04-operator-profiling-data-analysis/vllm_profile/${PROFILE_CASE}."
