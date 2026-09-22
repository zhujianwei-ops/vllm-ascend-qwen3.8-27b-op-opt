#!/usr/bin/env bash

vllm bench serve \
    --backend openai \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/completions \
    --model /home1/model/Qwen3.8-27B-w8a8/ \
    --served-model-name qwen3.8 \
    --dataset-name random \
    --random-input-len 8192 \
    --random-output-len 128 \
    --random-range-ratio 0 \
    --ignore-eos \
    --num-prompts 160 \
    --num-warmups 16 \
    --max-concurrency 16 \
    --request-rate inf \
    --seed 20260921 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99
