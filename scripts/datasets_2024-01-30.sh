#!/usr/bin/env bash

/opt/conda/bin/python -m repeng.datasets.activations \
    datasets_2024-02-05_v3_chat \
    --llms Llama-2-7b-chat-hf \
    --datasets all \
    --num_samples_per_dataset 800 \
    --num_validation_samples_per_dataset 200 \
    --num_tokens_from_end 1 \
    --device cuda

/opt/conda/bin/python -m repeng.datasets.activations \
    datasets_2024-02-05_v3 \
    --llms Llama-2-7b-hf \
    --datasets all \
    --num_samples_per_dataset 800 \
    --num_validation_samples_per_dataset 200 \
    --num_tokens_from_end 1 \
    --device cuda
