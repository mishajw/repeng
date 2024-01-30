#!/usr/bin/env bash

poetry run python -m repeng.datasets.activations \
    test_2024-01-30 \
    --llms pythia-tiny \
    --datasets common_sense_qa \
    --num_samples_per_dataset=10 \
    --num_validation_samples_per_dataset 10 \
    --device=cpu \
    --num_tokens_from_end 2 \
    --layers_start 0 \
    --layers_end 3

poetry run python -m repeng.datasets.activations \
    datasets_2024-01-31_tokensandlayers_v1 \
    --llms llama-13b-and-pythia-12b \
    --datasets small-varied \
    --num_samples_per_dataset=2000 \
    --num_validation_samples_per_dataset 200 \
    --device=gpu

# TODO: Update tokens & layers.
poetry run python -m repeng.datasets.activations \
    datasets_2024-01-31_datasets_v1 \
    --llms llama-13b-and-pythia-12b \
    --datasets all \
    --num_samples_per_dataset=2000 \
    --num_validation_samples_per_dataset 200 \
    --device=gpu \
    --num_tokens_from_end 2 \
    --layers_start 31 \
    --layers_end 31

# TODO: Update datasets & tokens.
poetry run python -m repeng.datasets.activations \
    datasets_2024-01-31_models_v1 \
    --llms pythia \
    --datasets small-varied \
    --num_samples_per_dataset=2000 \
    --num_validation_samples_per_dataset 200 \
    --device=gpu \
    --num_tokens_from_end 2
