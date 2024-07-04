#!/bin/bash

### CIFAR 10
python3 scripts/development/fpt_experiment.py \
    --config configs/frozen_pretrained_transformer/gpt2_med_fpt.yaml \
    --dataset CIFAR10 \
    --input_dim 48 \
    --output_dim 10 \
    --no-use_embeddings_for_input \
    --gpu_id 0

### MNIST
python3 scripts/development/fpt_experiment.py \
    --config configs/frozen_pretrained_transformer/gpt2_med_fpt.yaml \
    --dataset MNIST \
    --input_dim 16 \
    --output_dim 10 \
    --no-use_embeddings_for_input \
    --gpu_id 0

### BitMemory
python3 scripts/development/fpt_experiment.py \
    --config configs/frozen_pretrained_transformer/gpt2_med_fpt.yaml \
    --dataset BitMemory \
    --input_dim 50 \
    --output_dim 100 \
    --no-use_embeddings_for_input \
    --gpu_id 0

### IMDB
python3 scripts/development/fpt_experiment.py \
    --config configs/frozen_pretrained_transformer/gpt2_med_fpt.yaml \
    --dataset IMDB \
    --input_dim 50257 \
    --output_dim 2 \
    --use_embeddings_for_input \
    --gpu_id 0
    # --input_dim 32128 \
