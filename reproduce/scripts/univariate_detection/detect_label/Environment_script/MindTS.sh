#!/bin/bash
# Environment benchmark, adapted for this device: runs only on GPU 1.
# All results are written inside the reproduce folder.

# The original project root, where the ts_benchmark package, config and dataset live
PROJECT_ROOT="/home/robin/project/MindTS_0"

# Run from the reproduce folder so ./scripts/run_benchmark.py resolves to the copied scripts
cd "$(dirname "$0")/../../../.."

# Use the conda environment that has the required dependencies (torch 2.4.1+cu121, ray, ...)
PYTHON="/home/robin/miniconda3/envs/MindTS/bin/python"

# Restrict execution to GPU 1 only
export CUDA_VISIBLE_DEVICES=1

# Reduce allocator fragmentation on the 20 GiB card
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# batch_size reduced from 64 to 8: the MindTS model computes full-vocabulary
# LM-head logits over the 1024-token text input (55+ GiB at batch 64), which
# does not fit in this device's 20 GiB GPU
PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" "$PYTHON" ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-name-list "Environment.csv" --model-name "MindTS.MindTS" --model-hyper-params '{"batch_size": 8, "d_ff": 64, "d_model": 64, "e_layers": 1, "horizon": 0, "norm": true, "num_epochs": 1, "seq_len": 72, "patch_size": 6, "stride": 6, "mask_ratio": 0.4, "r":0.9, "enc_in_time": 1, "parallel_strategy": "DP"}' --gpus 1 --num-workers 1 --timeout 60000 --save-path "$(pwd)/result/label/Environment" --text-name-list "Environment_text.csv"
