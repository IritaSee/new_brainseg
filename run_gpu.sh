#!/bin/bash

# Script to run training on GPU
# Created: August 20, 2025

# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'Number of GPUs: {torch.cuda.device_count()}')"
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Set which GPU to use (uncomment and set the GPU index if needed)
# export CUDA_VISIBLE_DEVICES=0

# Run the training script
python run_and_train.py
