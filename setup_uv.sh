#!/bin/bash

# Script to set up UV for library versioning and GPU training
# Created: August 20, 2025

# Install UV if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add UV to path if not automatically added
    if ! grep -q "uv" ~/.zshrc; then
        echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc
        source ~/.zshrc
    fi
else
    echo "UV is already installed."
fi

# Create a new virtual environment with UV using Python 3.9
echo "Creating a new virtual environment with UV using Python 3.9..."
uv venv --python=3.9

# Activate the virtual environment
echo "Activating the virtual environment..."
source .venv/bin/activate

# Install e2cnn first
echo "Installing e2cnn..."
uv pip install e2cnn

# Install PyTorch with CUDA support
echo "Installing PyTorch 1.13.1 with CUDA 11.7 support..."
uv pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Install other dependencies
echo "Installing other dependencies..."
uv pip install nibabel matplotlib pandas numpy jupyter tqdm

# Verify installations
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import e2cnn; print(f'e2cnn imported successfully')"
python -c "import nibabel; print(f'nibabel imported successfully')"

echo "Setup completed successfully!"
echo "To activate the environment, run: source .venv/bin/activate"
echo "For GPU training, remember to set the CUDA device in your code."
