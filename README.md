# Brain Segmentation Project

This project focuses on brain segmentation using the BraTS2020 dataset with PyTorch and E2CNN.

## Setup with UV

This project uses [UV](https://github.com/astral-sh/uv) for Python package management, replacing the previous conda-based workflow. UV is a fast, reliable Python package installer and resolver.

### Initial Setup

1. Run the setup script to install UV and create a virtual environment with Python 3.9:

```bash
./setup_uv.sh
```

2. Activate the virtual environment:

```bash
source .venv/bin/activate
```

### Manual Setup (if script doesn't work)

1. Install UV:

```bash
curl -sSf https://astral.sh/uv/install.sh | sh
```

2. Create a virtual environment with Python 3.9:

```bash
uv venv --python=3.9
```

3. Activate the virtual environment:

```bash
source .venv/bin/activate
```

4. Install e2cnn:

```bash
uv pip install e2cnn
```

5. Install PyTorch with CUDA support:

```bash
uv pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

6. Install other dependencies:

```bash
uv pip install nibabel matplotlib pandas numpy jupyter tqdm
```

## Running on GPU

To run the training on a GPU:

1. Make sure your environment is activated:

```bash
source .venv/bin/activate
```

2. Run the GPU script:

```bash
./run_gpu.sh
```

## Project Structure

- `dataset.py`: Contains the BraTSDataset2D class for data loading
- `modelling.py`: Contains the SE2_CNNET model architecture
- `run_and_train.py`: Main script for training the model
- `checking.py`: Utility functions for testing
- `BrainSeg.ipynb`: Jupyter notebook with the full pipeline

## Remote GPU Training

For training on a remote GPU server:

1. Transfer the project files to the remote server
2. Install UV on the remote server and set up the environment using the same scripts
3. Adjust the `CUDA_VISIBLE_DEVICES` variable in `run_gpu.sh` to select the appropriate GPU
4. Run the training script

## Requirements

The project uses the following key dependencies:
- Python 3.9
- PyTorch 1.13.1 with CUDA 11.7
- e2cnn
- nibabel
- matplotlib
- pandas
- tqdm
- jupyter
