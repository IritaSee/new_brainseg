from dataset import BraTSDataset2D
from modelling import SE2_CNNET
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm 

def run_test():
    """
    Fungsi untuk menjalankan tes singkat pada dataset dan model.
    """
    try:
        print("Menjalankan uji coba model dan data loader...")
        DATA_DIRECTORY = r"./DataBrainSeg/BraTS2020_TrainingData"
        train_dataset = BraTSDataset2D(data_dir=DATA_DIRECTORY)
        single_image, _ = train_dataset[0]

        # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        device = 'cpu'
        print(device)

        model = SE2_CNNET(n_channels=4, n_classes=4).to(device)

        # Menambahkan dimensi batch
        input_tensor = single_image.unsqueeze(0).to(device)

        print(f"Menggunakan device: {device}")
        print(f"Bentuk input tensor: {input_tensor.shape}")

        model.eval()
        with torch.no_grad():
            output = model(input_tensor)

        print(f"Bentuk output tensor: {output.shape}")
        print("\nProses SE2-CNN selesai tanpa error.")

    except Exception as e:
        print(f"\nTerjadi error: {e}")
        traceback.print_exc()