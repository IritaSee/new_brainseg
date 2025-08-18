import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import nibabel as nib
import os

# Dataset untuk memuat slice 2D tengah dari dataset BraTS2020. 
# Setiap item mengembalikan 4 channel MRI (flair, t1, t1ce, t2) dan mask segmentasinya.
class BraTSDataset2D(Dataset):
    def __init__(self, data_dir):
        self.base_path = os.path.join(data_dir, 'MICCAI_BraTS2020_TrainingData')
        if not os.path.isdir(self.base_path):
            raise FileNotFoundError(f"Direktori tidak ditemukan di: '{self.base_path}'")
        self.patient_dirs = [p for p in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, p))]

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        patient_dir_name = self.patient_dirs[idx]
        patient_path = os.path.join(self.base_path, patient_dir_name)

        # Path untuk setiap modalitas citra
        flair_path = os.path.join(patient_path, f"{patient_dir_name}_flair.nii")
        t1_path = os.path.join(patient_path, f"{patient_dir_name}_t1.nii")
        t1ce_path = os.path.join(patient_path, f"{patient_dir_name}_t1ce.nii")
        t2_path = os.path.join(patient_path, f"{patient_dir_name}_t2.nii")
        seg_path = os.path.join(patient_path, f"{patient_dir_name}_seg.nii")

        # Memuat data citra
        flair_img = nib.load(flair_path).get_fdata()
        t1_img = nib.load(t1_path).get_fdata()
        t1ce_img = nib.load(t1ce_path).get_fdata()
        t2_img = nib.load(t2_path).get_fdata()
        seg_mask = nib.load(seg_path).get_fdata()

        # Mengambil slice tengah dari volume 3D
        mid_slice_idx = flair_img.shape[2] // 2
        image_stack = np.stack([
            flair_img[:, :, mid_slice_idx],
            t1_img[:, :, mid_slice_idx],
            t1ce_img[:, :, mid_slice_idx],
            t2_img[:, :, mid_slice_idx]
        ], axis=0)
        seg_slice = seg_mask[:, :, mid_slice_idx]

        # # Konversi ke tensor PyTorch
        # image_tensor = torch.tensor(image_stack, dtype=torch.float32)
        # label_tensor = torch.tensor(seg_slice, dtype=torch.long)
        # label_tensor[label_tensor == 4] = 3

        # Konversi ke tensor PyTorch
        image_tensor = torch.tensor(image_stack, dtype=torch.float32)
        label_tensor = torch.tensor(seg_slice, dtype=torch.long)
        
        # Ganti label 4 menjadi 3 menggunakan masked_fill_ dengan mask boolean
        label_tensor.masked_fill_(label_tensor == 4, 3)
        
        # Normalisasi sederhana per channel
        for i in range(image_tensor.shape[0]):
            max_val = torch.max(image_tensor[i])
            if max_val > 0:
                image_tensor[i] = image_tensor[i] / max_val

        return image_tensor, label_tensor