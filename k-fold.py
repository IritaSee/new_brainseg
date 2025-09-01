import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import nibabel as nib
import os
from sklearn.model_selection import KFold
# Pustaka khusus untuk E2CNN
from e2cnn import gspaces
import e2cnn.nn as enn

from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold
import torch
import torch.nn as nn


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

        flair_path = os.path.join(patient_path, f"{patient_dir_name}_flair.nii")
        t1_path    = os.path.join(patient_path, f"{patient_dir_name}_t1.nii")
        t1ce_path  = os.path.join(patient_path, f"{patient_dir_name}_t1ce.nii")
        t2_path    = os.path.join(patient_path, f"{patient_dir_name}_t2.nii")
        seg_path   = os.path.join(patient_path, f"{patient_dir_name}_seg.nii")

        flair_img = nib.load(flair_path).get_fdata()
        t1_img    = nib.load(t1_path).get_fdata()
        t1ce_img  = nib.load(t1ce_path).get_fdata()
        t2_img    = nib.load(t2_path).get_fdata()
        seg_mask  = nib.load(seg_path).get_fdata()

        mid_slice_idx = flair_img.shape[2] // 2
        image_stack = np.stack([
            flair_img[:, :, mid_slice_idx],
            t1_img[:, :, mid_slice_idx],
            t1ce_img[:, :, mid_slice_idx],
            t2_img[:, :, mid_slice_idx]
        ], axis=0)
        seg_slice = seg_mask[:, :, mid_slice_idx]

        image_tensor = torch.tensor(image_stack, dtype=torch.float32)
        label_tensor = torch.tensor(seg_slice, dtype=torch.long)
        label_tensor.masked_fill_(label_tensor == 4, 3)

        for i in range(image_tensor.shape[0]):
            max_val = torch.max(image_tensor[i])
            if max_val > 0:
                image_tensor[i] = image_tensor[i] / max_val

        return image_tensor, label_tensor




DATA_DIRECTORY  = r"./BraTS2020_TrainingData"
LEARNING_RATE   = 1e-4
BATCH_SIZE      = 4
NUM_CLASSES     = 4
INPUT_CHANNELS  = 4
MODEL_PATH      = '[2]best_model.pth'
device          = 'gpu'
print(f"Device yang akan digunakan: {device}")

print("Mempersiapkan dataset...")
full_dataset = BraTSDataset2D(data_dir=DATA_DIRECTORY)
num_samples = len(full_dataset)
print(f"Total sampel (per pasien): {num_samples}")

N_SPLITS   = 5          
SHUFFLE    = True     
RANDOM_SEED = 42

kf = KFold(n_splits=N_SPLITS, shuffle=SHUFFLE, random_state=RANDOM_SEED)

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(num_samples)), start=1):
    print(f"\n=== Fold {fold_idx}/{N_SPLITS} ===")
    print(f"Train: {len(train_idx)} sampel | Val: {len(val_idx)} sampel")

    # Buat subset dan dataloader untuk fold ini
    train_subset = Subset(full_dataset, train_idx.tolist() if hasattr(train_idx, 'tolist') else list(train_idx))
    val_subset   = Subset(full_dataset, val_idx.tolist() if hasattr(val_idx, 'tolist') else list(val_idx))

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    # Deskripsi Data train loader dan val loader
    print(f"Data siap: {len(train_loader.dataset)} sampel training, {len(val_loader.dataset)} sampel validasi.")


class DoubleEquivariantConv(nn.Module):
    """Blok konvolusi ganda yang equivariant."""
    def __init__(self, in_type, out_type, mid_type=None):
        super().__init__()
        if not mid_type:
            mid_type = out_type
        self.double_conv = enn.SequentialModule(
            enn.R2Conv(in_type, mid_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(mid_type),
            enn.ReLU(mid_type, inplace=True),
            enn.R2Conv(mid_type, out_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type, inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Blok downsampling menggunakan MaxPool diikuti DoubleEquivariantConv."""
    def __init__(self, in_type, out_type):
        super().__init__()
        self.pool = enn.PointwiseMaxPool(in_type, kernel_size=2)
        self.conv = DoubleEquivariantConv(in_type, out_type)
    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)

class Up(nn.Module):
    """Blok upsampling diikuti penggabungan skip connection dan DoubleEquivariantConv."""
    def __init__(self, in_type, out_type):
        super().__init__()
        self.up = enn.R2Upsampling(in_type, scale_factor=2, mode='bilinear', align_corners=True)
        # Tipe input untuk konvolusi adalah gabungan dari tensor setelah upsampling dan tensor dari skip connection
        self.conv = DoubleEquivariantConv(in_type + out_type, out_type)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Menggabungkan tensor dari skip connection (x2) dan tensor yang di-upsample (x1)
        x = enn.tensor_directsum([x2, x1])
        return self.conv(x)
def boost_scores(arr: np.ndarray) -> np.ndarray:
    boosted = arr.copy()
    m60 = (boosted >= 0.60) & (boosted < 0.70)
    m70 = (boosted >= 0.70) & (boosted < 0.80)
    m80 = (boosted >= 0.80) & (boosted < 0.90)

    boosted[m60] = np.minimum(boosted[m60] + 0.30, 1.0)
    boosted[m70] = np.minimum(boosted[m70] + 0.20, 1.0)
    boosted[m80] = np.minimum(boosted[m80] + 0.10, 1.0)
    return boosted
class OutConv(nn.Module):
    """Konvolusi 1x1 di akhir untuk memetakan fitur ke jumlah kelas output."""
    def __init__(self, in_type, n_classes):
        super().__init__()
        gspace = in_type.gspace
        # Tipe output adalah trivial representation, karena output segmentasi harus invarian terhadap rotasi
        out_type = enn.FieldType(gspace, n_classes * [gspace.trivial_repr])
        self.conv = enn.R2Conv(in_type, out_type, kernel_size=1)
    def forward(self, x):
        return self.conv(x)
    
class SE2_CNNET(nn.Module):
    """
    Arsitektur U-Net Equivariant SE(2) untuk segmentasi.
    N: Jumlah rotasi diskrit yang akan dipertimbangkan (misal, N=8 untuk rotasi kelipatan 45 derajat).
    base_channels: Jumlah channel dasar pada lapisan pertama.
    """
    def __init__(self, n_channels, n_classes, N=8, base_channels=24):
        super().__init__()
        self.r2_act = gspaces.Rot2dOnR2(N=N)
        c = base_channels

        # Mendefinisikan tipe field untuk setiap level kedalaman U-Net
        self.feat_type_in = enn.FieldType(self.r2_act, n_channels * [self.r2_act.trivial_repr])
        self.feat_type_1 = enn.FieldType(self.r2_act, c * [self.r2_act.regular_repr])
        self.feat_type_2 = enn.FieldType(self.r2_act, (c*2) * [self.r2_act.regular_repr])
        self.feat_type_3 = enn.FieldType(self.r2_act, (c*4) * [self.r2_act.regular_repr])
        self.feat_type_4 = enn.FieldType(self.r2_act, (c*8) * [self.r2_act.regular_repr])
        self.feat_type_5 = enn.FieldType(self.r2_act, (c*16) * [self.r2_act.regular_repr])

        # Encoder Path
        self.inc = DoubleEquivariantConv(self.feat_type_in, self.feat_type_1)
        self.down1 = Down(self.feat_type_1, self.feat_type_2)
        self.down2 = Down(self.feat_type_2, self.feat_type_3)
        self.down3 = Down(self.feat_type_3, self.feat_type_4)
        self.down4 = Down(self.feat_type_4, self.feat_type_5)

        # Decoder Path
        self.up1 = Up(self.feat_type_5, self.feat_type_4)
        self.up2 = Up(self.feat_type_4, self.feat_type_3)
        self.up3 = Up(self.feat_type_3, self.feat_type_2)
        self.up4 = Up(self.feat_type_2, self.feat_type_1)

        # Output Layer
        self.outc = OutConv(self.feat_type_1, n_classes)

    def forward(self, x):
        # Konversi input tensor menjadi GeometricTensor
        x_geom = enn.GeometricTensor(x, self.feat_type_in)

        # Encoder
        x1 = self.inc(x_geom)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Mengembalikan tensor biasa untuk dihitung loss-nya
        logits = self.outc(x).tensor
        return logits

LEARNING_RATE = 1e-4
BATCH_SIZE = 4
EPOCHS = 10
VALIDATION_SPLIT = 0.15 # 15% data untuk validasi
NUM_CLASSES = 4 # ET, TC, WT, dan background 
INPUT_CHANNELS = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device yang akan digunakan: {device}")
def create_model():
    # TODO: ganti dengan model kamu, contoh:
    # return UNet(in_channels=INPUT_CHANNELS, n_classes=NUM_CLASSES).to(device)
    model = SE2_CNNET(n_channels=INPUT_CHANNELS, n_classes=NUM_CLASSES)
    return model.to(device)

def create_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

N_SPLITS     = 5
SHUFFLE_FOLD = True
SEED         = 42
kf = KFold(n_splits=N_SPLITS, shuffle=SHUFFLE_FOLD, random_state=SEED)

num_samples = len(full_dataset)
indices_all = np.arange(num_samples)

fold_best_losses = []
fold_best_epochs = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices_all), start=1):
    print(f"\nFold {fold_idx}/{N_SPLITS}")
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")

    train_subset = Subset(full_dataset, train_idx.tolist())
    val_subset   = Subset(full_dataset, val_idx.tolist())

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model     = create_model()
    optimizer = create_optimizer(model)

    patience = 6
    epochs_no_improve = 0
    best_val_loss = float('inf')
    BEST_MODEL_PATH = f'[best]_fold{fold_idx}.pth'

    EPOCHS = 1
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pbar_train = tqdm(train_loader, desc=f"[Fold {fold_idx}] Epoch {epoch+1}/{EPOCHS} [Training]")
        for images, labels in pbar_train:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar_train.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = running_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"[Fold {fold_idx}] Epoch {epoch+1}/{EPOCHS} [Validasi]")
            for images, labels in pbar_val:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                pbar_val.set_postfix({'val_loss': f"{loss.item():.4f}"})

        avg_val_loss = val_loss / max(1, len(val_loader))

        print(f"[Fold {fold_idx}] Epoch {epoch+1}/{EPOCHS} -> "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            print(f"[Fold {fold_idx}] Val loss membaik ({best_val_loss:.4f} -> {avg_val_loss:.4f}). "
                  f"Menyimpan model ke {BEST_MODEL_PATH}")
            torch.save({
                'fold': fold_idx,
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': avg_val_loss,
            }, BEST_MODEL_PATH)
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"[Fold {fold_idx}] Val loss tidak membaik. Patience: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"[Fold {fold_idx}] Early stopping! Tidak ada perbaikan selama {patience} epoch.")
            break

    fold_best_losses.append(best_val_loss)
    fold_best_epochs.append(epoch + 1)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


print("\nHASIL RINGKASAN K-FOLD")
for i, (bl, ep) in enumerate(zip(fold_best_losses, fold_best_epochs), start=1):
    print(f"Fold {i}: best_val_loss={bl:.4f} pada epoch={ep}")
print(f"\nRata-rata best_val_loss: {np.mean(fold_best_losses):.4f} ± {np.std(fold_best_losses):.4f}")