### only run this if everything (data, hardware, etc) are ready
from checking import run_test
from dataset import BraTSDataset2D
from modelling import SE2_CNNET
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm 

import numpy as np # Diperlukan untuk float('inf')

run_test()

EPOCHS = 10

DATA_DIRECTORY = r"./DataBrainSeg/BraTS2020_TrainingData"
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
VALIDATION_SPLIT = 0.15 # 15% data untuk validasi
NUM_CLASSES = 4 # ET, TC, WT, dan background 
INPUT_CHANNELS = 4

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = 'cpu'
print(f"Device yang akan digunakan: {device}")


print("Mempersiapkan dataset...")
full_dataset = BraTSDataset2D(data_dir=DATA_DIRECTORY)

n_val = int(len(full_dataset) * VALIDATION_SPLIT)
n_train = len(full_dataset) - n_val
train_set, val_set = random_split(full_dataset, [n_train, n_val])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
print(f"Data siap: {len(train_set)} sampel training, {len(val_set)} sampel validasi.")

model = SE2_CNNET(n_channels=INPUT_CHANNELS, n_classes=NUM_CLASSES)

if torch.cuda.device_count() > 1:
        print(f"Menggunakan {torch.cuda.device_count()} GPU dengan DataParallel.")
        model = nn.DataParallel(model)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# if torch.cuda.device_count() > 1:
#     print(f"Menggunakan {torch.cuda.device_count()} GPU dengan DataParallel.")
#     model = nn.DataParallel(model)

model.to(device)


### training main code
# for epoch in range(EPOCHS):
#     model.train()
#     running_loss = 0.0

#     pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
#     for images, labels in pbar_train:
#         images = images.to(device)
#         labels = labels.to(device)

#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         pbar_train.set_postfix({'loss': loss.item()})

#     avg_train_loss = running_loss / len(train_loader)

#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validasi]")
#         for images, labels in pbar_val:
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()
#             pbar_val.set_postfix({'val_loss': loss.item()})

#     avg_val_loss = val_loss / len(val_loader)

#     print(f"Epoch {epoch+1}/{EPOCHS} -> Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

#     torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')


#### training with cutoff

patience = 6 
epochs_no_improve = 0
best_val_loss = float('inf') 
# Path untuk menyimpan model terbaik.
BEST_MODEL_PATH = '[1]best_model.pth'
EPOCHS = 30
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
    for images, labels in pbar_train:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar_train.set_postfix({'loss': loss.item()})

    avg_train_loss = running_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validasi]")
        for images, labels in pbar_val:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            pbar_val.set_postfix({'val_loss': loss.item()})

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} -> Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    
    if avg_val_loss < best_val_loss:
        print(f"Validation loss membaik ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Menyimpan model ke {BEST_MODEL_PATH}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, BEST_MODEL_PATH)
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"Validation loss tidak membaik. Counter kesabaran: {epochs_no_improve}/{patience}")

    if epochs_no_improve >= patience:
        print(f"Early stopping! Tidak ada perbaikan selama {patience} epoch.")
        break 