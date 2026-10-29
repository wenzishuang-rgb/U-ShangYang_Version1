#!/usr/bin/env python3
"""Train the U-ShangYang masked-reconstruction convolutional autoencoder.

Edit IVT_DIR and OUTPUT below before running. The script reads precomputed
monthly IVT, uIVT and vIVT fields and saves model weights and training states.

For questions, feedback or collaboration, please contact Zishuang Wen at
wenzishuang@stu.ouc.edu.cn. See README.md for affiliation and usage details.
"""

import os
import json
import warnings

warnings.filterwarnings("ignore")
import random
from collections import OrderedDict

import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Seed
SEED = 42


def seed_everything(seed):
    """Seed Python, NumPy and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Input and output paths (edit for your environment)
IVT_DIR = "/Global/Wen_AR_0.25degree/IVT"
OUTPUT = "/Global/Wen_AR_0.25degree/AR_Model/Model_Right"
os.makedirs(OUTPUT + "/checkpoints", exist_ok=True)
os.makedirs(OUTPUT + "/metadata", exist_ok=True)

# Training and validation periods
TRAIN_START = 1980
TRAIN_END = 2010
VAL_START = 2011
VAL_END = 2015
# These test years are recorded as metadata; no test loop is run here.
TEST_START = 2016
TEST_END = 2020
VALIDATION_TIME_SEED = SEED + 1


# Dataset
def collect_monthly_files(start_year, end_year):
    files = []
    for year in range(start_year, end_year + 1):
        year_path = os.path.join(IVT_DIR, str(year))
        if not os.path.exists(year_path):
            continue
        for f in sorted(os.listdir(year_path)):
            if f.endswith(".nc") and "IVT" in f:
                files.append(os.path.join(year_path, f))
    return files


class MonthlyRandomIVTDataset(Dataset):
    """Sample one analysis per monthly file, with fixed validation samples."""

    def __init__(self, start_year, end_year, mode="train", max_cache=2):
        self.mode = mode
        self.files = collect_monthly_files(start_year, end_year)
        self.validation_time_indices = {}
        self.cache = OrderedDict()
        self.max_cache = max_cache
        print("Loading", mode, "dataset...")
        if self.mode == "val":
            rng = np.random.default_rng(VALIDATION_TIME_SEED)
            validation_files = []
            for file in self.files:
                with xr.open_dataset(file) as ds:
                    nt = int(ds.IVT.shape[0])
                if nt < 1:
                    continue
                self.validation_time_indices[file] = int(rng.integers(0, nt))
                validation_files.append(file)
            self.files = validation_files
            print("Validation random time samples:", len(self.validation_time_indices))
        print(mode, "monthly samples:", len(self.files))

    def load_month(self, file):
        if file in self.cache:
            data = self.cache.pop(file)
            self.cache[file] = data
            return data
        ds = xr.open_dataset(file)
        ivt = ds.IVT.values.astype(np.float32)
        uivt = ds.uIVT.values.astype(np.float32)
        vivt = ds.vIVT.values.astype(np.float32)
        ds.close()
        data = (ivt, uivt, vivt)
        self.cache[file] = data
        if len(self.cache) > self.max_cache:
            self.cache.popitem(last=False)
        return data

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        ivt, uivt, vivt = self.load_month(file)
        nt = ivt.shape[0]
        if self.mode == "train":
            t = np.random.randint(0, nt)
        else:
            t = self.validation_time_indices[file]
        x = np.stack([ivt[t], uivt[t], vivt[t]], axis=0)
        x = np.nan_to_num(x)
        return torch.from_numpy(x)


# Model
class AutoEncoder(nn.Module):
    """Reconstruct three input channels through a 64-channel latent field."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        z = self.encoder(x)
        rec = self.decoder(z)
        return rec, z


# Independent random masking of input elements
def mask_input(x, ratio=0.3):
    mask = torch.rand_like(x) > ratio
    return x * mask


# Training, validation and checkpointing
def main():
    train_set = MonthlyRandomIVTDataset(
        TRAIN_START, TRAIN_END, mode="train", max_cache=2
    )
    val_set = MonthlyRandomIVTDataset(VAL_START, VAL_END, mode="val", max_cache=2)
    train_files = set(train_set.files)
    val_files = set(val_set.files)
    overlapping_files = train_files.intersection(val_files)
    print("Training/validation overlapping monthly files:", len(overlapping_files))
    if len(overlapping_files) > 0:
        raise RuntimeError("Training and validation datasets overlap.")
    validation_time_file = os.path.join(
        OUTPUT, "metadata", "validation_random_time_indices.json"
    )
    validation_time_metadata = []
    for file, time_index in val_set.validation_time_indices.items():
        validation_time_metadata.append(
            {
                "file": os.path.relpath(file, IVT_DIR).replace(os.sep, "/"),
                "time_index": time_index,
            }
        )
    with open(validation_time_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "validation_time_seed": VALIDATION_TIME_SEED,
                "training_years": [TRAIN_START, TRAIN_END],
                "validation_years": [VAL_START, VAL_END],
                "test_years": [TEST_START, TEST_END],
                "validation_time_indices": validation_time_metadata,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    # Load data in the main process to avoid duplicating monthly caches.
    train_loader = DataLoader(
        train_set, batch_size=4, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_set, batch_size=2, shuffle=False, num_workers=0, pin_memory=False
    )
    model = AutoEncoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()
    epochs = 120
    best_loss = np.inf
    start_epoch = 0
    # Resume
    resume_file = OUTPUT + "/checkpoints/last_training_state.pth"
    if os.path.exists(resume_file):
        checkpoint = torch.load(resume_file, map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["val_loss"]
        print("Resume from epoch", start_epoch)
    # Epoch loop
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        for x in train_loader:
            x = x.to(DEVICE)
            xm = mask_input(x)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                rec, z = model(xm)
                loss = criterion(rec, x)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        # Evaluate reconstruction on unmasked validation inputs.
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(DEVICE)
                rec, z = model(x)
                val_loss += criterion(rec, x).item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1:03d}", f"Train={train_loss:.6f}", f"Val={val_loss:.6f}")
        # Save every epoch
        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(state, OUTPUT + "/checkpoints/last_training_state.pth")
        torch.save(state, OUTPUT + f"/checkpoints/epoch_{epoch+1:03d}.pth")
        print("Epoch checkpoint saved")
        # Save weights when the validation loss improves.
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), OUTPUT + "/checkpoints/best_model.pth")
            print("Best model updated")
    print("====================")
    print("Training finished")
    print("Best validation loss:", best_loss)


if __name__ == "__main__":
    main()
