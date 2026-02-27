# train_kfold.py
import argparse
import os
import gc
import random

import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset
from torchvision import transforms
from sklearn.metrics import f1_score
from tqdm import tqdm
import timm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_CSV = "crop_summary_with_folds.csv"
MODE = "STAGE1"  # or "STAGE2"
N_SPLITS = 5
MODEL_NAME = "convnext_base.fb_in22k_ft_in1k_384"
IMG_SIZE = 384
EPOCHS = 20
BATCH_SIZE = 24
SAVE_DIR = f"result_weights/patient_kfold/260227_{MODE}_{MODEL_NAME}"

NUM_CLASSES = 3 if MODE == "STAGE1" else 2

# -----------------------------
# Transforms
# -----------------------------
class HistogramFlattening:
    def __call__(self, img):
        img_np = np.array(img.convert("L"))
        img_eq = cv2.equalizeHist(img_np)
        img_eq = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(img_eq)


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        p_left = (max_wh - w) // 2
        p_top = (max_wh - h) // 2
        padding_img = Image.new(image.mode, (max_wh, max_wh), (0, 0, 0))
        padding_img.paste(image, (p_left, p_top))
        return padding_img


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# -----------------------------
# Dataset
# -----------------------------
class FractureDataset(Dataset):
    """
    Expect df columns:
      - Crop_Image_Path
      - Diagnosis
    """
    def __init__(self, df, transform=None, mode="STAGE1"):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.mode = mode
        self.labels = self._build_labels()

    def _build_labels(self):
        labels = []
        for diag in self.df["Diagnosis"].astype(str).str.strip().tolist():
            if self.mode == "STAGE1":
                # Normal(0) vs Fracture(1) vs VP(2)
                if diag == "Normal":
                    labels.append(0)
                elif diag in ["Acute", "Chronic"]:
                    labels.append(1)
                elif diag == "VP":
                    labels.append(2)
                else:
                    labels.append(0)
            else:
                # STAGE2 (default): Acute(0) vs Chronic(1)
                if diag == "Acute":
                    labels.append(0)
                elif diag == "Chronic":
                    labels.append(1)
                else:
                    # should not exist if filtered properly
                    labels.append(0)
        return labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = str(row["Crop_Image_Path"])
        label = int(self.labels[idx])

        try:
            image = Image.open(img_path)
        except Exception:
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
        return image, label


def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        HistogramFlattening(),
        SquarePad(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        HistogramFlattening(),
        SquarePad(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def make_sampler(labels: np.ndarray):
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def train_one_fold(fold, df_all):
    # indices from fold column (patient-level fixed)
    train_idx = df_all.index[df_all["fold"] != fold].to_numpy()
    val_idx = df_all.index[df_all["fold"] == fold].to_numpy()

    train_tf, val_tf = build_transforms(IMG_SIZE)

    train_ds = FractureDataset(df_all, transform=train_tf, mode=MODE)
    val_ds = FractureDataset(df_all, transform=val_tf, mode=MODE)

    train_sub = Subset(train_ds, train_idx)
    val_sub = Subset(val_ds, val_idx)

    # sampler on train labels
    all_labels = np.array(train_ds.labels)
    train_labels_fold = all_labels[train_idx]
    sampler = make_sampler(train_labels_fold)

    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)

    if MODE == "STAGE1":
        criterion = FocalLoss(alpha=5.0, gamma=2.0)
    else:
        # STAGE2 2-class default
        criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

    best_f1 = -1.0
    best_path = os.path.join(SAVE_DIR, f"best_model_fold_{fold}.pth")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in tqdm(train_loader, desc=f"[Fold {fold}] Epoch {epoch+1}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # val
        model.eval()
        vloss = 0.0
        vp, vy = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                vloss += loss.item() * x.size(0)
                vp.extend(out.argmax(1).cpu().numpy().tolist())
                vy.extend(y.cpu().numpy().tolist())

        val_loss = vloss / len(val_loader.dataset)
        val_f1 = f1_score(vy, vp, average="macro")
        scheduler.step()

        print(
            f"[Fold {fold} | Epoch {epoch+1}] "
            f"TrainLoss {train_loss:.4f} Acc {train_acc:.4f} | "
            f"ValLoss {val_loss:.4f} MacroF1 {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)

    # clean up
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return best_f1


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data_csv", type=str, required=True, help="CSV with fold column")
    # parser.add_argument("--mode", type=str, choices=["STAGE1", "STAGE2"], default="STAGE1")
    # parser.add_argument("--n_splits", type=int, default=5)
    # parser.add_argument("--model_name", type=str, default="convnext_base.fb_in22k_ft_in1k_384")
    # parser.add_argument("--img_size", type=int, default=384)
    # parser.add_argument("--epochs", type=int, default=20)
    # parser.add_argument("--batch_size", type=int, default=24)
    # parser.add_argument("--weight_decay", type=float, default=1e-4)
    # parser.add_argument("--label_smoothing", type=float, default=0.0)
    # parser.add_argument("--focal_alpha", type=float, default=0.5)
    # parser.add_argument("--focal_gamma", type=float, default=2.0)
    # parser.add_argument("--num_workers", type=int, default=4)
    # parser.add_argument("--save_dir", type=str, default="result_weights/kfold_patient_level")
    # parser.add_argument("--seed", type=int, default=42)
    # args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(SAVE_DIR, exist_ok=True)

    df_all = pd.read_csv(DATA_CSV)

    # basic cleaning
    df_all["RegID"] = df_all["RegID"].astype(str).str.strip()
    df_all["Diagnosis"] = df_all["Diagnosis"].astype(str).str.strip()
    df_all["Crop_Image_Path"] = df_all["Crop_Image_Path"].astype(str)

    # assert fold exists
    if "fold" not in df_all.columns:
        raise RuntimeError("fold column not found. Run make_folds.py first.")

    # Stage2 default: only Acute/Chronic (2-class)
    if MODE == "STAGE2":
        df_all = df_all[df_all["Diagnosis"].isin(["Acute", "Chronic"])].reset_index(drop=True)

    # num_classes
    # if MODE == "STAGE1":
    #     num_classes = 3
    # else:
    #     num_classes = 2

    print(f"[INFO] Mode={MODE} | Total rows={len(df_all)} | num_classes={NUM_CLASSES}")
    print("[INFO] Fold distribution (rows):")
    print(df_all["fold"].value_counts().sort_index())
    print("[INFO] Label distribution:")
    print(df_all["Diagnosis"].value_counts())

    fold_scores = []
    for fold in range(N_SPLITS):
        print("\n" + "=" * 30)
        print(f"Fold {fold}/{N_SPLITS-1} start")
        print("=" * 30)

        f1 = train_one_fold(
            fold=fold,
            df_all=df_all,
            # mode=args.mode,
            # model_name=args.model_name,
            # num_classes=num_classes,
            # img_size=args.img_size,
            # epochs=args.epochs,
            # batch_size=args.batch_size,
            # weight_decay=args.weight_decay,
            # label_smoothing=args.label_smoothing,
            # focal_alpha=args.focal_alpha,
            # focal_gamma=args.focal_gamma,
            # save_dir=args.save_dir,
            # num_workers=args.num_workers,
        )
        print(f"[DONE] Fold {fold} best MacroF1 = {f1:.4f}")
        fold_scores.append(f1)

    print("\nAll folds done.")
    print(f"MacroF1 mean={float(np.mean(fold_scores)):.4f} std={float(np.std(fold_scores)):.4f}")


if __name__ == "__main__":
    main()