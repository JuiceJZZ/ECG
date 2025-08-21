# ==========================================
# ECG ST-T Change vs Normal (Colab, read images from Google Drive robustly)
# - Uses OUT_IMG_ROOT you provided; DOES NOT change that block.
# - Robust scanner: case-insensitive class folders; fallback by filename prefix anywhere.
# - Two-stream model; class weights; optional undersample.
# - ReduceLROnPlateau without 'verbose'; Drive-friendly dataloader (num_workers=0).
# ==========================================

import os, re, io, random, warnings, time
from pathlib import Path
from collections import Counter
import numpy as np

import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score, cohen_kappa_score,
                             matthews_corrcoef, roc_auc_score)

warnings.filterwarnings("ignore")

# ---------------- DO NOT CHANGE (as requested) ----------------
OUT_IMG_ROOT = "/content/drive/MyDrive/ecg_stt_images"  # <<< only change this path if needed
IMG_EXTS     = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

classes      = ["STChange", "Normal"]   # 0: STChange, 1: Normal
class_to_idx = {c:i for i,c in enumerate(classes)}
# --------------------------------------------------------------

# Mount Drive if needed
if not Path("/content/drive").exists() or not Path(OUT_IMG_ROOT).exists():
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
    except Exception:
        pass

DATA_ROOT = Path(OUT_IMG_ROOT)
assert DATA_ROOT.exists(), f"目录不存在：{DATA_ROOT}（请确认已挂载Drive且路径正确）"

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Drive-friendly loader knobs
IMG_SIZE_TRAIN = 224
BATCH_SIZE   = 16
NUM_WORKERS  = 0        # FUSE/Drive 推荐0，稳定性更好
PIN_MEMORY   = False
PERSISTENT   = False
PREFETCH     = None

TOTAL_EPOCHS = 30
HEAD_EPOCHS  = 4
PHASE1_PATIENCE = 5
PHASE2_PATIENCE = 8
LR_HEAD      = 1e-3
LR_FINETUNE  = 1e-4
WEIGHT_DECAY = 1e-5

ENABLE_TRAIN_UNDERSAMPLE = True
TARGET_POS_NEG_RATIO     = 1.0

# ---------- Robust scanning ----------
# Patterns: Normal_mitdb_100_seg0000123.png / STChange_edb_e0105_seg0012345.png
RECNAME_RE = re.compile(r"^(Normal|STChange)_(mitdb|edb)_(.+?)_seg\d+\.png$", re.IGNORECASE)

def extract_record_id(path: str):
    name = Path(path).name
    m = RECNAME_RE.match(name)
    if m: return m.group(3)
    base = name.rsplit(".", 1)[0]
    if "_seg" in base: base = base.split("_seg")[0]
    parts = base.split("_")
    if len(parts) >= 2: return parts[-1]
    return base

def _is_img(p: Path):
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def find_class_folder(root: Path, class_name: str):
    """
    Case-insensitive search for a folder named like class_name directly under root.
    Returns Path or None.
    """
    cname = class_name.casefold()
    if root.exists():
        for child in root.iterdir():
            if child.is_dir() and child.name.casefold() == cname:
                return child
    return None

def gather_from_folder(folder: Path):
    out = []
    for p in folder.rglob("*"):
        if _is_img(p):
            out.append(str(p))
    return out

def gather_by_filename_prefix(root: Path, prefix: str):
    """
    Fallback: search recursively under root for files whose filename starts with 'prefix' (case-insensitive).
    """
    pref = prefix.casefold()
    out = []
    for p in root.rglob("*"):
        if _is_img(p):
            if p.name.casefold().startswith(pref):
                out.append(str(p))
    return out

def scan_images_robust(root: Path):
    """
    Returns: paths, labels, rec_ids (numpy arrays)
    Tries folder Normal/ STChange/ (case-insensitive). If missing/empty, fallback by filename prefix anywhere.
    """
    all_paths, all_labels, all_recids = [], [], []
    for c in classes:
        found = []
        cls_dir = find_class_folder(root, c)
        if cls_dir is not None:
            found = gather_from_folder(cls_dir)
            if len(found) > 0:
                print(f"[SCAN] Found folder for {c}: {cls_dir} | files={len(found)}")
        if len(found) == 0:
            # fallback by filename prefix
            found = gather_by_filename_prefix(root, c + "_")
            if len(found) > 0:
                print(f"[SCAN] Fallback by filename prefix '{c}_' under {root} | files={len(found)}")

        # Log sample files
        if len(found) > 0:
            print(f"[SCAN] {c} sample:", *[Path(x).name for x in found[:3]], sep="\n  - ")

        # append
        all_paths.extend(found)
        all_labels.extend([class_to_idx[c]] * len(found))
        all_recids.extend([extract_record_id(x) for x in found])

    return np.array(all_paths), np.array(all_labels), np.array(all_recids)

all_paths, all_labels, all_recids = scan_images_robust(DATA_ROOT)
n_stc = int((all_labels==0).sum())
n_nor = int((all_labels==1).sum())
print("Total images:", len(all_paths), "| STChange:", n_stc, "| Normal:", n_nor)

# 明确报错但给排查指引
if n_stc == 0 or n_nor == 0:
    # 打印目录树的一级信息辅助排障
    print("\n[DIR] Listing first-level subfolders and few files:")
    for child in DATA_ROOT.iterdir():
        if child.is_dir():
            cnt = sum(1 for _ in child.rglob("*") if _is_img(_))
            print(f"  - {child.name}/ -> {cnt} images")
        else:
            if _is_img(child):
                print(f"  - (file in root) {child.name}")
    raise AssertionError("某一类样本为 0。已尝试文件名前缀回退仍未找到。"
                         "请检查：1) 是否存在 Normal/ 与 STChange/（大小写无关）；"
                         "2) 图片名是否以 'Normal_'、'STChange_' 开头；"
                         "3) 图片扩展名；4) 是否放错了上层目录。")

# ---------- Split (record-level preferred; fallback stratified) ----------
def split_by_record(rec_ids, labels, test_size=0.20, val_size=0.10, seed=SEED):
    unique_rec, first_idx = np.unique(rec_ids, return_index=True)
    rec_labels = labels[first_idx]
    # check cross-class records or unstable rec-ids
    rid_to_labels = {}
    for rid, lab in zip(rec_ids, labels):
        rid_to_labels.setdefault(rid, set()).add(lab)
    multi_label_records = any(len(v) > 1 for v in rid_to_labels.values())
    too_unique = (len(unique_rec) > 0.9*len(labels))

    if multi_label_records or too_unique:
        print("[Split] 回退到分层划分 (记录ID不可用/不稳定)。")
        idx_all = np.arange(len(labels))
        tr, te, ytr, yte = train_test_split(idx_all, labels, test_size=test_size,
                                            stratify=labels, random_state=seed)
        tr2, va, ytr2, yva = train_test_split(tr, ytr, test_size=val_size,
                                              stratify=ytr, random_state=seed)
        return tr2, va, te, None, None, None

    rec_train, rec_test, y_train, y_test = train_test_split(
        unique_rec, rec_labels, test_size=test_size, stratify=rec_labels, random_state=seed
    )
    rec_tr, rec_val, y_tr, y_val = train_test_split(
        rec_train, y_train, test_size=val_size, stratify=y_train, random_state=seed
    )
    def mask_from_recs(target_recs):
        return np.where(np.isin(rec_ids, target_recs))[0]
    idx_tr  = mask_from_recs(rec_tr)
    idx_val = mask_from_recs(rec_val)
    idx_te  = mask_from_recs(rec_test)
    print(f"Records split -> Train:{len(rec_tr)} Val:{len(rec_val)} Test:{len(rec_test)}")
    return idx_tr, idx_val, idx_te, rec_tr, rec_val, rec_test

idx_tr, idx_val, idx_te, rec_tr, rec_val, rec_test = split_by_record(all_recids, all_labels)

train_paths = all_paths[idx_tr].tolist();   train_labels = all_labels[idx_tr].tolist()
val_paths   = all_paths[idx_val].tolist();  val_labels   = all_labels[idx_val].tolist()
test_paths  = all_paths[idx_te].tolist();   test_labels  = all_labels[idx_te].tolist()

print(f"Images split  -> Train:{len(train_paths)} | Val:{len(val_paths)} | Test:{len(test_paths)}")

# ---------- Optional undersample ----------
def undersample_majority(paths, labels, target_ratio=1.0, seed=SEED):
    paths = np.array(paths); labels = np.array(labels)
    idx_pos = np.where(labels==0)[0]; idx_neg = np.where(labels==1)[0]
    n_pos, n_neg = len(idx_pos), len(idx_neg)
    if n_pos == 0 or n_neg == 0:
        return paths.tolist(), labels.tolist()
    rng = np.random.default_rng(seed)
    if n_pos > target_ratio * n_neg:
        keep_pos = max(int(target_ratio * n_neg), 1)
        chosen = rng.choice(idx_pos, size=keep_pos, replace=False)
        kept_idx = np.concatenate([chosen, idx_neg])
    elif n_neg > (1.0/target_ratio) * n_pos:
        keep_neg = max(int(n_pos / target_ratio), 1)
        chosen = rng.choice(idx_neg, size=keep_neg, replace=False)
        kept_idx = np.concatenate([idx_pos, chosen])
    else:
        kept_idx = np.arange(len(labels))
    rng.shuffle(kept_idx)
    return paths[kept_idx].tolist(), labels[kept_idx].tolist()

if ENABLE_TRAIN_UNDERSAMPLE:
    train_paths, train_labels = undersample_majority(train_paths, train_labels, TARGET_POS_NEG_RATIO)
    print("[Balance] After undersample -> Train STChange:",
          sum(1 for y in train_labels if y==0), "| Normal:", sum(1 for y in train_labels if y==1))

# ---------- Proxy temporal features ----------
def proxy_temporal_features(pil_img, width=1024):
    g = pil_img.convert("L").resize((width, IMG_SIZE_TRAIN), Image.BILINEAR)
    arr = np.asarray(g).astype(np.float32) / 255.0
    sig = arr.mean(axis=0)
    sig = (sig - sig.mean()) / (sig.std() + 1e-6)

    fft = np.fft.rfft(sig); mag = np.abs(fft)
    freqs = np.fft.rfftfreq(sig.size, d=1.0)
    if len(mag) > 1:
        peak_idx = int(np.argmax(mag[1:]) + 1)
        dom_freq, dom_mag = float(freqs[peak_idx]), float(mag[peak_idx])
    else:
        dom_freq, dom_mag = 0.0, 0.0
    spec_centroid = float((freqs * mag).sum() / (mag.sum() + 1e-6))
    thirds = len(mag)//3
    if thirds >= 1:
        band1 = float(mag[:thirds].mean()); band2 = float(mag[thirds:2*thirds].mean()); band3 = float(mag[2*thirds:].mean())
    else:
        band1 = band2 = band3 = float(mag.mean())
    zcr = float(((sig[:-1]*sig[1:]) < 0).mean())
    ac = np.correlate(sig, sig, mode='full')[sig.size-1:]; ac[0]=0
    ac_peak_lag = int(np.argmax(ac[:len(ac)//2])) if len(ac)>1 else 0
    ac_peak_val = float(ac[ac_peak_lag]) if len(ac)>0 else 0.0
    var, skw = float(np.var(sig)), float(((sig-sig.mean())**3).mean() / (np.std(sig)+1e-6)**3)
    kur = float(((sig-sig.mean())**4).mean() / (np.var(sig)+1e-6)**2)
    hist,_ = np.histogram(sig, bins=32, range=(-3,3), density=True); hist = hist + 1e-12; hist = hist/hist.sum()
    ent = float(-(hist*np.log(hist)).sum())
    return np.array([dom_freq, dom_mag, spec_centroid,
                     band1, band2, band3, zcr,
                     ac_peak_lag, ac_peak_val, var, skw, kur, ent], dtype=np.float32)

# ---------- Dataset & DataLoaders ----------
class ECGImgDataset(Dataset):
    def __init__(self, paths, labels, transform, retry=2, sleep=0.3):
        self.paths = paths; self.labels = labels; self.transform=transform
        self.retry = retry; self.sleep = sleep
    def __len__(self): return len(self.paths)
    def _open_with_retry(self, p):
        last_err = None
        for _ in range(self.retry):
            try:
                with open(p, "rb") as f:
                    img = Image.open(io.BytesIO(f.read())).convert("RGB")
                return img
            except Exception as e:
                last_err = e
                time.sleep(self.sleep)
        print(f"[WARN] 读取失败，使用占位图: {p} | err={last_err}")
        return Image.new("RGB", (IMG_SIZE_TRAIN, IMG_SIZE_TRAIN), (0,0,0))
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = self._open_with_retry(p)
        feats = proxy_temporal_features(img)
        x = self.transform(img)
        y = self.labels[idx]
        return x, torch.tensor(feats, dtype=torch.float32), y

train_tf = transforms.Compose([
    transforms.Resize((int(IMG_SIZE_TRAIN*1.05), int(IMG_SIZE_TRAIN*1.05))),
    transforms.RandomCrop((IMG_SIZE_TRAIN, IMG_SIZE_TRAIN)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(4),
    transforms.ColorJitter(brightness=0.05, contrast=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE_TRAIN, IMG_SIZE_TRAIN)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

train_ds = ECGImgDataset([*train_paths], [*train_labels], transform=train_tf)
val_ds   = ECGImgDataset([*val_paths],   [*val_labels],   transform=test_tf)
test_ds  = ECGImgDataset([*test_paths],  [*test_labels],  transform=test_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

# quick I/O check
try:
    it = iter(train_loader)
    for _ in range(2):
        _ = next(it)
    print("[I/O] First batches OK.")
except Exception as e:
    raise RuntimeError(f"[I/O] 读取训练批次失败: {e}")

# ---------- Two-stream model ----------
class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(c, c//r, bias=False), nn.ReLU(inplace=True),
            nn.Linear(c//r, c, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        b,c,_,_ = x.size()
        y = self.avg(x).view(b,c); y = self.fc(y).view(b,c,1,1)
        return x * y

class ImageBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            net = models.resnet18(pretrained=True)
        self.stem = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4
        )
        self.se = SEBlock(512)
        self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x = self.stem(x); x = self.se(x); x = self.pool(x).flatten(1)
        return x  # [B,512]

class TemporalMLP(nn.Module):
    def __init__(self, in_dim=13, hid=64, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(inplace=True),
            nn.BatchNorm1d(hid),
            nn.Linear(hid, out_dim), nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_dim), nn.Dropout(0.2),
        )
    def forward(self, x): return self.mlp(x)

class TwoStreamECGNet(nn.Module):
    def __init__(self, num_classes=2, freeze_image_backbone=True):
        super().__init__()
        self.image = ImageBackbone()
        self.temporal = TemporalMLP(13, 64, 128)
        self.fuse = nn.Sequential(
            nn.Linear(512+128, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        if freeze_image_backbone:
            for p in self.image.parameters(): p.requires_grad = False
    def unfreeze_image(self):
        for p in self.image.parameters(): p.requires_grad = True
    def forward(self, img, feats):
        f1 = self.image(img); f2 = self.temporal(feats)
        return self.fuse(torch.cat([f1,f2], dim=1))

def run_epoch(model, loader, optimizer, criterion, train=True):
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for img, feat, lab in loader:
            img, feat, lab = img.to(DEVICE), feat.to(DEVICE), lab.to(DEVICE)
            if train: optimizer.zero_grad()
            logits = model(img, feat)
            loss = criterion(logits, lab)
            if train:
                loss.backward(); optimizer.step()
            pred = logits.argmax(1)
            bs = lab.size(0)
            total += bs; correct += (pred==lab).sum().item(); loss_sum += float(loss)*bs
    return loss_sum/total, correct/total

@torch.no_grad()
def eval_full(model, loader, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_preds, all_labels, all_probs = [], [], []
    for img, feat, lab in loader:
        img, feat, lab = img.to(DEVICE), feat.to(DEVICE), lab.to(DEVICE)
        logits = model(img, feat)
        loss = criterion(logits, lab)
        prob = F.softmax(logits, dim=1)[:,1]
        pred = logits.argmax(1)
        bs = lab.size(0)
        total += bs; correct += (pred==lab).sum().item(); loss_sum += float(loss)*bs
        all_preds += pred.cpu().tolist(); all_labels += lab.cpu().tolist(); all_probs += prob.cpu().tolist()
    acc = correct/total
    ba = balanced_accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = float('nan')
    return acc, ba, kappa, mcc, auc, all_labels, all_preds

# class weights
ctr = Counter(train_labels)
counts = [max(ctr.get(i, 0), 1) for i in range(len(classes))]
class_weights = torch.tensor([sum(counts)/c for c in counts], dtype=torch.float32, device=DEVICE)
print("Class weights:", class_weights.cpu().numpy().round(3))

model = TwoStreamECGNet(num_classes=2, freeze_image_backbone=True).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Phase 1
opt1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                  lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
sch1 = optim.lr_scheduler.ReduceLROnPlateau(opt1, mode='min', factor=0.5, patience=2)

print(f"\n=== Phase 1: head+temporal for {HEAD_EPOCHS} epochs ===")
best_ba, best_state, wait = -1, None, 0
for ep in range(1, HEAD_EPOCHS+1):
    tr_loss, tr_acc = run_epoch(model, train_loader, opt1, criterion, train=True)
    val_loss, val_acc = run_epoch(model, val_loader,   opt1, criterion, train=False)
    acc, ba, kappa, mcc, auc, _, _ = eval_full(model, val_loader, criterion)
    sch1.step(val_loss)
    print(f"[Head] {ep}/{HEAD_EPOCHS} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
          f"val_loss={val_loss:.4f} acc={val_acc:.4f} ba={ba:.4f}")
    if ba > best_ba:
        best_ba, best_state, wait = ba, {k:v.cpu() for k,v in model.state_dict().items()}, 0
    else:
        wait += 1
        if wait >= PHASE1_PATIENCE:
            print(f"Early stop Phase1 at epoch {ep}")
            break
if best_state is not None:
    model.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})

# Phase 2
model.unfreeze_image()
opt2 = optim.Adam(model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
sch2 = optim.lr_scheduler.ReduceLROnPlateau(opt2, mode='min', factor=0.5, patience=3)

print(f"\n=== Phase 2: fine-tune for {TOTAL_EPOCHS-HEAD_EPOCHS} epochs ===")
best_ba, best_state, wait = -1, None, 0
for ep in range(HEAD_EPOCHS+1, TOTAL_EPOCHS+1):
    tr_loss, tr_acc = run_epoch(model, train_loader, opt2, criterion, train=True)
    val_loss, val_acc = run_epoch(model, val_loader,   opt2, criterion, train=False)
    acc, ba, kappa, mcc, auc, _, _ = eval_full(model, val_loader, criterion)
    sch2.step(val_loss)
    print(f"[FT ] {ep}/{TOTAL_EPOCHS} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
          f"val_loss={val_loss:.4f} acc={val_acc:.4f} ba={ba:.4f}")
    if ba > best_ba:
        best_ba, best_state, wait = ba, {k:v.cpu() for k,v in model.state_dict().items()}, 0
    else:
        wait += 1
        if wait >= PHASE2_PATIENCE:
            print(f"Early stop Phase2 at epoch {ep}")
            break
if best_state is not None:
    model.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})

# Final Test
acc, ba, kappa, mcc, auc, y_true, y_pred = eval_full(model, test_loader, criterion)
print(f"\n=== TEST ===\nAccuracy: {acc:.4f}")
print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
print(confusion_matrix(y_true, y_pred))
print(f"Balanced Accuracy: {ba:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"AUC: {auc:.4f}")

save_path = "/content/ecg_twostream_on_drive.pth"
torch.save(model.state_dict(), save_path)
print("Model saved to:", save_path)