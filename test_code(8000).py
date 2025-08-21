# === Train only on pre-generated ECG images (skip image building) ===
# Uses images under /content/drive/MyDrive/ecg_stt_images/{STChange,Normal}
# Saves: best_model.pth, ecg_train_log_staged.csv, final_classification_report_staged.csv,
#        misclassified_staged.txt, train_val_curves_staged.png, confusion_matrix_staged.png, predictions_staged.csv

import os, re, random, warnings, json
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError

import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score, cohen_kappa_score,
                             matthews_corrcoef, roc_auc_score)

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

warnings.filterwarnings("ignore")

# ----------------- Config -----------------
OUT_IMG_ROOT = "/content/drive/MyDrive/ecg_stt_images"  # 已生成图片根目录（保留你现有路径）
IMG_EXTS     = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

classes      = ["STChange", "Normal"]  # 0: STChange, 1: Normal
class_to_idx = {c:i for i,c in enumerate(classes)}

# 抽样设置：总样本数与 STChange 目标占比（ 0.4 表示 40% STChange, 60% Normal）
MAX_TOTAL_IMAGES = 8000
STCHANGE_TARGET_FRAC = 0.40  # “给 STChange 多抽一点”

# 训练相关
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
IMG_SIZE_TRAIN    = 224
BATCH_SIZE        = 16
NUM_WORKERS       = 2   # 如果 Drive I/O 卡顿可试 0
TOTAL_EPOCHS      = 30
HEAD_EPOCHS       = 4
PHASE1_PATIENCE   = 5
PHASE2_PATIENCE   = 8
LR_HEAD           = 1e-3
LR_FINETUNE       = 1e-4
WEIGHT_DECAY      = 1e-5

# 记录级划分与可选均衡
TEST_SIZE_BY_REC          = 0.20
VAL_SIZE_BY_REC           = 0.10
ENABLE_TRAIN_UNDERSAMPLE  = True
TARGET_POS_NEG_RATIO      = 1.0  # 训练集近似 1:1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Utils: counting & scanning -----------------
def is_img(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTS

def count_images_in_folder(root):
    counts = {}
    for c in classes:
        folder = Path(root)/c
        if not folder.exists():
            counts[c] = 0
            continue
        counts[c] = sum(1 for p in folder.rglob("*") if is_img(p))
    return counts

# 文件名里抽取 record_id，便于记录级拆分
RECNAME_RE = re.compile(r"^(Normal|STChange)_(mitdb|edb)_(.+?)_seg\d+\.png$")
def extract_record_id(path):
    name = Path(path).name
    m = RECNAME_RE.match(name)
    if m:
        return m.group(3)
    base = name.rsplit("_seg", 1)[0]
    return base.split("_")[-1]

# 扫描并按目标比例抽样
def scan_images_with_bias(root, max_total=MAX_TOTAL_IMAGES, st_frac=STCHANGE_TARGET_FRAC, seed=SEED):
    rng = np.random.default_rng(seed)
    all_paths = {c: [] for c in classes}
    all_recs  = {c: [] for c in classes}
    all_labels= {c: [] for c in classes}

    for c in classes:
        folder = Path(root)/c
        if not folder.exists(): continue
        for p in sorted(folder.rglob("*")):
            if is_img(p):
                all_paths[c].append(str(p))
                all_recs[c].append(extract_record_id(str(p)))
                all_labels[c].append(class_to_idx[c])

    n_st_all = len(all_paths["STChange"])
    n_no_all = len(all_paths["Normal"])
    print(f"STChange images: {n_st_all}")
    print(f"Normal images: {n_no_all}")

    if max_total is None:
        # 不抽样：用全部
        paths = np.array(all_paths["STChange"] + all_paths["Normal"])
        labels= np.array(all_labels["STChange"]+ all_labels["Normal"])
        recids= np.array(all_recs["STChange"] + all_recs["Normal"])
        return paths, labels, recids

    # 目标采样量
    tgt_st = int(round(max_total * st_frac))
    tgt_no = max_total - tgt_st

    # 若类别数量不足，则取能取到的最大值
    take_st = min(tgt_st, n_st_all)
    take_no = min(tgt_no, n_no_all)

    # 随机抽样（不放回）
    st_idx = rng.choice(n_st_all, size=take_st, replace=False) if take_st>0 else np.array([], dtype=int)
    no_idx = rng.choice(n_no_all, size=take_no, replace=False) if take_no>0 else np.array([], dtype=int)

    paths = [all_paths["STChange"][i] for i in st_idx] + [all_paths["Normal"][i] for i in no_idx]
    labels= [all_labels["STChange"][i] for i in st_idx] + [all_labels["Normal"][i] for i in no_idx]
    recids= [all_recs["STChange"][i]  for i in st_idx] + [all_recs["Normal"][i]  for i in no_idx]

    # 打乱
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    paths  = np.array(paths)[idx]
    labels = np.array(labels)[idx]
    recids = np.array(recids)[idx]
    return paths, labels, recids

# === 扫描并抽样（1 万张，STChange 比例 40%） ===
all_paths, all_labels, all_recids = scan_images_with_bias(OUT_IMG_ROOT, MAX_TOTAL_IMAGES, STCHANGE_TARGET_FRAC, SEED)

print("Total images:", len(all_paths),
      "| STChange:", int((all_labels==0).sum()),
      "| Normal:",   int((all_labels==1).sum()))

# ----------------- Record-level split (avoid leakage) -----------------
def split_by_record(rec_ids, labels, test_size=TEST_SIZE_BY_REC, val_size=VAL_SIZE_BY_REC, seed=SEED):
    unique_rec, first_idx = np.unique(rec_ids, return_index=True)
    rec_labels = labels[first_idx]
    rec_train, rec_test, y_train, y_test = train_test_split(
        unique_rec, rec_labels, test_size=test_size, stratify=rec_labels, random_state=seed
    )
    rec_tr, rec_val, y_tr, y_val = train_test_split(
        rec_train, y_train, test_size=val_size, stratify=y_train, random_state=seed
    )
    def mask_from_recs(target_recs):
        return np.where(np.isin(rec_ids, target_recs))[0]
    return (mask_from_recs(rec_tr), mask_from_recs(rec_val), mask_from_recs(rec_test),
            rec_tr, rec_val, rec_test)

idx_tr, idx_val, idx_te, rec_tr, rec_val, rec_test = split_by_record(all_recids, all_labels)
train_paths = all_paths[idx_tr].tolist(); train_labels = all_labels[idx_tr].tolist()
val_paths   = all_paths[idx_val].tolist(); val_labels = all_labels[idx_val].tolist()
test_paths  = all_paths[idx_te].tolist(); test_labels= all_labels[idx_te].tolist()

print(f"Records split -> Train:{len(rec_tr)} Val:{len(rec_val)} Test:{len(rec_test)}")
print(f"Images split  -> Train:{len(train_paths)} | Val:{len(val_paths)} | Test:{len(test_paths)}")

# ----------------- Optional: undersample majority on train -----------------
def undersample_majority(paths, labels, target_ratio=1.0, seed=SEED):
    paths = np.array(paths); labels = np.array(labels)
    idx_pos = np.where(labels==0)[0]  # STChange
    idx_neg = np.where(labels==1)[0]  # Normal
    n_pos, n_neg = len(idx_pos), len(idx_neg)
    if n_pos == 0 or n_neg == 0: return paths.tolist(), labels.tolist()
    rng = np.random.default_rng(seed)
    if n_pos > target_ratio * n_neg:
        keep_pos = max(int(target_ratio * n_neg), 1)
        chosen = rng.choice(idx_pos, size=keep_pos, replace=False)
        kept = np.concatenate([chosen, idx_neg])
    elif n_neg > (1.0/target_ratio) * n_pos:
        keep_neg = max(int(n_pos / target_ratio), 1)
        chosen = rng.choice(idx_neg, size=keep_neg, replace=False)
        kept = np.concatenate([idx_pos, chosen])
    else:
        kept = np.arange(len(labels))
    rng.shuffle(kept)
    return paths[kept].tolist(), labels[kept].tolist()

if ENABLE_TRAIN_UNDERSAMPLE:
    train_paths, train_labels = undersample_majority(train_paths, train_labels, TARGET_POS_NEG_RATIO)
    print("[Balance] After undersample -> Train STChange:",
          sum(1 for y in train_labels if y==0),
          "| Normal:", sum(1 for y in train_labels if y==1))

# ----------------- Datasets & Dataloaders -----------------
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

def proxy_temporal_features(pil_img, width=1024):
    g = pil_img.convert("L").resize((width, IMG_SIZE_TRAIN), Image.BILINEAR)
    arr = np.asarray(g).astype(np.float32) / 255.0
    sig = arr.mean(axis=0)
    sig = (sig - sig.mean()) / (sig.std() + 1e-6)
    fft = np.fft.rfft(sig); mag = np.abs(fft)
    freqs = np.fft.rfftfreq(sig.size, d=1.0)
    if len(mag) > 1:
        peak_idx = np.argmax(mag[1:]) + 1
        dom_freq, dom_mag = float(freqs[peak_idx]), float(mag[peak_idx])
    else:
        dom_freq, dom_mag = 0.0, 0.0
    spec_centroid = float((freqs * mag).sum() / (mag.sum() + 1e-6))
    thirds = len(mag)//3
    if thirds >= 1:
        band1, band2, band3 = float(mag[:thirds].mean()), float(mag[thirds:2*thirds].mean()), float(mag[2*thirds:].mean())
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

class ECGImgDataset(Dataset):
    """返回 (img_tensor, proxy_feats, label, path) —— 训练里只用前三个；保存误分类时用 path"""
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        try:
            img = Image.open(p).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError):
            img = Image.new("RGB", (IMG_SIZE_TRAIN, IMG_SIZE_TRAIN), (0,0,0))
        feats = proxy_temporal_features(img)
        x = self.transform(img)
        y = self.labels[idx]
        return x, torch.tensor(feats, dtype=torch.float32), y, p

train_ds = ECGImgDataset(train_paths, train_labels, transform=train_tf)
val_ds   = ECGImgDataset(val_paths,   val_labels,   transform=test_tf)
test_ds  = ECGImgDataset(test_paths,  test_labels,  transform=test_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# 小 I/O 自检
try:
    _b = next(iter(train_loader))
    print("[I/O] First train batch OK. shapes:", _b[0].shape, _b[1].shape)
except Exception as e:
    print("[I/O] First train batch failed:", e)

# ----------------- Model (ResNet18 + SE + temporal MLP) -----------------
class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(c, c//r, bias=False), nn.ReLU(True),
            nn.Linear(c//r, c, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        b,c,_,_ = x.size()
        y = self.avg(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x * y

class ImageBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                  net.layer1, net.layer2, net.layer3, net.layer4)
        self.se = SEBlock(512)
        self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x = self.stem(x)
        x = self.se(x)
        x = self.pool(x).flatten(1)
        return x

class TemporalMLP(nn.Module):
    def __init__(self, in_dim=13, hid=64, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(True), nn.BatchNorm1d(hid),
            nn.Linear(hid, out_dim), nn.ReLU(True), nn.BatchNorm1d(out_dim), nn.Dropout(0.2),
        )
    def forward(self, x): return self.mlp(x)

class TwoStreamECGNet(nn.Module):
    def __init__(self, num_classes=2, freeze_image_backbone=True):
        super().__init__()
        self.image = ImageBackbone()
        self.temporal = TemporalMLP(13, 64, 128)
        self.fuse = nn.Sequential(
            nn.Linear(512+128, 256), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        if freeze_image_backbone:
            for p in self.image.parameters(): p.requires_grad = False
    def unfreeze_image(self):
        for p in self.image.parameters(): p.requires_grad = True
    def forward(self, img, feats):
        f1 = self.image(img)
        f2 = self.temporal(feats)
        return self.fuse(torch.cat([f1, f2], dim=1))

# ----------------- Train utils & bookkeeping -----------------
ctr = Counter(train_labels)
counts = [max(ctr.get(i, 0), 1) for i in range(len(classes))]  # 防零
class_weights = torch.tensor([sum(counts)/c for c in counts], dtype=torch.float32, device=DEVICE)
print("Class weights:", class_weights.cpu().numpy().round(3))

model = TwoStreamECGNet(num_classes=2, freeze_image_backbone=True).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 产物保存路径
model_save_path              = "best_model.pth"
logs_save_path               = "ecg_train_log_staged.csv"
result_save_path             = "final_classification_report_staged.csv"
misclassified_save_path      = "misclassified_staged.txt"
train_val_curves_save_path   = "train_val_curves_staged.png"
confusion_matrix_fig_path    = "confusion_matrix_staged.png"
predictions_csv_path         = "predictions_staged.csv"

train_log = []  # [epoch, train_loss, train_acc, val_loss, val_acc, val_ba]

def run_epoch(model, loader, optimizer, criterion, train=True):
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for batch in loader:
            img, feat, lab = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
            if train: optimizer.zero_grad()
            logits = model(img, feat)
            loss = criterion(logits, lab)
            if train:
                loss.backward(); optimizer.step()
            pred = logits.argmax(1)
            bs = lab.size(0)
            total += bs
            correct += (pred==lab).sum().item()
            loss_sum += float(loss)*bs
    return loss_sum/total, correct/total

@torch.no_grad()
def eval_full(model, loader, criterion, need_paths=False):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_preds, all_labels, all_probs, all_paths = [], [], [], []
    for batch in loader:
        img, feat, lab = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
        logits = model(img, feat)
        loss = criterion(logits, lab)
        prob = F.softmax(logits, dim=1)[:,1]
        pred = logits.argmax(1)
        bs = lab.size(0)
        total += bs
        correct += (pred==lab).sum().item()
        loss_sum += float(loss)*bs
        all_preds += pred.cpu().tolist()
        all_labels += lab.cpu().tolist()
        all_probs  += prob.cpu().tolist()
        if need_paths:
            all_paths += list(batch[3])  # 第四个返回是路径
    acc = correct/total
    ba = balanced_accuracy_score(all_labels, all_preds) if len(set(all_labels))>1 else acc
    kappa = cohen_kappa_score(all_labels, all_preds) if len(set(all_labels))>1 else 0.0
    mcc = matthews_corrcoef(all_labels, all_preds) if len(set(all_labels))>1 else 0.0
    try:
        auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels))>1 else float('nan')
    except Exception:
        auc = float('nan')
    if need_paths:
        return acc, ba, kappa, mcc, auc, all_labels, all_preds, all_probs, all_paths
    return acc, ba, kappa, mcc, auc, all_labels, all_preds

def save_model(model, epoch, model_path=model_save_path):
    torch.save(model.state_dict(), model_path)
    print(f"[Save] Model saved at epoch {epoch} -> {model_path}")

def save_predictions_csv(paths, y_true, y_pred, probs, out_path=predictions_csv_path):
    df = pd.DataFrame({
        "path": paths,
        "y_true": y_true,
        "y_pred": y_pred,
        "prob_STChange": probs
    })
    df.to_csv(out_path, index=False)
    print(f"[Save] Predictions -> {out_path}")

def save_misclassified(paths, y_true, y_pred, out_path=misclassified_save_path):
    bad = [p for p, t, pr in zip(paths, y_true, y_pred) if t != pr]
    with open(out_path, "w") as f:
        for p in bad:
            f.write(p + "\n")
    print(f"[Save] Misclassified list -> {out_path} (total {len(bad)})")

# ----------------- Phase 1: head (freeze image backbone) -----------------
opt1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                  lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
sch1 = optim.lr_scheduler.ReduceLROnPlateau(opt1, mode='min', factor=0.5, patience=2)

print(f"\n=== Phase 1: head+temporal for {HEAD_EPOCHS} epochs ===")
best_ba, best_state, wait = -1, None, 0
for ep in range(1, HEAD_EPOCHS+1):
    tr_loss, tr_acc = run_epoch(model, train_loader, opt1, criterion, train=True)
    val_loss, val_acc = run_epoch(model, val_loader,   opt1, criterion, train=False)
    acc, ba, kappa, mcc, auc, y_true, y_pred = eval_full(model, val_loader, criterion)
    sch1.step(val_loss)
    print(f"[Head] {ep}/{HEAD_EPOCHS} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
          f"val_loss={val_loss:.4f} acc={val_acc:.4f} ba={ba:.4f}")
    train_log.append([ep, tr_loss, tr_acc, val_loss, val_acc, ba])

    if ba > best_ba:
        best_ba, best_state, wait = ba, {k:v.cpu() for k,v in model.state_dict().items()}, 0
    else:
        wait += 1
        if wait >= PHASE1_PATIENCE:
            print(f"Early stop Phase1 at epoch {ep}")
            break

if best_state is not None:
    model.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})
    save_model(model, ep)

# ----------------- Phase 2: fine-tune (unfreeze) -----------------
model.unfreeze_image()
opt2 = optim.Adam(model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
sch2 = optim.lr_scheduler.ReduceLROnPlateau(opt2, mode='min', factor=0.5, patience=3)

print(f"\n=== Phase 2: fine-tune for {TOTAL_EPOCHS-HEAD_EPOCHS} epochs ===")
best_ba, best_state, wait = -1, None, 0
for ep in range(HEAD_EPOCHS+1, TOTAL_EPOCHS+1):
    tr_loss, tr_acc = run_epoch(model, train_loader, opt2, criterion, train=True)
    val_loss, val_acc = run_epoch(model, val_loader,   opt2, criterion, train=False)
    acc, ba, kappa, mcc, auc, y_true, y_pred = eval_full(model, val_loader, criterion)
    sch2.step(val_loss)
    print(f"[FT ] {ep}/{TOTAL_EPOCHS} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
          f"val_loss={val_loss:.4f} acc={val_acc:.4f} ba={ba:.4f}")
    train_log.append([ep, tr_loss, tr_acc, val_loss, val_acc, ba])

    if ba > best_ba:
        best_ba, best_state, wait = ba, {k:v.cpu() for k,v in model.state_dict().items()}, 0
    else:
        wait += 1
        if wait >= PHASE2_PATIENCE:
            print(f"Early stop Phase2 at epoch {ep}")
            break

if best_state is not None:
    model.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})
    save_model(model, ep)

# ----------------- Final Test & Saving Artifacts -----------------
acc, ba, kappa, mcc, auc, y_true, y_pred, y_prob, y_paths = eval_full(model, test_loader, criterion, need_paths=True)

# 1) 分类报告 CSV
report_df = pd.DataFrame(classification_report(y_true, y_pred, target_names=classes, zero_division=0, output_dict=True)).transpose()
report_df.to_csv("final_classification_report_staged.csv")
print("[Save] Classification report -> final_classification_report_staged.csv")

# 2) 混淆矩阵 文本 + 图像
cm = confusion_matrix(y_true, y_pred)
with open("misclassified_staged.txt", "w") as f:  # 放错分清单和矩阵文本
    f.write("Confusion Matrix:\n")
    for row in cm:
        f.write(" ".join(map(str, row)) + "\n")
print(f"[Save] Confusion matrix (txt) -> misclassified_staged.txt")

plt.figure(figsize=(4,4))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig("confusion_matrix_staged.png", bbox_inches='tight')
plt.close()
print("[Save] Confusion matrix (png) -> confusion_matrix_staged.png")

# 3) 误分类清单 & 预测明细 CSV
save_misclassified(y_paths, y_true, y_pred, "misclassified_staged.txt")  # 追加错分列表
save_predictions_csv(y_paths, y_true, y_pred, y_prob, "predictions_staged.csv")

# 4) 训练日志 & 学习曲线图
log_df = pd.DataFrame(train_log, columns=["epoch","train_loss","train_acc","val_loss","val_acc","val_ba"])
log_df.to_csv("ecg_train_log_staged.csv", index=False)
print("[Save] Train log -> ecg_train_log_staged.csv")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(log_df["epoch"], log_df["train_loss"], label="Train Loss")
plt.plot(log_df["epoch"], log_df["val_loss"], label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(log_df["epoch"], log_df["train_acc"], label="Train Acc")
plt.plot(log_df["epoch"], log_df["val_acc"], label="Val Acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Accuracy")
plt.tight_layout()
plt.savefig("train_val_curves_staged.png", bbox_inches='tight')
plt.close()
print("[Save] Curves -> train_val_curves_staged.png")

# 5) 终端汇总
print(f"\n=== TEST ===\nAccuracy: {acc:.4f}")
print(f"Balanced Accuracy: {ba:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"AUC: {auc:.4f}")