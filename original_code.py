# ==========================================
# ECG ST-T Change vs Normal (Strategy B, multi-lead RGB images)
# - Preprocess MIT-BIH (Normal) & European ST-T (STChange)
# - Generate multi-lead images with semantic names
# - Record-level (group) data split to avoid leakage
# - Optional train-set balancing (undersample majority)
# - Two-stream training (Image CNN + Proxy-temporal MLP)
# - Class weights + Early stopping + ReduceLROnPlateau
# ==========================================

import sys, subprocess, os, re, math, random, json, warnings
import numpy as np
from pathlib import Path

# ---- install deps (WFDB, SciPy, Matplotlib) ----
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wfdb", "matplotlib", "scipy"], check=True)

import wfdb
from wfdb.processing import resample_sig
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score, cohen_kappa_score,
                             matthews_corrcoef, roc_auc_score)
from collections import Counter

warnings.filterwarnings("ignore")

# ---------------- Config ----------------
RAW_ROOT   = "/content/drive/MyDrive/dataset1"
STT_DIR    = str(Path(RAW_ROOT) / "ST-T")      # European ST-T (edb)
MITDB_DIR  = str(Path(RAW_ROOT) / "MIT-BIH")   # MIT-BIH Arrhythmia (mitdb)

OUT_IMG_ROOT = "/content/drive/MyDrive/ecg_stt_images"
OUT_NORMAL   = str(Path(OUT_IMG_ROOT) / "Normal")
OUT_STCHANGE = str(Path(OUT_IMG_ROOT) / "STChange")
os.makedirs(OUT_NORMAL, exist_ok=True)
os.makedirs(OUT_STCHANGE, exist_ok=True)

# 采样与分窗
FS_TARGET     = 250
WIN_SEC       = 1.2
HOP_SEC       = 0.6
WIN_SAMPLES   = int(FS_TARGET * WIN_SEC)   # 300
HOP_SAMPLES   = int(FS_TARGET * HOP_SEC)   # 150

# 图像
IMG_SIZE    = 256
LINE_WIDTH  = 1.5

# 每记录采样上限（避免单一记录主导）
MAX_PER_REC_NORMAL   = 1500
MAX_PER_REC_STCHANGE = 600

# 训练集可选均衡（对多数类做随机下采样至 ~ratio）
ENABLE_TRAIN_UNDERSAMPLE = True
TARGET_POS_NEG_RATIO = 1.0  # 近似 1:1

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------- Helper (WFDB) ----------------
def list_records_from_dir(db_dir):
    recs = [hea.stem for hea in Path(db_dir).glob("*.hea")]
    return sorted(recs)

def rdrecord_local(db_dir, rec):
    recpath = str(Path(db_dir) / rec)
    record = wfdb.rdrecord(recpath)
    fs     = record.fs
    p      = record.p_signal.astype(np.float32)
    sig_names = record.sig_name
    return p, fs, sig_names

def rdann_local(db_dir, rec, anntype="atr"):
    annpath = str(Path(db_dir) / rec)
    try:
        ann = wfdb.rdann(annpath, anntype)
        return ann
    except Exception:
        return None

# ---------------- Filtering & Resample ----------------
def bandpass_filter(sig, fs, low=0.5, high=40.0, order=3):
    ny = 0.5 * fs
    b, a = butter(order, [low/ny, high/ny], btype='bandpass')
    out = filtfilt(b, a, sig, axis=0)
    return out

def to_target_fs(sig, fs_src, fs_tar=FS_TARGET):
    fs_src = int(fs_src); fs_tar = int(fs_tar)
    if fs_src == fs_tar:
        return sig.astype(np.float32), fs_src
    if sig.ndim == 1:
        out, _ = resample_sig(sig.astype(np.float64), fs=fs_src, fs_target=fs_tar)
        return out.astype(np.float32), fs_tar
    elif sig.ndim == 2:
        outs = []
        for i in range(sig.shape[1]):
            ch, _ = resample_sig(sig[:, i].astype(np.float64), fs=fs_src, fs_target=fs_tar)
            outs.append(ch)
        out = np.stack(outs, axis=1)
        return out.astype(np.float32), fs_tar
    else:
        T = sig.shape[0]
        sig2 = sig.reshape(T, -1).astype(np.float64)
        cols = []
        for i in range(sig2.shape[1]):
            ch, _ = resample_sig(sig2[:, i], fs=fs_src, fs_target=fs_tar)
            cols.append(ch)
        out = np.stack(cols, axis=1)
        return out.astype(np.float32), fs_tar

# ---------- 注释/episode 采样率对齐 ----------
def scale_ann_for_resampled_signal(ann, fs_src, fs_tar):
    scale = float(fs_tar) / float(fs_src)
    samples_scaled = np.rint(np.asarray(ann.sample, dtype=float) * scale).astype(int)
    symbols = np.asarray(ann.symbol, dtype=object) if hasattr(ann, "symbol") else np.array([], dtype=object)
    aux = list(ann.aux_note) if hasattr(ann, "aux_note") else []
    return samples_scaled, symbols, aux

def scale_episodes_for_resampled_signal(episodes, fs_src, fs_tar):
    scale = float(fs_tar) / float(fs_src)
    out = []
    for s,e,lead,typ in episodes:
        ss = int(round(s * scale))
        ee = int(round(e * scale))
        out.append((ss, ee, lead, typ))
    return out

# --------------- ST-T episodes parsing ---------------
ST_BEGIN_RE = re.compile(r"^\(ST([01])([+-])")
ST_END_RE   = re.compile(r"^ST([01]).*\)$")
T_BEGIN_RE  = re.compile(r"^\(T([01])([+-])")
T_END_RE    = re.compile(r"^T([01]).*\)$")
AXIS_SHIFT_PREFIXES = ("(st", "st)", "(t", "t)")

def parse_stt_episodes(ann):
    episodes = []
    pending = {}
    samples = np.asarray(ann.sample, dtype=int)
    auxs = ann.aux_note
    for samp, aux in zip(samples, auxs):
        if not aux:
            continue
        s = aux.strip()
        if s[:3].lower() in AXIS_SHIFT_PREFIXES:
            continue
        mb = ST_BEGIN_RE.match(s); me = ST_END_RE.match(s)
        if mb:
            lead = int(mb.group(1)); pending[("ST", lead)] = samp
        elif me:
            lead = int(me.group(1)); key = ("ST", lead)
            if key in pending:
                episodes.append((pending[key], samp, lead, "ST"))
                pending.pop(key, None)
        mb_t = T_BEGIN_RE.match(s); me_t = T_END_RE.match(s)
        if mb_t:
            lead = int(mb_t.group(1)); pending[("T", lead)] = samp
        elif me_t:
            lead = int(me_t.group(1)); key = ("T", lead)
            if key in pending:
                episodes.append((pending[key], samp, lead, "T"))
                pending.pop(key, None)
    return episodes

def within_any_episode(center, episodes):
    for s, e, lead, typ in episodes:
        if s <= center <= e:
            return True
    return False

# --------------- Image Rendering (two-lead RGB) ---------------
def render_twolead_rgb(seg, fs, out_png, title=None, lw=LINE_WIDTH):
    N = seg.shape[0]
    t = np.linspace(0, N/fs, N)
    plt.figure(figsize=(2.56, 2.56), dpi=100)
    plt.axis('off')
    for s in (0,1):
        v = seg[:,s]
        m, sd = np.mean(v), np.std(v) + 1e-6
        v = np.clip((v-m)/(5*sd), -1.0, 1.0)
        seg[:,s] = v
    s0 = seg[:,0]; s1 = seg[:,1]
    plt.plot(t, s0, color='red',  linewidth=lw)
    plt.plot(t, s1, color='green',linewidth=lw)
    if title: plt.title(title)
    plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0)
    plt.close()

# --------------- Strategy B Segmentation ---------------
def gen_windows_total(n_samples, win=WIN_SAMPLES, hop=HOP_SAMPLES):
    starts, s = [], 0
    while s + win <= n_samples:
        starts.append(s)
        s += hop
    return starts

# --------------- Build Image Dataset ---------------
def build_images_from_mitdb():
    recs = list_records_from_dir(MITDB_DIR)
    print(f"[MIT-BIH] {len(recs)} records")
    total = 0
    for rec in recs:
        try:
            sig, fs, names = rdrecord_local(MITDB_DIR, rec)
        except Exception as e:
            print(f"  skip {rec}: read error {e}")
            continue
        if sig.ndim == 1:
            print(f"  skip {rec}: only 1 channel")
            continue
        sig = sig[:, :2]
        sig = bandpass_filter(sig, fs)
        sig, fs2 = to_target_fs(sig, fs, FS_TARGET)
        n = sig.shape[0]

        ann = rdann_local(MITDB_DIR, rec, "atr")
        if ann is not None:
            beat_samples, beat_syms, _ = scale_ann_for_resampled_signal(ann, fs_src=fs, fs_tar=FS_TARGET)
        else:
            beat_samples, beat_syms = None, None

        starts = gen_windows_total(n, WIN_SAMPLES, HOP_SAMPLES)
        cnt = 0
        for st in starts:
            ed = st + WIN_SAMPLES
            ok = True
            if ann is not None:
                idx = np.where((beat_samples >= st) & (beat_samples < ed))[0]
                if idx.size == 0:
                    ok = False
                else:
                    if np.any(beat_syms[idx] != 'N'):
                        ok = False
            else:
                ok = False
            if not ok:
                continue
            seg = sig[st:ed, :2].copy()
            fname = f"Normal_mitdb_{rec}_seg{st:07d}.png"
            outp  = str(Path(OUT_NORMAL) / fname)
            try:
                render_twolead_rgb(seg, FS_TARGET, outp, title=None)
                cnt += 1; total += 1
                if cnt >= MAX_PER_REC_NORMAL:
                    break
            except Exception as e:
                print("render err:", rec, e)
        print(f"  {rec}: saved Normal {cnt}")
    print(f"[MIT-BIH] total Normal images: {total}")

def build_images_from_stt():
    recs = list_records_from_dir(STT_DIR)
    print(f"[ST-T] {len(recs)} records")
    total = 0
    for rec in recs:
        try:
            sig, fs, names = rdrecord_local(STT_DIR, rec)
        except Exception as e:
            print(f"  skip {rec}: read error {e}")
            continue
        if sig.ndim == 1:
            print(f"  skip {rec}: only 1 channel")
            continue
        sig = sig[:, :2]
        sig = bandpass_filter(sig, fs)
        sig, fs2 = to_target_fs(sig, fs, FS_TARGET)
        n = sig.shape[0]

        ann = rdann_local(STT_DIR, rec, "atr")
        if ann is None or ann.aux_note is None:
            print(f"  skip {rec}: no aux ST/T annotations")
            continue
        episodes_src = parse_stt_episodes(ann)
        if len(episodes_src) == 0:
            print(f"  skip {rec}: no ST/T episodes parsed")
            continue
        episodes = scale_episodes_for_resampled_signal(episodes_src, fs_src=fs, fs_tar=FS_TARGET)

        starts = gen_windows_total(n, WIN_SAMPLES, HOP_SAMPLES)
        cnt = 0
        for st in starts:
            ed = st + WIN_SAMPLES
            center = st + WIN_SAMPLES//2
            if within_any_episode(center, episodes):
                seg = sig[st:ed, :2].copy()
                fname = f"STChange_edb_{rec}_seg{st:07d}.png"
                outp  = str(Path(OUT_STCHANGE) / fname)
                try:
                    render_twolead_rgb(seg, FS_TARGET, outp, title=None)
                    cnt += 1; total += 1
                    if cnt >= MAX_PER_REC_STCHANGE:
                        break
                except Exception as e:
                    print("render err:", rec, e)
        print(f"  {rec}: saved STChange {cnt}")
    print(f"[ST-T] total STChange images: {total}")

# ---------- Run building (如已生成可注释掉) ----------
print("=== Step 1/2: Build image dataset ===")
build_images_from_mitdb()
build_images_from_stt()
print("Images are under:", OUT_IMG_ROOT)

# ============= Dataset scanning & RECORD-LEVEL SPLIT =============
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
classes = ["STChange", "Normal"]  # idx: 0=STChange, 1=Normal
class_to_idx = {c:i for i,c in enumerate(classes)}

RECNAME_RE = re.compile(r"^(Normal|STChange)_(mitdb|edb)_(.+?)_seg\d+\.png$")

def extract_record_id(path):
    name = Path(path).name
    m = RECNAME_RE.match(name)
    if m:
        return m.group(3)  # record id
    # 兜底：去掉前缀直到最后一个 "_seg"
    base = name.rsplit("_seg", 1)[0]
    return base.split("_")[-1]

def scan_images_with_records(root):
    paths, labels, rec_ids = [], [], []
    for c in classes:
        folder = Path(root)/c
        if not folder.exists():
            continue
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() in IMG_EXTS and p.is_file():
                rid = extract_record_id(str(p))
                paths.append(str(p))
                labels.append(class_to_idx[c])
                rec_ids.append(rid)
    return np.array(paths), np.array(labels), np.array(rec_ids)

all_paths, all_labels, all_recids = scan_images_with_records(OUT_IMG_ROOT)
print("Total images:", len(all_paths), "| STChange:", int((all_labels==0).sum()),
      "| Normal:", int((all_labels==1).sum()))

# ---- record-level split: 保证同一记录的图片只出现在一个子集 ----
def split_by_record(rec_ids, labels, test_size=0.20, val_size=0.10, seed=SEED):
    # 记录 -> 类别（取该记录对应的第一个类别，ST-T 和 MITDB 数据集中不会跨类）
    unique_rec, first_idx = np.unique(rec_ids, return_index=True)
    rec_labels = labels[first_idx]
    # 先按记录层面划 test
    rec_train, rec_test, y_train, y_test = train_test_split(
        unique_rec, rec_labels, test_size=test_size, stratify=rec_labels, random_state=seed
    )
    # 再从 train 里划 val
    rec_tr, rec_val, y_tr, y_val = train_test_split(
        rec_train, y_train, test_size=val_size, stratify=y_train, random_state=seed
    )
    # 根据记录集合过滤出样本
    def mask_from_recs(target_recs):
        m = np.isin(rec_ids, target_recs)
        return np.where(m)[0]

    idx_tr  = mask_from_recs(rec_tr)
    idx_val = mask_from_recs(rec_val)
    idx_te  = mask_from_recs(rec_test)
    return idx_tr, idx_val, idx_te, rec_tr, rec_val, rec_test

idx_tr, idx_val, idx_te, rec_tr, rec_val, rec_test = split_by_record(all_recids, all_labels)
print(f"Records split -> Train:{len(rec_tr)} Val:{len(rec_val)} Test:{len(rec_test)}")

train_paths = all_paths[idx_tr].tolist();   train_labels = all_labels[idx_tr].tolist()
val_paths   = all_paths[idx_val].tolist();  val_labels   = all_labels[idx_val].tolist()
test_paths  = all_paths[idx_te].tolist();   test_labels  = all_labels[idx_te].tolist()

print(f"Images split -> Train:{len(train_paths)} | Val:{len(val_paths)} | Test:{len(test_paths)}")

# ---- 可选：训练集多数类下采样，缓解严重不平衡 ----
def undersample_majority(paths, labels, target_ratio=1.0, seed=SEED):
    paths = np.array(paths); labels = np.array(labels)
    idx_pos = np.where(labels==0)[0]  # STChange
    idx_neg = np.where(labels==1)[0]  # Normal
    n_pos, n_neg = len(idx_pos), len(idx_neg)
    if n_pos == 0 or n_neg == 0:
        return paths.tolist(), labels.tolist()
    # 希望 n_pos / n_neg ~= target_ratio
    if n_pos > target_ratio * n_neg:
        keep_pos = max(int(target_ratio * n_neg), 1)
        rng = np.random.default_rng(seed)
        chosen = rng.choice(idx_pos, size=keep_pos, replace=False)
        kept_idx = np.concatenate([chosen, idx_neg])
    elif n_neg > (1.0/target_ratio) * n_pos:
        keep_neg = max(int(n_pos / target_ratio), 1)
        rng = np.random.default_rng(seed)
        chosen = rng.choice(idx_neg, size=keep_neg, replace=False)
        kept_idx = np.concatenate([idx_pos, chosen])
    else:
        kept_idx = np.arange(len(labels))
    # 打乱
    rng = np.random.default_rng(seed)
    rng.shuffle(kept_idx)
    return paths[kept_idx].tolist(), labels[kept_idx].tolist()

if ENABLE_TRAIN_UNDERSAMPLE:
    train_paths, train_labels = undersample_majority(train_paths, train_labels, TARGET_POS_NEG_RATIO)
    print("[Balance] After undersample -> Train STChange:",
          sum(1 for y in train_labels if y==0), "| Normal:", sum(1 for y in train_labels if y==1))

# ============= PyTorch Dataset & Dataloaders =============
IMG_SIZE_TRAIN = 224
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
    def __init__(self, paths, labels, transform):
        self.paths = paths; self.labels = labels; self.transform=transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        try:
            img = Image.open(p).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError):
            # 遇到坏图时返回一个空样本占位（极少发生）
            img = Image.new("RGB", (IMG_SIZE_TRAIN, IMG_SIZE_TRAIN), (0,0,0))
        feats = proxy_temporal_features(img)
        x = self.transform(img)
        y = self.labels[idx]
        return x, torch.tensor(feats, dtype=torch.float32), y

BATCH_SIZE = 16
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = ECGImgDataset(train_paths, train_labels, transform=train_tf)
val_ds   = ECGImgDataset(val_paths,   val_labels,   transform=test_tf)
test_ds  = ECGImgDataset(test_paths,  test_labels,  transform=test_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# ============= Model: Image backbone + SE + Temporal MLP =============
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
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
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

# ============= Training utils =============
TOTAL_EPOCHS = 30
HEAD_EPOCHS  = 4
PHASE1_PATIENCE = 5
PHASE2_PATIENCE = 8
LR_HEAD      = 1e-3
LR_FINETUNE  = 1e-4
WEIGHT_DECAY = 1e-5

# 类权重按训练子集计算（平衡交叉熵）
ctr = Counter(train_labels)
counts = [ctr.get(i, 1) for i in range(len(classes))]
class_weights = torch.tensor([sum(counts)/c for c in counts], dtype=torch.float32, device=DEVICE)
print("Class weights:", class_weights.cpu().numpy().round(3))

model = TwoStreamECGNet(num_classes=2, freeze_image_backbone=True).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

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
            total += bs
            correct += (pred==lab).sum().item()
            loss_sum += float(loss)*bs
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
        total += bs
        correct += (pred==lab).sum().item()
        loss_sum += float(loss)*bs
        all_preds += pred.cpu().tolist()
        all_labels += lab.cpu().tolist()
        all_probs  += prob.cpu().tolist()
    acc = correct/total
    ba = balanced_accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = float('nan')
    return acc, ba, kappa, mcc, auc, all_labels, all_preds

# ------- Phase 1: 冻结图像分支，仅训 temporal+fuse -------
opt1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                  lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
sch1 = optim.lr_scheduler.ReduceLROnPlateau(opt1, mode='min', factor=0.5, patience=2, verbose=True)

print(f"\n=== Phase 1: head+temporal for {HEAD_EPOCHS} epochs ===")
best_ba, best_state, wait = -1, None, 0
for ep in range(1, HEAD_EPOCHS+1):
    tr_loss, tr_acc = run_epoch(model, train_loader, opt1, criterion, train=True)
    val_loss, val_acc = run_epoch(model, val_loader,   opt1, criterion, train=False)
    acc, ba, kappa, mcc, auc, y_true, y_pred = eval_full(model, val_loader, criterion)
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

# ------- Phase 2: 解冻图像分支，整体微调 -------
model.unfreeze_image()
opt2 = optim.Adam(model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
sch2 = optim.lr_scheduler.ReduceLROnPlateau(opt2, mode='min', factor=0.5, patience=3, verbose=True)

print(f"\n=== Phase 2: fine-tune for {TOTAL_EPOCHS-HEAD_EPOCHS} epochs ===")
best_ba, best_state, wait = -1, None, 0
for ep in range(HEAD_EPOCHS+1, TOTAL_EPOCHS+1):
    tr_loss, tr_acc = run_epoch(model, train_loader, opt2, criterion, train=True)
    val_loss, val_acc = run_epoch(model, val_loader,   opt2, criterion, train=False)
    acc, ba, kappa, mcc, auc, y_true, y_pred = eval_full(model, val_loader, criterion)
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

# ------- Final Test -------
acc, ba, kappa, mcc, auc, y_true, y_pred = eval_full(model, test_loader, criterion)
print(f"\n=== TEST ===\nAccuracy: {acc:.4f}")
print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
print(confusion_matrix(y_true, y_pred))
print(f"Balanced Accuracy: {ba:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"AUC: {auc:.4f}")
