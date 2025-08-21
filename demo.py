# =========================
# Colab: Backbone comparison (ResNet18, MobileNetV2, ResNet50)
# - staged training (head -> finetune)
# - consistent transforms/splits
# - per-model logs, checkpoints, final comparison CSV
# - ensemble (avg logits) evaluation
# Copy entire cell into Colab and run.
# =========================

# 如果还没挂载 Drive，请先：
# from google.colab import drive
# drive.mount('/content/drive')

import os, time, random
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# ---------------- Config ----------------
DATA_DIR = "/content/drive/MyDrive/data"   # <-- 改成你的路径
BACKBONES = ["resnet18", "mobilenet_v2", "resnet50"]  # 要比较的模型
IMG_SIZE = 224
BATCH_SIZE = 16
HEAD_EPOCHS = 3    # 阶段1：只训head
FT_EPOCHS = 12     # 阶段2：全网微调
TOTAL_EPOCHS = HEAD_EPOCHS + FT_EPOCHS
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = "/content/compare_checkpoints"
LOG_DIR = "/content/compare_logs"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------- discover classes ----------------
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
classes = []
for d in sorted(os.listdir(DATA_DIR)):
    if d.startswith('.'): continue
    folder = Path(DATA_DIR)/d
    if folder.is_dir() and any(p.suffix.lower() in IMG_EXTS for p in folder.iterdir()):
        classes.append(d)
print("Classes:", classes)
num_classes = len(classes)

# ---------------- collect sample paths ----------------
samples, labels = [], []
class_to_idx = {c:i for i,c in enumerate(classes)}
for c in classes:
    folder = Path(DATA_DIR)/c
    files = sorted([str(p) for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    for p in files:
        samples.append(p); labels.append(class_to_idx[c])
print("Total samples:", len(samples))

# ---------------- stratified split (fixed) ----------------
train_paths, test_paths, train_labels, test_labels = train_test_split(
    samples, labels, test_size=0.20, stratify=labels, random_state=SEED)
print("Train:", len(train_paths), "Test:", len(test_paths))

# ---------------- transforms (conservative) ----------------
train_transform = transforms.Compose([
    transforms.Resize((int(IMG_SIZE*1.05), int(IMG_SIZE*1.05))),
    transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(6),
    transforms.ColorJitter(brightness=0.06, contrast=0.06),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------- dataset class ----------------
class SimpleDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths; self.labels = labels; self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.labels[idx], p

train_ds = SimpleDataset(train_paths, train_labels, transform=train_transform)
test_ds  = SimpleDataset(test_paths, test_labels, transform=test_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# ---------------- helper: make backbone ----------------
def make_backbone(name="resnet18", pretrained=True, num_classes=num_classes):
    if name=="resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feat = model.fc.in_features; model.fc = nn.Linear(in_feat, num_classes)
    elif name=="resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feat = model.fc.in_features; model.fc = nn.Linear(in_feat, num_classes)
    elif name=="mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feat = model.classifier[1].in_features; model.classifier[1] = nn.Linear(in_feat, num_classes)
    else:
        raise ValueError("Unsupported backbone")
    return model

# ---------------- label smoothing + class weights ----------------
class LabelSmoothingCE(nn.Module):
    def __init__(self, eps=0.05):
        super().__init__(); self.eps = eps
    def forward(self, logits, target):
        c = logits.size(-1)
        logp = torch.log_softmax(logits, dim=-1)
        nll = -logp.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth = -logp.mean(dim=-1)
        loss = (1-self.eps) * nll + self.eps * smooth
        return loss.mean()

counter = Counter(train_labels)
class_counts = [counter[i] for i in range(num_classes)]
class_weights = torch.tensor([sum(class_counts)/c for c in class_counts], dtype=torch.float).to(DEVICE)
smoothing = 0.05

def weighted_smooth_loss(logits, targets):
    logp = torch.log_softmax(logits, dim=-1)
    nll = -logp.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
    smooth_term = -logp.mean(dim=-1)
    per = (1 - smoothing) * nll + smoothing * smooth_term
    weights = class_weights[targets]
    return (per * weights).mean()

# ---------------- training & eval functions ----------------
def evaluate_model(model, loader):
    model.eval()
    y_true, y_pred = [], []
    loss_sum = 0.0; total = 0
    with torch.no_grad():
        for imgs, labs, _ in loader:
            imgs = imgs.to(DEVICE); labs = labs.to(DEVICE)
            logits = model(imgs)
            loss = weighted_smooth_loss(logits, labs)
            preds = logits.argmax(dim=1)
            y_true += labs.cpu().tolist()
            y_pred += preds.cpu().tolist()
            loss_sum += float(loss) * imgs.size(0)
            total += imgs.size(0)
    return loss_sum/total, accuracy_score(y_true, y_pred), classification_report(y_true, y_pred, target_names=classes, zero_division=0, output_dict=True), confusion_matrix(y_true, y_pred), y_true, y_pred

def train_backbone(backbone_name):
    print("\n=== Training backbone:", backbone_name, "===")
    model = make_backbone(backbone_name, pretrained=True).to(DEVICE)
    # Phase 1: freeze backbone, train head
    for p in model.parameters(): p.requires_grad = False
    if backbone_name.startswith("resnet"):
        for p in model.fc.parameters(): p.requires_grad = True
    else:
        for p in model.classifier.parameters(): p.requires_grad = True

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3, verbose=False)

    log_rows = []
    best_val = 0.0; best_epoch = -1

    # HEAD_EPOCHS
    for epoch in range(1, HEAD_EPOCHS+1):
        t0 = time.time(); model.train()
        rloss=0.0; rcorrect=0; rtotal=0
        for imgs, labs, _ in train_loader:
            imgs = imgs.to(DEVICE); labs = labs.to(DEVICE)
            opt.zero_grad(); logits = model(imgs)
            loss = weighted_smooth_loss(logits, labs); loss.backward(); opt.step()
            preds = logits.argmax(dim=1)
            rloss += float(loss) * imgs.size(0)
            rcorrect += (preds==labs).sum().item()
            rtotal += imgs.size(0)
        train_loss = rloss/rtotal; train_acc = rcorrect/rtotal
        val_loss, val_acc, _, _, _, _ = evaluate_model(model, test_loader)
        row = {"epoch":epoch, "phase":"head", "train_loss":train_loss, "train_acc":train_acc, "val_loss":val_loss, "val_acc":val_acc}
        log_rows.append(row)
        t = time.time()-t0
        print(f"[{backbone_name}][head] E{epoch}/{HEAD_EPOCHS} train_acc={train_acc:.4f} val_acc={val_acc:.4f} time={int(t)}s")
        sched.step(val_loss)
        # save best
        if val_acc > best_val:
            best_val = val_acc; best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{backbone_name}_best.pth"))

    # Phase 2: unfreeze all
    for p in model.parameters(): p.requires_grad = True
    opt = optim.Adam(model.parameters(), lr=LR_FINETUNE, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3, verbose=False)

    for epoch in range(1, FT_EPOCHS+1):
        epoch_idx = HEAD_EPOCHS + epoch
        t0 = time.time(); model.train()
        rloss=0.0; rcorrect=0; rtotal=0
        for imgs, labs, _ in train_loader:
            imgs = imgs.to(DEVICE); labs = labs.to(DEVICE)
            opt.zero_grad(); logits = model(imgs)
            loss = weighted_smooth_loss(logits, labs); loss.backward(); opt.step()
            preds = logits.argmax(dim=1)
            rloss += float(loss) * imgs.size(0)
            rcorrect += (preds==labs).sum().item()
            rtotal += imgs.size(0)
        train_loss = rloss/rtotal; train_acc = rcorrect/rtotal
        val_loss, val_acc, _, _, _, _ = evaluate_model(model, test_loader)
        row = {"epoch":epoch_idx, "phase":"finetune", "train_loss":train_loss, "train_acc":train_acc, "val_loss":val_loss, "val_acc":val_acc}
        log_rows.append(row)
        t = time.time()-t0
        print(f"[{backbone_name}][finetune] E{epoch_idx}/{TOTAL_EPOCHS} train_acc={train_acc:.4f} val_acc={val_acc:.4f} time={int(t)}s")
        sched.step(val_loss)
        # save best
        if val_acc > best_val:
            best_val = val_acc; best_epoch = epoch_idx
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{backbone_name}_best.pth"))
    # save per-model log
    df = pd.DataFrame(log_rows)
    log_path = os.path.join(LOG_DIR, f"{backbone_name}_log.csv")
    df.to_csv(log_path, index=False)
    # final eval with best weights
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"{backbone_name}_best.pth"), map_location=DEVICE))
    val_loss, val_acc, report, cm, y_true, y_pred = evaluate_model(model, test_loader)
    # compute f1
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    params = sum(p.numel() for p in model.parameters())
    # save final report
    summary = {"backbone":backbone_name, "best_val_acc":val_acc, "macro_f1":macro_f1, "weighted_f1":weighted_f1, "params":params, "best_epoch":best_epoch}
    return summary, report, cm, y_true, y_pred

# ---------------- run comparison ----------------
summaries = []
reports = {}
cms = {}
preds_by_model = {}
trues = None

for bk in BACKBONES:
    s, rep, cm, y_true, y_pred = train_backbone(bk)
    summaries.append(s)
    reports[bk] = rep
    cms[bk] = cm
    preds_by_model[bk] = y_pred
    if trues is None:
        trues = y_true

# save comparison CSV
comp_df = pd.DataFrame(summaries)
comp_df.to_csv("/content/model_comparison.csv", index=False)
print("\nSaved comparison to /content/model_comparison.csv")
print(comp_df)

# ---------------- ensemble (avg logits) ----------------
print("\n=== Ensemble (average logits of saved best models) ===")
# load models, compute logits on test set, average
logits_sum = None
count = 0
test_dataset_paths = [p for p in test_paths]
# build a simple test loader that returns path too (but we'll load images with test_transform)
single_loader = DataLoader(SimpleDataset(test_paths, test_labels, transform=test_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

for bk in BACKBONES:
    model = make_backbone(bk, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"{bk}_best.pth"), map_location=DEVICE))
    model.eval()
    all_logits = []
    with torch.no_grad():
        for imgs, _, _ in single_loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            all_logits.append(out.cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0)  # [N, C]
    if logits_sum is None:
        logits_sum = all_logits
    else:
        logits_sum += all_logits
    count += 1

avg_logits = logits_sum / count
ens_preds = np.argmax(avg_logits, axis=1)
ens_acc = accuracy_score(test_labels, ens_preds)
ens_macro = f1_score(test_labels, ens_preds, average='macro')
ens_weighted = f1_score(test_labels, ens_preds, average='weighted')
ens_report = classification_report(test_labels, ens_preds, target_names=classes, zero_division=0, output_dict=True)
print("Ensemble acc:", ens_acc, "macro_f1:", ens_macro, "weighted_f1:", ens_weighted)
pd.DataFrame(ens_report).to_csv("/content/ensemble_report.csv")
print("Saved ensemble_report.csv and model_comparison.csv")

# ---------------- print summaries ----------------
print("\nPer-model summary:")
print(comp_df)
