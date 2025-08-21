# =========================
# Colab-ready: Stable, staged fine-tuning for ECG single-image classification
# - Phase 1: train head only (HEAD_EPOCHS)
# - Phase 2: unfreeze and fine-tune whole model (TOTAL_EPOCHS-HEAD_EPOCHS)
# - Conservative augmentations (ECG-friendly)
# - Label smoothing + class weights
# - ReduceLROnPlateau scheduler
# - CSV logging, per-epoch checkpoints, best_model saved
# - Plots saved to /content
# =========================

# If first time, ensure drive mounted:
# from google.colab import drive
# drive.mount('/content/drive')

import os, time, random
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- Config ----------------
DATA_DIR = "/content/drive/MyDrive/data"   # <- 改成你的路径
BACKBONE = "resnet18"   # "resnet18" / "resnet50" / "mobilenet_v2"
TOTAL_EPOCHS = 30
HEAD_EPOCHS = 4         # 先训练头部的轮数（一般 2-5）
BATCH_SIZE = 16
IMG_SIZE = 224
LR_HEAD = 1e-3          # LR when training head only
LR_FINETUNE = 1e-4      # LR when fine-tuning whole model
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = "/content/ecg_train_log_staged.csv"
CKPT_DIR = "/content/ecg_checkpoints_staged"
os.makedirs(CKPT_DIR, exist_ok=True)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------- discover classes (ignore hidden) ----------
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
classes = []
for d in sorted(os.listdir(DATA_DIR)):
    if d.startswith('.'): continue
    folder = Path(DATA_DIR)/d
    if folder.is_dir() and any(p.suffix.lower() in IMG_EXTS for p in folder.iterdir()):
        classes.append(d)
print("Classes:", classes)
class_to_idx = {c:i for i,c in enumerate(classes)}

# ---------- collect samples ----------
samples, labels = [], []
for c in classes:
    folder = Path(DATA_DIR)/c
    files = sorted([str(p) for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    print(f"  {c}: {len(files)} images")
    for p in files:
        samples.append(p); labels.append(class_to_idx[c])
print("Total samples:", len(samples))

# ---------- stratified split ----------
train_paths, test_paths, train_labels, test_labels = train_test_split(
    samples, labels, test_size=0.20, stratify=labels, random_state=SEED)
print("Train:", len(train_paths), "Test:", len(test_paths))

# ---------------- Transforms (conservative) ----------------
train_transform = transforms.Compose([
    transforms.Resize((int(IMG_SIZE*1.05), int(IMG_SIZE*1.05))),
    transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(6),                         # small rotation
    transforms.ColorJitter(brightness=0.06, contrast=0.06),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------- Dataset ----------------
class SimpleImageDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx], p  # return path for debugging

train_ds = SimpleImageDataset(train_paths, train_labels, transform=train_transform)
test_ds  = SimpleImageDataset(test_paths, test_labels, transform=test_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# ----------------- Backbone selection -----------------
def make_backbone(name="resnet18", pretrained=True, num_classes=len(classes)):
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

model = make_backbone(BACKBONE, pretrained=True).to(DEVICE)
print("Backbone:", BACKBONE)

# ---------------- Label smoothing loss ----------------
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.05):
        super().__init__()
        self.eps = eps
    def forward(self, logits, target):
        c = logits.size(-1)
        log_preds = torch.log_softmax(logits, dim=-1)
        # Negative log likelihood for true class:
        nll = -log_preds.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth = -log_preds.mean(dim=-1)
        loss = (1 - self.eps) * nll + self.eps * smooth
        return loss.mean()

# ---------------- class weights ----------------
counter = Counter(train_labels)
class_counts = [counter[i] for i in range(len(classes))]
print("Train class counts:", class_counts)
class_weights = torch.tensor([sum(class_counts)/c for c in class_counts], dtype=torch.float).to(DEVICE)

# We'll use label smoothing loss; combine with class weights by scaling per sample:
# Implement wrapper loss that multiplies loss by class weight per sample
smoothing = 0.05
base_loss_fn = LabelSmoothingCrossEntropy(eps=smoothing)

def weighted_smooth_loss(logits, targets):
    # per-sample losses
    c = logits.size(-1)
    log_preds = torch.log_softmax(logits, dim=-1)
    nll = -log_preds.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
    smooth_term = -log_preds.mean(dim=-1)
    per_sample = (1 - smoothing) * nll + smoothing * smooth_term
    weights = class_weights[targets]  # per-sample weight
    return (per_sample * weights).mean()

# ---------------- Phase 1: train head only ----------------
# Freeze backbone params
for name,param in model.named_parameters():
    param.requires_grad = False
# Unfreeze classifier head
if BACKBONE.startswith("resnet"):
    for p in model.fc.parameters(): p.requires_grad = True
elif BACKBONE=="mobilenet_v2":
    for p in model.classifier.parameters(): p.requires_grad = True

optimizer_head = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD, weight_decay=1e-5)
scheduler_head = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_head, factor=0.5, patience=3, verbose=True)

# Logging CSV init
log_cols = ["phase","epoch","train_loss","train_acc","val_loss","val_acc","epoch_time"]
pd.DataFrame(columns=log_cols).to_csv(LOG_PATH, index=False)

best_val = 0.0
best_path = None

def evaluate(model, loader):
    model.eval()
    total = 0; correct = 0; loss_sum = 0.0
    all_preds=[]; all_labels=[]; all_paths=[]
    with torch.no_grad():
        for imgs, labs, paths in loader:
            imgs = imgs.to(DEVICE); labs = labs.to(DEVICE)
            logits = model(imgs)
            loss = weighted_smooth_loss(logits, labs)
            preds = logits.argmax(dim=1)
            total += imgs.size(0)
            correct += (preds==labs).sum().item()
            loss_sum += float(loss) * imgs.size(0)
            all_preds += preds.cpu().tolist()
            all_labels += labs.cpu().tolist()
            all_paths += list(paths)
    return loss_sum/total, correct/total, all_preds, all_labels, all_paths

print("=== Phase 1: training classifier head only for", HEAD_EPOCHS, "epochs ===")
for epoch in range(1, HEAD_EPOCHS+1):
    t0 = time.time()
    model.train()
    running_loss=0.0; running_correct=0; running_total=0
    for imgs, labs, _ in train_loader:
        imgs = imgs.to(DEVICE); labs = labs.to(DEVICE)
        optimizer_head.zero_grad()
        logits = model(imgs)
        loss = weighted_smooth_loss(logits, labs)
        loss.backward(); optimizer_head.step()
        preds = logits.argmax(dim=1)
        running_loss += float(loss) * imgs.size(0)
        running_correct += (preds==labs).sum().item()
        running_total += imgs.size(0)
    train_loss = running_loss / running_total
    train_acc = running_correct / running_total

    val_loss, val_acc, _, _, _ = evaluate(model, test_loader)
    epoch_time = time.time()-t0
    print(f"[Head] Epoch {epoch}/{HEAD_EPOCHS} train_acc={train_acc:.4f} val_acc={val_acc:.4f} train_loss={train_loss:.4f} val_loss={val_loss:.4f} time={int(epoch_time)}s")
    # log
    df = pd.read_csv(LOG_PATH)
    row = {"phase":"head","epoch":epoch,"train_loss":train_loss,"train_acc":train_acc,"val_loss":val_loss,"val_acc":val_acc,"epoch_time":epoch_time}
    df.loc[len(df)] = row
    df.to_csv(LOG_PATH, index=False)
    # checkpoint & scheduler
    ckpt = os.path.join(CKPT_DIR, f"head_epoch_{epoch}.pth")
    torch.save(model.state_dict(), ckpt)
    scheduler_head.step(val_loss)
    if val_acc > best_val:
        best_val = val_acc
        best_path = os.path.join(CKPT_DIR, "best_model.pth")
        torch.save(model.state_dict(), best_path)
        print("  New best saved:", best_path)

# ---------------- Phase 2: unfreeze and fine-tune whole model ----------------
print("=== Phase 2: unfreeze all params and fine-tune for remaining epochs ===")
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=LR_FINETUNE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

remaining = TOTAL_EPOCHS - HEAD_EPOCHS
if remaining <= 0:
    print("TOTAL_EPOCHS <= HEAD_EPOCHS, skipping phase 2.")
else:
    early_stop_patience = 8
    wait = 0
    for e in range(1, remaining+1):
        epoch = HEAD_EPOCHS + e
        t0 = time.time()
        model.train()
        running_loss=0.0; running_correct=0; running_total=0
        for imgs, labs, _ in train_loader:
            imgs = imgs.to(DEVICE); labs = labs.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = weighted_smooth_loss(logits, labs)
            loss.backward(); optimizer.step()
            preds = logits.argmax(dim=1)
            running_loss += float(loss) * imgs.size(0)
            running_correct += (preds==labs).sum().item()
            running_total += imgs.size(0)
        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        val_loss, val_acc, all_preds, all_labels, all_paths = evaluate(model, test_loader)
        epoch_time = time.time()-t0
        print(f"[FineTune] Epoch {epoch}/{TOTAL_EPOCHS} train_acc={train_acc:.4f} val_acc={val_acc:.4f} train_loss={train_loss:.4f} val_loss={val_loss:.4f} time={int(epoch_time)}s")

        # log
        df = pd.read_csv(LOG_PATH)
        row = {"phase":"finetune","epoch":epoch,"train_loss":train_loss,"train_acc":train_acc,"val_loss":val_loss,"val_acc":val_acc,"epoch_time":epoch_time}
        df.loc[len(df)] = row
        df.to_csv(LOG_PATH, index=False)

        # checkpoint and best
        ckpt = os.path.join(CKPT_DIR, f"finetune_epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt)
        if val_acc > best_val:
            best_val = val_acc
            best_path = os.path.join(CKPT_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print("  New best saved:", best_path)
            wait = 0
        else:
            wait += 1
        scheduler.step(val_loss)

        # early stopping
        if wait >= early_stop_patience:
            print("Early stopping triggered in phase 2. wait =", wait)
            break

print("Training finished. Best val:", best_val, "best_path:", best_path)

# ---------------- final evaluation ----------------
model.load_state_dict(torch.load(best_path))
model.eval()
all_preds=[]; all_labels=[]; all_paths=[]
with torch.no_grad():
    for imgs, labs, paths in test_loader:
        imgs = imgs.to(DEVICE)
        labs_cpu = labs.cpu().tolist()
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        all_preds += preds.cpu().tolist()
        all_labels += labs_cpu
        all_paths += list(paths)

print("Classification report:\n", classification_report(all_labels, all_preds, target_names=classes, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(all_labels, all_preds))

# Save final report
final_report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True, zero_division=0)
pd.DataFrame(final_report).to_csv("/content/final_classification_report_staged.csv")
print("Saved logs:", LOG_PATH)
print("Saved final report: /content/final_classification_report_staged.csv")

# ---------------- plot curves ----------------
df = pd.read_csv(LOG_PATH)
# separate phases for plotting
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(df.index, df['train_acc'], label='train_acc')
plt.plot(df.index, df['val_acc'], label='val_acc')
plt.xlabel('epoch-logrow'); plt.ylabel('accuracy'); plt.legend(); plt.grid(True)
plt.subplot(1,2,2)
plt.plot(df.index, df['train_loss'], label='train_loss')
plt.plot(df.index, df['val_loss'], label='val_loss')
plt.xlabel('epoch-logrow'); plt.ylabel('loss'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig('/content/train_val_curves_staged.png')
print("Saved curves: /content/train_val_curves_staged.png")
plt.show()

# ---------------- save misclassified examples (optional) ----------------
misfile = "/content/misclassified_staged.txt"
with open(misfile,"w") as f:
    for p,t,pr in zip(all_paths, all_labels, all_preds):
        if t != pr:
            f.write(f"{p}\ttrue={classes[t]}\tpred={classes[pr]}\n")
print("Saved misclassified list:", misfile)
