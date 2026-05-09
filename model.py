"""
🧠 ResNet3D-18 (Inflate-1 Strategy) - VERSION MASTER PFA
Inclus : Tight Cropping, Focal Loss, Data Augmentation Avancée et TTA.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import pandas as pd
from scipy.ndimage import zoom
import scipy.ndimage as ndimage
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, 
    ConfusionMatrixDisplay, classification_report, roc_auc_score
)
from torchvision.models import resnet18, ResNet18_Weights
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
SKULL_STRIPPED_DIR = r'C:\Users\youne\Desktop\ADNI_SKULL_STRIPPED'
CSV_PATH           = r'dataset_preprocessed.csv'

TARGET_SHAPE = (112, 112, 112) 
BATCH_SIZE   = 8
EPOCHS       = 30
LR           = 1e-4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# ==========================================
# 🛠️ TECHNIQUE AVANCÉE DE PRÉTRAITEMENT
# ==========================================
def crop_to_brain(data, threshold=1e-5):
    """
    Supprime le padding noir (le vide) autour du cerveau.
    """
    mask = data > threshold
    if not np.any(mask):
        return data
        
    coords = np.array(np.where(mask))
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    z_min, z_max = coords[2].min(), coords[2].max()
    
    return data[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]

# ==========================================
# 🎯 FOCAL LOSS (Focus sur les cas difficiles)
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        import torch.nn.functional as F
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        else:
            return F_loss

# ==========================================
# 🧬 ARCHITECTURE RESNET-3D INFLATED
# ==========================================
def inflate_conv2d_to_3d(conv2d):
    k, s, p = conv2d.kernel_size, conv2d.stride, conv2d.padding
    k = (k[0], k[0], k[1]) if isinstance(k, tuple) else (k, k, k)
    s = (s[0], s[0], s[1]) if isinstance(s, tuple) else (s, s, s)
    p = (p[0], p[0], p[1]) if isinstance(p, tuple) else (p, p, p)
    
    conv3d = nn.Conv3d(conv2d.in_channels, conv2d.out_channels, k, s, p, bias=(conv2d.bias is not None))
    with torch.no_grad():
        mid = k[0] // 2
        conv3d.weight.zero_()
        conv3d.weight[:, :, mid, :, :] = conv2d.weight
        if conv2d.bias is not None: conv3d.bias.copy_(conv2d.bias)
    return conv3d

def convert_resnet_to_3d(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, inflate_conv2d_to_3d(child))
        elif isinstance(child, nn.BatchNorm2d):
            bn = child
            setattr(module, name, nn.BatchNorm3d(bn.num_features, bn.eps, bn.momentum, bn.affine))
        elif isinstance(child, nn.MaxPool2d):
            k, s, p = child.kernel_size, child.stride, child.padding
            setattr(module, name, nn.MaxPool3d((k,k,k), (s,s,s), (p,p,p)))
        elif isinstance(child, nn.AdaptiveAvgPool2d):
            setattr(module, name, nn.AdaptiveAvgPool3d((1, 1, 1)))
        else:
            convert_resnet_to_3d(child)

class InflatedResNet3D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        convert_resnet_to_3d(self.model)
        
        old_conv1 = self.model.conv1
        self.model.conv1 = nn.Conv3d(1, 64, kernel_size=old_conv1.kernel_size, stride=old_conv1.stride, padding=old_conv1.padding, bias=False)
        with torch.no_grad():
            self.model.conv1.weight = nn.Parameter(old_conv1.weight.mean(dim=1, keepdim=True))
        
        self.model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.model.fc.in_features, num_classes))

    def forward(self, x):
        return self.model(x)

# ==========================================
# 📂 DATASET ET AUGMENTATION AVANCÉE
# ==========================================
class ADNI3DDataset(Dataset):
    def __init__(self, paths, labels, augment=False):
        self.paths = paths
        self.labels = labels
        self.augment = augment

    def __len__(self): 
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = nib.load(self.paths[idx])
            img = nib.as_closest_canonical(img)
            data = img.get_fdata().astype(np.float32)
        except:
            data = np.zeros(TARGET_SHAPE, dtype=np.float32)

        # 1. Tight Cropping
        data = crop_to_brain(data)

        # 2. Resize 3D
        import torch.nn.functional as F
        tensor_data = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
        tensor_data = F.interpolate(tensor_data, size=TARGET_SHAPE, mode='trilinear', align_corners=False)
        data = tensor_data.squeeze(0).squeeze(0).numpy()

        # 3. Augmentation Avancée (Spécifique IRM)
        if self.augment:
            if random.random() > 0.5: 
                data = np.flip(data, axis=0).copy()
            if random.random() > 0.3:
                data = ndimage.rotate(data, random.uniform(-5, 5), axes=(0, 1), reshape=False, order=1)
            # Contraste aléatoire
            if random.random() > 0.5:
                data = data * random.uniform(0.8, 1.2)
            # Bruit Gaussien
            if random.random() > 0.5:
                data = data + np.random.normal(0, 0.05, data.shape).astype(np.float32)

        # 4. Normalisation
        std = data.std()
        if std > 1e-6:
            data = (data - data.mean()) / std 

        return torch.from_numpy(data).float().unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.long)

def build_dataset():
    df = pd.read_csv(CSV_PATH)
    df = df[df['original_group'].isin(['AD', 'CN'])].copy()
    id_to_class, id_to_subject = {}, {}
    for _, row in df.iterrows():
        match = re.search(r'I(\d+)', row['file_name'])
        if match:
            img_id = 'I' + match.group(1)
            id_to_class[img_id] = row['original_group']
            id_to_subject[img_id] = row['subject']

    paths, labels, subjects = [], [], []
    class_map = {'CN': 0, 'AD': 1}
    for fname in os.listdir(SKULL_STRIPPED_DIR):
        if fname.endswith(('.nii', '.nii.gz')):
            match = re.search(r'I(\d+)', fname)
            if match and ('I'+match.group(1)) in id_to_class:
                img_id = 'I' + match.group(1)
                paths.append(os.path.join(SKULL_STRIPPED_DIR, fname))
                labels.append(class_map[id_to_class[img_id]])
                subjects.append(id_to_subject[img_id])
    return np.array(paths), np.array(labels), np.array(subjects)

# ==========================================
# 📊 FONCTION D'ÉVALUATION AVEC TTA (réutilisable)
# ==========================================
def evaluate_with_tta(model, loader, criterion, split_name="Validation"):
    model.eval()
    all_preds, all_trues, all_probs = [], [], []
    total_loss, total_correct, total_count = 0.0, 0, 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            X_flipped = torch.flip(X, dims=[2])

            with torch.amp.autocast('cuda'):
                logits_normal  = model(X)
                probs_normal   = torch.softmax(logits_normal.float(), dim=1)

                logits_flipped = model(X_flipped)
                probs_flipped  = torch.softmax(logits_flipped.float(), dim=1)

                probs_final = (probs_normal + probs_flipped) / 2.0

                loss = criterion(logits_normal, y)

            total_loss    += loss.item() * X.size(0)
            total_correct += (probs_final.argmax(1) == y).sum().item()
            total_count   += y.size(0)

            all_preds.extend(probs_final.argmax(1).cpu().numpy())
            all_trues.extend(y.cpu().numpy())
            all_probs.extend(probs_final[:, 1].cpu().numpy())

    acc = accuracy_score(all_trues, all_preds) * 100
    f1  = f1_score(all_trues, all_preds, average='weighted') * 100
    auc = roc_auc_score(all_trues, all_probs) * 100
    cm  = confusion_matrix(all_trues, all_preds)

    print(f"\n📈 RÉSULTATS {split_name.upper()} (TTA) :")
    print(f"   Accuracy  : {acc:.2f}%")
    print(f"   F1-Score  : {f1:.2f}%")
    print(f"   AUC (ROC) : {auc:.2f}%")
    print(f"\n📋 Rapport de Classification ({split_name}) :")
    print(classification_report(all_trues, all_preds, target_names=['CN (Sains)', 'AD (Alzheimer)']))

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['CN', 'AD'])
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title(
        f'Matrice de Confusion — {split_name} (Focal Loss + TTA)\n'
        f'Acc: {acc:.1f}% | F1: {f1:.1f}% | AUC: {auc:.1f}%',
        fontweight='bold'
    )
    plt.tight_layout()
    fname_out = f'confusion_matrix_{split_name.lower()}.png'
    plt.savefig(fname_out, dpi=150)
    print(f"✅ Fichier '{fname_out}' sauvegardé !")

    return acc, f1, auc

# ==========================================
# 🚀 BOUCLE D'ENTRAÎNEMENT
# ==========================================
if __name__ == '__main__':
    all_paths, all_labels, all_subjects = build_dataset()
    
    subject_to_label = {s: l for s, l in zip(all_subjects, all_labels)}
    unique_subjects  = np.array(list(subject_to_label.keys()))
    unique_labels    = np.array([subject_to_label[s] for s in unique_subjects])

    # ── Split 80 / 10 / 10 au niveau sujet (anti data-leakage) ──────────────
    # Étape 1 : extraire 10% test d'abord
    trainval_subj, test_subj = train_test_split(
        unique_subjects, test_size=0.10, random_state=42, stratify=unique_labels
    )
    trainval_labels = np.array([subject_to_label[s] for s in trainval_subj])

    # Étape 2 : sur le reste (90%), extraire ~11.1% → donne 10% global pour val
    train_subj, val_subj = train_test_split(
        trainval_subj, test_size=0.111, random_state=42, stratify=trainval_labels
    )

    # Vérification anti data-leakage (3 assertions)
    assert len(set(train_subj) & set(val_subj))  == 0, 'DATA LEAKAGE train/val !'
    assert len(set(train_subj) & set(test_subj)) == 0, 'DATA LEAKAGE train/test !'
    assert len(set(val_subj)   & set(test_subj)) == 0, 'DATA LEAKAGE val/test !'

    train_mask = np.isin(all_subjects, train_subj)
    val_mask   = np.isin(all_subjects, val_subj)
    test_mask  = np.isin(all_subjects, test_subj)

    print(f"📊 Split volumes — Train: {train_mask.sum()} | Val: {val_mask.sum()} | Test: {test_mask.sum()}")

    train_ds = ADNI3DDataset(all_paths[train_mask], all_labels[train_mask], augment=True)
    val_ds   = ADNI3DDataset(all_paths[val_mask],   all_labels[val_mask],   augment=False)
    test_ds  = ADNI3DDataset(all_paths[test_mask],  all_labels[test_mask],  augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True, num_workers=8, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8, persistent_workers=True)

    model = InflatedResNet3D(num_classes=2).to(DEVICE)
    
    criterion = FocalLoss(alpha=0.25, gamma=2.0).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler('cuda')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_acc = 0.0

    print(f"\n🚀 Entraînement MASTER (Tight Crop + Focal Loss + Data Aug) sur {EPOCHS} époques...")
    print(f"{'Epoch':<6} | {'Temps':<6} | {'Train Loss':<10} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<8}")
    print("-" * 75)
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        
        # --- TRAIN ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(X)
                loss   = criterion(logits, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss    += loss.item() * X.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total   += X.size(0)

        epoch_train_loss = train_loss    / train_total
        epoch_train_acc  = train_correct / train_total * 100

        # --- VALIDATION ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    logits = model(X)
                    loss   = criterion(logits, y)
                
                val_loss    += loss.item() * X.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total   += y.size(0)

        epoch_val_loss = val_loss    / val_total
        epoch_val_acc  = val_correct / val_total * 100

        scheduler.step(epoch_val_acc)

        marker = " ⭐" if epoch_val_acc > best_acc else ""
        if epoch_val_acc > best_acc: 
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), "best_resnet3d_model.pth")
        
        epoch_time = time.time() - epoch_start
        print(f"{epoch:2d}/{EPOCHS}  | {epoch_time:4.1f}s | {epoch_train_loss:.4f}     | {epoch_train_acc:5.2f}%    | {epoch_val_loss:.4f}     | {epoch_val_acc:5.2f}%{marker}")

    total_time = (time.time() - start_time) / 60
    print("\n✅ Entraînement terminé en {:.1f} minutes !".format(total_time))

    # Charger le meilleur modèle pour les évaluations
    model.load_state_dict(torch.load("best_resnet3d_model.pth", weights_only=True))

    # ==========================================
    # 📊 ÉVALUATION SUR VALIDATION (TTA)
    # ==========================================
    val_acc, val_f1, val_auc = evaluate_with_tta(model, val_loader, criterion, split_name="Validation")

    # ==========================================
    # 🧪 ÉVALUATION FINALE SUR TEST (TTA)
    # ==========================================
    print("\n" + "=" * 75)
    print("🧪 ÉVALUATION FINALE SUR LE SET DE TEST (données jamais vues)")
    print("=" * 75)
    test_acc, test_f1, test_auc = evaluate_with_tta(model, test_loader, criterion, split_name="Test")

    # ==========================================
    # 📋 RÉSUMÉ COMPARATIF Val vs Test
    # ==========================================
    print("\n" + "=" * 75)
    print("📋 RÉSUMÉ COMPARATIF")
    print("=" * 75)
    print(f"{'Métrique':<15} {'Validation':>12} {'Test':>12}")
    print("-" * 40)
    print(f"{'Accuracy':<15} {val_acc:>11.2f}% {test_acc:>11.2f}%")
    print(f"{'F1-Score':<15} {val_f1:>11.2f}% {test_f1:>11.2f}%")
    print(f"{'AUC (ROC)':<15} {val_auc:>11.2f}% {test_auc:>11.2f}%")
    print("=" * 75)
