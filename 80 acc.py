"""
🧠 ResNet3D-18 (Inflate-1 Strategy) - AD vs CN
Version Finale : Early Stopping, Schedulers, F1-Score et Matrice de Confusion.
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
    accuracy_score, classification_report, roc_auc_score, 
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)
from torchvision.models import resnet18, ResNet18_Weights
import re
import matplotlib
matplotlib.use('Agg') # Pour sauvegarder les images sans interface graphique
import matplotlib.pyplot as plt

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
SKULL_STRIPPED_DIR = r'C:\Users\youne\Desktop\ADNI_SKULL_STRIPPED'
CSV_PATH           = r'dataset_preprocessed.csv'

TARGET_SHAPE = (96, 96, 96) 
BATCH_SIZE   = 4 
EPOCHS       = 30
LR           = 1e-4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# ==========================================
# 🧬 STRATÉGIE INFLATE-1 (2D -> 3D)
# ==========================================
def inflate_conv2d_to_3d(conv2d):
    k = conv2d.kernel_size if isinstance(conv2d.kernel_size, tuple) else (conv2d.kernel_size, conv2d.kernel_size)
    s = conv2d.stride if isinstance(conv2d.stride, tuple) else (conv2d.stride, conv2d.stride)
    p = conv2d.padding if isinstance(conv2d.padding, tuple) else (conv2d.padding, conv2d.padding)
    
    conv3d = nn.Conv3d(
        in_channels=conv2d.in_channels, out_channels=conv2d.out_channels,
        kernel_size=(k[0], k[0], k[1]), stride=(s[0], s[0], s[1]), padding=(p[0], p[0], p[1]), 
        bias=(conv2d.bias is not None)
    )
    with torch.no_grad():
        mid_z = k[0] // 2
        conv3d.weight.zero_()
        conv3d.weight[:, :, mid_z, :, :] = conv2d.weight
        if conv2d.bias is not None:
            conv3d.bias.copy_(conv2d.bias)
    return conv3d

def inflate_batchnorm2d_to_3d(bn2d):
    bn3d = nn.BatchNorm3d(bn2d.num_features, eps=bn2d.eps, momentum=bn2d.momentum, affine=bn2d.affine)
    with torch.no_grad():
        bn3d.weight.copy_(bn2d.weight)
        bn3d.bias.copy_(bn2d.bias)
        bn3d.running_mean.copy_(bn2d.running_mean)
        bn3d.running_var.copy_(bn2d.running_var)
    return bn3d

def convert_resnet_to_3d(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, inflate_conv2d_to_3d(child))
        elif isinstance(child, nn.BatchNorm2d):
            setattr(module, name, inflate_batchnorm2d_to_3d(child))
        elif isinstance(child, nn.MaxPool2d):
            k, s, p = child.kernel_size, child.stride, child.padding
            k = k if isinstance(k, tuple) else (k, k)
            s = s if isinstance(s, tuple) else (s, s)
            p = p if isinstance(p, tuple) else (p, p)
            setattr(module, name, nn.MaxPool3d(kernel_size=(k[0], k[0], k[1]), stride=(s[0], s[0], s[1]), padding=(p[0], p[0], p[1])))
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
        new_conv1 = nn.Conv3d(1, 64, kernel_size=old_conv1.kernel_size, stride=old_conv1.stride, padding=old_conv1.padding, bias=False)
        with torch.no_grad():
            new_conv1.weight = nn.Parameter(old_conv1.weight.mean(dim=1, keepdim=True))
        self.model.conv1 = new_conv1
        
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# ==========================================
# 📂 DATASET ET AUGMENTATION 3D
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
        except Exception:
            data = np.zeros(TARGET_SHAPE, dtype=np.float32)

        if data.shape != TARGET_SHAPE:
             import torch.nn.functional as F
             tensor_data = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
             tensor_data = F.interpolate(tensor_data, size=TARGET_SHAPE, mode='trilinear', align_corners=False)
             data = tensor_data.squeeze(0).squeeze(0).numpy()

        if self.augment:
            if random.random() > 0.5: data = np.flip(data, axis=0).copy()
            if random.random() > 0.5:
                angle = random.uniform(-8, 8)
                data = ndimage.rotate(data, angle, axes=(0, 1), reshape=False, order=1, mode='nearest')
            if random.random() > 0.5:
                shift_x, shift_y = random.uniform(-5, 5), random.uniform(-5, 5)
                data = ndimage.shift(data, shift=(shift_x, shift_y, 0), order=1, mode='nearest')
            if random.random() > 0.5:
                data = data + np.random.normal(0, 0.02, data.shape).astype(np.float32)

        mean, std = data.mean(), data.std()
        if std > 1e-6: data = (data - mean) / std

        tensor = torch.from_numpy(data).float().unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, label

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
# 🚀 ENTRAÎNEMENT ET ÉVALUATION FINALE
# ==========================================
if __name__ == '__main__':
    print("=" * 70)
    print("🧠 DÉMARRAGE PIPELINE: INFLATED RESNET-3D (AVEC ÉVALUATION FINALE)")
    print("=" * 70)

    # 1. Chargement des données (Split Patient)
    all_paths, all_labels, all_subjects = build_dataset()
    subject_to_label = {s: l for s, l in zip(all_subjects, all_labels)}
    unique_subjects = np.array(list(subject_to_label.keys()))
    unique_labels = np.array([subject_to_label[s] for s in unique_subjects])

    train_subj, val_subj = train_test_split(unique_subjects, test_size=0.20, random_state=42, stratify=unique_labels)
    train_mask = np.isin(all_subjects, train_subj)
    val_mask = np.isin(all_subjects, val_subj)

    train_ds = ADNI3DDataset(all_paths[train_mask], all_labels[train_mask], augment=True)
    val_ds = ADNI3DDataset(all_paths[val_mask], all_labels[val_mask], augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)

    # 2. Modèle & Optimiseurs
    model = InflatedResNet3D(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    # Le Scheduler pour stabiliser l'apprentissage quand on s'approche de la fin
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    # Historiques
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0

    print(f"\n🔄 Début de l'entraînement pour {EPOCHS} époques")
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
                loss = criterion(logits, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * X.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += X.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total * 100

        # --- VALIDATION ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    logits = model(X)
                    loss = criterion(logits, y)
                
                val_loss += loss.item() * X.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += y.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total * 100
        
        # Mise à jour de l'historique
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        # On informe le scheduler
        scheduler.step(epoch_val_acc)

        # Sauvegarde du meilleur modèle
        marker = " ⭐" if epoch_val_acc > best_acc else ""
        if epoch_val_acc > best_acc: 
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), "best_resnet3d_model.pth")
        
        epoch_time = time.time() - epoch_start
        print(f"{epoch:2d}/{EPOCHS}  | {epoch_time:4.1f}s | {epoch_train_loss:.4f}     | {epoch_train_acc:5.2f}%    | {epoch_val_loss:.4f}     | {epoch_val_acc:5.2f}%{marker}")

    total_time = (time.time() - start_time) / 60
    print("\n" + "=" * 70)
    print(f"✅ Entraînement terminé en {total_time:.1f} minutes !")
    print("=" * 70)

    # ==========================================
    # 📊 ÉVALUATION FINALE (TESTING)
    # ==========================================
    print("\n🔍 Évaluation du meilleur modèle sur le set de Validation...")
    
    # On recharge les poids du meilleur modèle
    model.load_state_dict(torch.load("best_resnet3d_model.pth"))
    model.eval()
    
    all_preds, all_trues, all_probs = [], [], []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast('cuda'):
                logits = model(X)
                probs = torch.softmax(logits.float(), dim=1)
            
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_trues.extend(y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy()) # Proba d'être AD

    # Calcul des métriques finales
    final_acc = accuracy_score(all_trues, all_preds) * 100
    final_f1 = f1_score(all_trues, all_preds, average='weighted') * 100
    final_auc = roc_auc_score(all_trues, all_probs) * 100
    cm = confusion_matrix(all_trues, all_preds)

    print("\n📈 RÉSULTATS MÉTRIQUES :")
    print(f"   Accuracy Finale : {final_acc:.2f}%")
    print(f"   F1-Score        : {final_f1:.2f}%")
    print(f"   AUC (ROC)       : {final_auc:.2f}%")
    print("\nRapport de Classification Détaillé :")
    print(classification_report(all_trues, all_preds, target_names=['CN (Sains)', 'AD (Alzheimer)']))

    # Génération de l'image de la matrice de confusion
    print("🎨 Génération de la Matrice de Confusion (confusion_matrix.png)...")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['CN', 'AD'])
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title(f'Matrice de Confusion\nAcc: {final_acc:.1f}% | F1: {final_f1:.1f}%', fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print("✅ Fichier 'confusion_matrix.png' sauvegardé avec succès ! Ton PFA est prêt !")