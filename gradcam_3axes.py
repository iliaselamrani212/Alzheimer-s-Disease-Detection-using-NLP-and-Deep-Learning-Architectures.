"""
🔍 Grad-CAM 3D MULTI-AXES - Explainable AI (XAI)
Visualisation complète : Axial + Sagittal + Coronal avec plusieurs coupes par plan.

Permet de voir exactement où le modèle regarde dans tout le volume 3D.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import nibabel as nib
from sklearn.model_selection import train_test_split

# IMPORTATION VERS TON FICHIER PRINCIPAL
from st_2 import InflatedResNet3D, TARGET_SHAPE, DEVICE, build_dataset, crop_to_brain

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
MODEL_WEIGHTS = 'best_resnet3d_model.pth'
N_SLICES_PER_AXIS = 5  # Nombre de coupes par plan (axial, sagittal, coronal)

# ==========================================
# 🧠 CLASSE GRAD-CAM 3D
# ==========================================
class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, x, class_idx=None):
        self.model.eval()
        logits = self.model(x)
        
        pred_class = logits.argmax(1).item()
        probs = torch.softmax(logits.float(), dim=1)[0].detach().cpu().numpy()
        
        if class_idx is None:
            class_idx = pred_class
            
        self.model.zero_grad()
        target = logits[0, class_idx]
        target.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3, 4])
        activations = self.activations.detach()
        
        for i in range(activations.size(1)):
            activations[:, i, :, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
            
        return heatmap.cpu().numpy(), pred_class, probs

# ==========================================
# 🖼️ UTILITAIRES DE PRÉPROCESSING
# ==========================================
def preprocess_image(path):
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    data = img.get_fdata().astype(np.float32)
    
    # 1. Tight Cropping (comme à l'entraînement)
    data = crop_to_brain(data)
    
    # 2. Resize 3D
    if data.shape != TARGET_SHAPE:
        tensor_data = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
        tensor_data = F.interpolate(tensor_data, size=TARGET_SHAPE, mode='trilinear', align_corners=False)
        data = tensor_data.squeeze(0).squeeze(0).numpy()
        
    # 3. Normalisation (comme à l'entraînement)
    mean, std = data.mean(), data.std()
    if std > 1e-6: 
        data = (data - mean) / std
    
    tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    return tensor, data

# ==========================================
# 🎨 VISUALISATION MULTI-AXES
# ==========================================
def plot_gradcam_3axes(original_vol, heatmap_vol, pred_class, probs, true_label, filename):
    """
    Affiche le Grad-CAM sur les 3 plans anatomiques avec plusieurs coupes par plan.
    
    - Axial   : vue du dessus (XY)        → axe Z
    - Sagittal: vue de profil (YZ)        → axe X
    - Coronal : vue de face (XZ)          → axe Y
    """
    class_names = ['CN (Sain)', 'AD (Alzheimer)']
    pred_name   = class_names[pred_class]
    true_name   = class_names[true_label]
    confidence  = probs[pred_class] * 100
    is_correct  = (pred_class == true_label)
    
    # Sélection des coupes (centrées dans le volume cerveau)
    # On évite les bords (10%-90%) car ils contiennent peu de cerveau
    sx, sy, sz = original_vol.shape
    axial_slices    = np.linspace(int(sz * 0.30), int(sz * 0.75), N_SLICES_PER_AXIS).astype(int)
    sagittal_slices = np.linspace(int(sx * 0.30), int(sx * 0.70), N_SLICES_PER_AXIS).astype(int)
    coronal_slices  = np.linspace(int(sy * 0.30), int(sy * 0.75), N_SLICES_PER_AXIS).astype(int)
    
    # Configuration du plot : 3 lignes (1 par axe), N_SLICES colonnes
    fig, axes = plt.subplots(3, N_SLICES_PER_AXIS, figsize=(4 * N_SLICES_PER_AXIS, 12))
    
    # Titre global
    correct_marker = "✅ CORRECT" if is_correct else "❌ ERREUR"
    title_color    = 'green' if is_correct else 'red'
    fig.suptitle(
        f'Grad-CAM 3D Multi-Axes  |  Vrai: {true_name}  |  Prédit: {pred_name} ({confidence:.1f}%)  |  {correct_marker}',
        fontsize=15, fontweight='bold', color=title_color, y=0.995
    )
    
    # ───────────────── AXIAL (vue du dessus) ─────────────────
    for i, z in enumerate(axial_slices):
        slice_img     = np.rot90(original_vol[:, :, z])
        slice_heatmap = np.rot90(heatmap_vol[:, :, z])
        
        axes[0, i].imshow(slice_img, cmap='bone')
        axes[0, i].imshow(slice_heatmap, cmap='jet', alpha=0.45)
        axes[0, i].set_title(f'Axial z={z}', fontweight='bold', fontsize=11)
        axes[0, i].axis('off')
    
    axes[0, 0].set_ylabel('AXIAL\n(vue du dessus)', fontsize=13, fontweight='bold', 
                          rotation=90, labelpad=20, color='#2E86AB')
    
    # ───────────────── SAGITTAL (vue de profil) ─────────────────
    for i, x in enumerate(sagittal_slices):
        slice_img     = np.rot90(original_vol[x, :, :])
        slice_heatmap = np.rot90(heatmap_vol[x, :, :])
        
        axes[1, i].imshow(slice_img, cmap='bone')
        axes[1, i].imshow(slice_heatmap, cmap='jet', alpha=0.45)
        axes[1, i].set_title(f'Sagittal x={x}', fontweight='bold', fontsize=11)
        axes[1, i].axis('off')
    
    axes[1, 0].set_ylabel('SAGITTAL\n(vue de profil)', fontsize=13, fontweight='bold',
                          rotation=90, labelpad=20, color='#A23B72')
    
    # ───────────────── CORONAL (vue de face) ─────────────────
    for i, y in enumerate(coronal_slices):
        slice_img     = np.rot90(original_vol[:, y, :])
        slice_heatmap = np.rot90(heatmap_vol[:, y, :])
        
        axes[2, i].imshow(slice_img, cmap='bone')
        axes[2, i].imshow(slice_heatmap, cmap='jet', alpha=0.45)
        axes[2, i].set_title(f'Coronal y={y}', fontweight='bold', fontsize=11)
        axes[2, i].axis('off')
    
    axes[2, 0].set_ylabel('CORONAL\n(vue de face)', fontsize=13, fontweight='bold',
                          rotation=90, labelpad=20, color='#F18F01')
    
    # Légende couleur
    fig.text(0.5, 0.01, 
             '🔥 ROUGE = Forte attention du modèle  |  🟢 VERT/JAUNE = Attention modérée  |  🔵 BLEU = Faible attention',
             ha='center', fontsize=11, style='italic')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Visualisation Multi-Axes sauvegardée : '{filename}'")

# ==========================================
# 🚀 EXÉCUTION
# ==========================================
if __name__ == '__main__':
    print("=" * 70)
    print("🔍 GRAD-CAM 3D MULTI-AXES — EXPLAINABLE AI")
    print("=" * 70)
    
    # 1. Reconstruction du dataset et split (identique à l'entraînement)
    print("\n📂 Chargement du dataset et reconstruction du split...")
    all_paths, all_labels, all_subjects = build_dataset()
    subject_to_label = {s: l for s, l in zip(all_subjects, all_labels)}
    unique_subjects  = np.array(list(subject_to_label.keys()))
    unique_labels    = np.array([subject_to_label[s] for s in unique_subjects])

    trainval_subj, test_subj = train_test_split(
        unique_subjects, test_size=0.10, random_state=42, stratify=unique_labels
    )
    trainval_labels = np.array([subject_to_label[s] for s in trainval_subj])
    train_subj, val_subj = train_test_split(
        trainval_subj, test_size=0.111, random_state=42, stratify=trainval_labels
    )
    val_mask = np.isin(all_subjects, val_subj)
    val_paths  = all_paths[val_mask]
    val_labels = all_labels[val_mask]

    # 2. Sélection d'un patient AD aléatoire de la validation
    ad_indices = np.where(val_labels == 1)[0]
    if len(ad_indices) == 0:
        print("❌ Erreur : Aucun patient AD trouvé.")
        exit()
        
    index_aleatoire   = np.random.choice(ad_indices)
    test_image_path   = val_paths[index_aleatoire]
    test_filename     = os.path.basename(test_image_path)
    true_label        = val_labels[index_aleatoire]
    print(f"✅ Patient sélectionné : {test_filename}")
    print(f"   Vraie classe : {'AD (Alzheimer)' if true_label == 1 else 'CN (Sain)'}")

    # 3. Chargement du modèle entraîné
    print("\n🧠 Chargement du modèle MASTER...")
    model = InflatedResNet3D(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, weights_only=True))
    model.eval()
    
    # 4. Setup du Grad-CAM sur layer3 (équilibre détail/contexte)
    target_layer = model.model.layer3[-1].conv2
    gradcam = GradCAM3D(model, target_layer)
    
    # 5. Prétraitement et inférence
    print("\n🔬 Génération de la heatmap 3D...")
    input_tensor, original_vol = preprocess_image(test_image_path)
    heatmap, pred_class, probs = gradcam.generate_heatmap(input_tensor, class_idx=None)
    
    # 6. Resize de la heatmap au volume complet
    heatmap_tensor   = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
    heatmap_resized  = F.interpolate(heatmap_tensor, size=TARGET_SHAPE, mode='trilinear', align_corners=False)
    heatmap_final    = heatmap_resized.squeeze().numpy()
    
    # 7. Affichage des résultats numériques
    class_names = ['CN (Sain)', 'AD (Alzheimer)']
    print(f"\n📊 RÉSULTAT DE LA PRÉDICTION :")
    print(f"   Probabilité CN : {probs[0]*100:.2f}%")
    print(f"   Probabilité AD : {probs[1]*100:.2f}%")
    print(f"   → Prédiction   : {class_names[pred_class]}")
    print(f"   → Vraie classe : {class_names[true_label]}")
    print(f"   → Statut       : {'✅ CORRECT' if pred_class == true_label else '❌ ERREUR'}")

    # 8. Génération de la visualisation multi-axes
    print("\n🎨 Création de la visualisation 3 axes anatomiques...")
    output_filename = 'gradcam_3axes_multi_slices.png'
    plot_gradcam_3axes(original_vol, heatmap_final, pred_class, probs, true_label, output_filename)
    
    print("\n" + "=" * 70)
    print(f"✅ TERMINÉ ! Visualisation : {output_filename}")
    print("=" * 70)
