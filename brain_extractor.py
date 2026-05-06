import os
import glob
import subprocess

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
INPUT_DIR = r'C:\Users\youne\Desktop\ADNI_PREPROCESSED'
OUTPUT_DIR = r'C:\Users\youne\Desktop\ADNI_SKULL_STRIPPED'

def run_resilient_extraction():
    print("🚀 Démarrage du mode de secours ultra-robuste...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Recherche récursive de tous les fichiers .nii
    all_files = glob.glob(os.path.join(INPUT_DIR, "**/*.nii"), recursive=True)
    print(f"📂 {len(all_files)} fichiers détectés au total.")

    # 2. Filtrage des fichiers déjà traités
    to_process = []
    for f in all_files:
        out_name = f"brain_{os.path.basename(f)}.gz"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        if not os.path.exists(out_path):
            to_process.append((f, out_path))

    print(f"⚡ {len(to_process)} fichiers restants à traiter.")
    
    if not to_process:
        print("✅ Tout est déjà traité ! Tu peux passer à l'entraînement de ton CNN 3D.")
        return

    # 3. Boucle de traitement par commande système
    for i, (in_path, out_path) in enumerate(to_process):
        print(f"\n🔄 [{i+1}/{len(to_process)}] Traitement de : {os.path.basename(in_path)}")
        
        # On utilise la commande qui a fonctionné au début
        # -device cuda:0 pour ta RTX 4070
        cmd = [
            "hd-bet",
            "-i", in_path,
            "-o", out_path,
            "-device", "cuda:0",
            "--disable_tta"
        ]
        
        try:
            # On lance la commande et on attend qu'elle finisse
            subprocess.run(cmd, check=True)
            print(f"✅ Terminé : {os.path.basename(out_path)}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur critique sur ce fichier : {e}")
            continue

    print("\n🎉 MISSION ACCOMPLIE ! Ton dataset ADNI est enfin prêt.")

if __name__ == "__main__":
    run_resilient_extraction()