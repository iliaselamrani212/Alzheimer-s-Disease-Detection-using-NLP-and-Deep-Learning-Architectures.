"""
Extraction parallèle des ZIP ADNI preprocessed
Utilise tous les cores CPU + lecture/écriture optimisée

Usage:
    cd C:\\chemin\\vers\\le\\script
    python extract_zips_fast.py
"""

import os
import zipfile
import time
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# CONFIGURATION
# ============================================================
EXTRACT_DIR = r'C:\Users\youne\Desktop\ADNI_PREPROCESSED'
ZIP_DIR     = r'C:\Users\youne\Downloads'

ZIPS_TO_EXTRACT = [
    'sclaed.zip',
    'sclaed1.zip',
    'sclaed2.zip',
    'sclaed3.zip',
    'sclaed_IDA_Metadata.zip'
]

# ============================================================
# FONCTION : Décompresse 1 fichier d'un ZIP (utilisé par les threads)
# ============================================================
def extract_one_member(args):
    """Extrait UN fichier d'un ZIP (appelé en parallèle)"""
    zip_path, member, target_dir = args
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extract(member, target_dir)
        return True, None
    except Exception as e:
        return False, str(e)


# ============================================================
# FONCTION : Décompresse 1 ZIP entier en multi-thread
# ============================================================
def extract_zip_parallel(zip_path, extract_dir, n_threads=8):
    """
    Décompresse 1 ZIP en utilisant plusieurs threads I/O
    (les threads sont efficaces pour I/O même avec GIL Python)
    """
    if not os.path.exists(zip_path):
        return zip_path, 0, 0, 'NOT_FOUND'
    
    zip_name = os.path.basename(zip_path)
    print(f'\n📦 Démarrage : {zip_name}')
    start = time.time()
    
    # Lister tous les membres
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        total_size = sum(zf.getinfo(m).file_size for m in members)
    
    print(f'   Fichiers à extraire : {len(members):,}')
    print(f'   Taille totale       : {total_size / 1e9:.2f} GB')
    print(f'   Threads             : {n_threads}')
    
    # Créer la liste de tâches
    tasks = [(zip_path, m, extract_dir) for m in members]
    
    success = 0
    errors = 0
    
    # Décompression parallèle avec ThreadPool
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(extract_one_member, task) for task in tasks]
        
        # Progression en temps réel
        last_print = time.time()
        for i, future in enumerate(as_completed(futures), 1):
            ok, err = future.result()
            if ok:
                success += 1
            else:
                errors += 1
            
            # Affichage progression toutes les 2 secondes
            now = time.time()
            if now - last_print > 2 or i == len(futures):
                pct = i / len(futures) * 100
                speed = i / (now - start) if now - start > 0 else 0
                eta = (len(futures) - i) / speed if speed > 0 else 0
                print(f'   {zip_name}: {i:,}/{len(futures):,} '
                      f'({pct:.1f}%) | {speed:.0f} files/s | ETA: {eta:.0f}s',
                      end='\r')
                last_print = now
    
    elapsed = time.time() - start
    print()  # newline
    print(f'   ✅ {zip_name} terminé en {elapsed:.1f}s')
    print(f'      Succès : {success:,} | Erreurs : {errors}')
    
    return zip_path, success, errors, elapsed


# ============================================================
# MAIN
# ============================================================
def main():
    print('=' * 70)
    print('  🚀 EXTRACTION PARALLÈLE ADNI PREPROCESSED')
    print('=' * 70)
    
    n_cores = cpu_count()
    print(f'\n💻 Configuration système :')
    print(f'   CPU cores      : {n_cores}')
    print(f'   ZIP source     : {ZIP_DIR}')
    print(f'   Destination    : {EXTRACT_DIR}')
    
    # Créer dossier
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    
    # Vérifier ZIPs existent
    print(f'\n📦 Vérification des ZIPs :')
    valid_zips = []
    total_size = 0
    for zip_name in ZIPS_TO_EXTRACT:
        zip_path = os.path.join(ZIP_DIR, zip_name)
        if os.path.exists(zip_path):
            size = os.path.getsize(zip_path)
            total_size += size
            print(f'   ✅ {zip_name:30s} ({size / 1e9:.2f} GB)')
            valid_zips.append(zip_path)
        else:
            print(f'   ❌ {zip_name:30s} INTROUVABLE')
    
    if not valid_zips:
        print('\n❌ Aucun ZIP trouvé. Vérifie ZIP_DIR.')
        return
    
    print(f'\n📊 Total à décompresser : {total_size / 1e9:.2f} GB')
    
    # ============================================================
    # STRATÉGIE D'EXTRACTION
    # ============================================================
    # Pour 5 ZIPs sur un système à N cores :
    # - On lance les ZIPs en SÉQUENCE (1 à la fois)
    # - Mais CHAQUE zip utilise N threads pour ses fichiers internes
    # 
    # Pourquoi ? Décompresser 5 ZIPs simultanément sature le disque
    # qui devient le bottleneck (pas le CPU)
    # ============================================================
    
    threads_per_zip = max(8, n_cores)  # 8 threads min, ou plus si possible
    
    print(f'\n🔧 Stratégie : {threads_per_zip} threads par ZIP')
    print(f'   (les ZIPs sont traités séquentiellement pour ne pas saturer le disque)\n')
    
    global_start = time.time()
    total_success = 0
    total_errors = 0
    
    for zip_path in valid_zips:
        _, success, errors, _ = extract_zip_parallel(
            zip_path, EXTRACT_DIR, n_threads=threads_per_zip
        )
        total_success += success
        total_errors += errors
    
    global_elapsed = time.time() - global_start
    
    # ============================================================
    # RÉSUMÉ FINAL
    # ============================================================
    print('\n' + '=' * 70)
    print('  ✅ EXTRACTION TERMINÉE')
    print('=' * 70)
    print(f'  Fichiers extraits : {total_success:,}')
    print(f'  Erreurs           : {total_errors}')
    print(f'  Temps total       : {global_elapsed:.1f}s ({global_elapsed/60:.1f} min)')
    print(f'  Vitesse moyenne   : {total_success / global_elapsed:.0f} fichiers/s')
    print(f'  Destination       : {EXTRACT_DIR}')
    
    # Stats du dossier de sortie
    print(f'\n📊 Contenu du dossier de sortie :')
    n_files = 0
    n_size = 0
    for root, dirs, files in os.walk(EXTRACT_DIR):
        for f in files:
            n_files += 1
            try:
                n_size += os.path.getsize(os.path.join(root, f))
            except:
                pass
    print(f'   Fichiers : {n_files:,}')
    print(f'   Taille   : {n_size / 1e9:.2f} GB')
    
    # Détecter ce qu'il y a dedans
    extensions = {}
    for root, dirs, files in os.walk(EXTRACT_DIR):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            extensions[ext] = extensions.get(ext, 0) + 1
    
    if extensions:
        print(f'\n   Types de fichiers :')
        for ext, cnt in sorted(extensions.items(), key=lambda x: -x[1])[:10]:
            print(f'      {ext or "(no ext)":10s} : {cnt:,}')


if __name__ == '__main__':
    # ⚠️ Sous Windows, le multiprocessing nécessite ce guard
    main()
