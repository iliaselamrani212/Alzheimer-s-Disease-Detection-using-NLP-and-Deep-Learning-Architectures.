"""
Crée le CSV final mappant les fichiers .nii à leurs labels
À partir du CSV ADNI officiel (déjà fourni)

Usage:
    python create_csv_v2.py
"""

import os
import re
import pandas as pd
from tqdm import tqdm

# ============================================================
# CONFIGURATION - ADAPTE LES CHEMINS
# ============================================================
EXTRACT_DIR = r'C:\Users\youne\Desktop\ADNI_PREPROCESSED'
ADNI_CSV    = r'C:\Users\youne\Downloads\sclaed_5_05_2026.csv'  # ← le CSV ADNI officiel
OUTPUT_CSV  = r'dataset_preprocessed.csv'

print('=' * 70)
print('  📋 CRÉATION DU CSV FINAL (méthode CSV ADNI)')
print('=' * 70)

# ============================================================
# 1. CHARGER LE CSV ADNI OFFICIEL
# ============================================================
print(f'\n📥 Chargement du CSV ADNI : {ADNI_CSV}')

if not os.path.exists(ADNI_CSV):
    print(f'❌ ERREUR : Le CSV ADNI n\'existe pas à : {ADNI_CSV}')
    print('   Vérifie le chemin et adapte la variable ADNI_CSV')
    exit()

df_adni = pd.read_csv(ADNI_CSV)
print(f'✅ CSV ADNI chargé : {len(df_adni)} lignes')
print(f'   Colonnes : {df_adni.columns.tolist()}')

# Aperçu
print(f'\n   5 premières lignes :')
print(df_adni.head().to_string())

# Distribution dans le CSV ADNI
print(f'\n📊 Distribution dans le CSV ADNI :')
print(df_adni['Group'].value_counts())

# ============================================================
# 2. TROUVER TOUS LES FICHIERS .nii
# ============================================================
print(f'\n🔍 Recherche des fichiers .nii dans {EXTRACT_DIR}...')

nii_files = []
for root, dirs, files in os.walk(EXTRACT_DIR):
    for f in files:
        if f.endswith('.nii') or f.endswith('.nii.gz'):
            nii_files.append(os.path.join(root, f))

print(f'   ✅ {len(nii_files)} fichiers .nii trouvés')

# ============================================================
# 3. EXTRAIRE LES IMAGE DATA ID DEPUIS LES NOMS
# ============================================================
print(f'\n🔗 Matching NIfTI ↔ CSV ADNI...')

# Créer un dict : Image Data ID → row du CSV
adni_dict = {}
for idx, row in df_adni.iterrows():
    img_id = str(row['Image Data ID']).strip()
    adni_dict[img_id] = row

# Pour chaque fichier .nii, extraire l'ID et matcher
mapping = []
unmatched = []

for nii_path in tqdm(nii_files, desc='Matching'):
    nii_name = os.path.basename(nii_path)
    
    # Pattern pour extraire I123456 du nom de fichier
    # Exemples possibles :
    #   ADNI_002_S_0413_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_2_Br_..._I63879.nii
    #   *_I98902.nii
    match = re.search(r'_I(\d+)', nii_name)
    
    if match:
        image_id = 'I' + match.group(1)
        
        if image_id in adni_dict:
            row = adni_dict[image_id]
            mapping.append({
                'file_name': nii_name,
                'path': nii_path,
                'image_id': image_id,
                'subject': row['Subject'],
                'original_group': row['Group'],
                'sex': row.get('Sex', ''),
                'age': row.get('Age', ''),
                'visit': row.get('Visit', ''),
                'description': row.get('Description', ''),
                'acq_date': row.get('Acq Date', '')
            })
        else:
            unmatched.append((nii_name, image_id, 'ID not in CSV'))
    else:
        unmatched.append((nii_name, None, 'No ID extracted'))

print(f'\n   ✅ Matchés    : {len(mapping)}')
print(f'   ❌ Non matchés : {len(unmatched)}')

if unmatched[:5]:
    print(f'\n   Premiers non matchés :')
    for name, uid, reason in unmatched[:5]:
        print(f'      {name[:60]}... → {reason}')

# ============================================================
# 4. CRÉER LE DATAFRAME
# ============================================================
df = pd.DataFrame(mapping)

if len(df) == 0:
    print('\n❌ ERREUR : Aucun fichier matché !')
    print('   Le CSV ADNI et les .nii ne correspondent pas.')
    print('\n   Diagnostic :')
    print(f'   - CSV ADNI a {len(df_adni)} lignes avec ID comme : {df_adni["Image Data ID"].iloc[0]}')
    if unmatched:
        print(f'   - Fichier .nii exemple : {unmatched[0][0]}')
    exit()

# Mapping label numérique
LABEL_MAP = {
    'CN': 0,
    'AD': 1,
    'LMCI': 2,
    'EMCI': 3,
    'MCI': 4,
    'SMC': 5
}
df['label_num'] = df['original_group'].map(LABEL_MAP)

# ============================================================
# 5. STATS COMPLÈTES
# ============================================================
print('\n' + '=' * 70)
print('  📊 STATISTIQUES DU DATASET FINAL')
print('=' * 70)

print(f'\n📊 Total volumes    : {len(df)}')
print(f'📊 Sujets uniques   : {df["subject"].nunique()}')

print(f'\n📊 Distribution par classe :')
group_counts = df['original_group'].value_counts()
for group, count in group_counts.items():
    n_subj = df[df['original_group'] == group]['subject'].nunique()
    print(f'   {group:5s} : {count:5d} volumes  ({n_subj:4d} sujets uniques)')

# Cohérence sujet-label
print(f'\n🔍 Vérification cohérence sujet ↔ label...')
subject_groups = df.groupby('subject')['original_group'].nunique()
multi_label = subject_groups[subject_groups > 1]

if len(multi_label) == 0:
    print(f'   ✅ Tous les sujets ont 1 seul label')
else:
    print(f'   ⚠️ {len(multi_label)} sujets ont des labels différents')
    print(f'   → On gardera le 1er label par sujet')

# Distribution par visite (si disponible)
if 'visit' in df.columns and df['visit'].notna().any():
    print(f'\n📊 Distribution par visite (top 5) :')
    visit_counts = df['visit'].value_counts().head(5)
    for visit, cnt in visit_counts.items():
        print(f'   {visit:10s} : {cnt:4d}')

# Vérifier que tous les paths existent
print(f'\n🔍 Vérification des paths...')
existing = sum(1 for p in df['path'] if os.path.exists(p))
print(f'   ✅ Existent : {existing}/{len(df)}')

# ============================================================
# 6. SAUVEGARDE
# ============================================================
df.to_csv(OUTPUT_CSV, index=False)
print(f'\n💾 CSV sauvegardé : {OUTPUT_CSV}')
print(f'   Colonnes : {df.columns.tolist()}')

# Aperçu
print(f'\n📄 5 premières lignes :')
print(df[['file_name', 'subject', 'original_group', 'visit', 'age']].head().to_string())

print('\n' + '=' * 70)
print('  ✅ TERMINÉ')
print('  ➡️  Prochaine étape : extraction des slices')
print('=' * 70)
