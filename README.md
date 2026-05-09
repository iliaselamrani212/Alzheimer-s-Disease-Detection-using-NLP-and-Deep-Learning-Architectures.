# Alzheimer’s Disease Detection using NLP and Deep Learning Architectures.

Ce dépôt contient le pipeline complet de préparation des données (Data
Preprocessing) pour un projet de détection de la maladie d'Alzheimer
basé sur des volumes IRM 3D issus de la base de données ADNI.

L'objectif de ce pipeline est de transformer des archives brutes
massives en un dataset propre, labellisé et "skull-stripped" (sans boîte
crânienne), prêt à être ingéré par une architecture **CNN 3D (Inflated
ResNet-18)**.

## 🚀 Fonctionnalités du Pipeline

1.  **Extraction massive et multithreadée** des données IRM.
2.  **Génération automatique d'un dataset structuré (CSV)** pour lier
    les images aux diagnostics cliniques (AD vs CN).
3.  **Extraction cérébrale (Skull-Stripping)** accélérée par GPU via
    HD-BET pour isoler les tissus pertinents.

## 📋 Prérequis

Pour exécuter ce pipeline, vous aurez besoin de : \* Python 3.8+ \* Une
carte graphique NVIDIA compatible CUDA (ex: RTX 4070) avec au moins 8 Go
de VRAM. \* Les bibliothèques listées ci-dessous :

``` bash
pip install pandas nibabel scipy tqdm scikit-learn matplotlib
```

HD-BET (High-Definition Brain Extraction Tool) doit être installé et
configuré pour l'extraction crânienne.

## ⚙️ Ordre d'Exécution et Utilisation

Le traitement des données doit impérativement suivre cet ordre
chronologique :

### Étape 1 : Extraction des données brutes

**Fichier :** `extract_zips_fast.py`

Ce script se charge de décompresser les lourdes archives .zip
téléchargées depuis l'ADNI. Il utilise le multithreading
(ThreadPoolExecutor) pour extraire les fichiers en parallèle tout en
évitant la saturation du disque (I/O bottleneck).

``` bash
python extract_zips_fast.py
```

### Étape 2 : Création du Dataset (Mapping)

**Fichier :** `create_csv_v2.py`

Une fois les images décompressées, ce script parcourt les fichiers .nii
extraits et utilise des expressions régulières pour isoler l'identifiant
de chaque image. Il croise ensuite ces identifiants avec le fichier
clinique de l'ADNI pour générer un fichier final
dataset_preprocessed.csv.

``` bash
python create_csv_v2.py
```

Génère un CSV contenant : Nom du fichier, ID Sujet, Label (AD/CN), Âge,
etc.

### Étape 3 : Skull-Stripping (Extraction du Cerveau)

**Fichier :** `brain_extractor.py`

La dernière étape de préparation. Ce script utilise l'outil HD-BET
accéléré par GPU (cuda:0) pour nettoyer les IRM (suppression du crâne,
du cou et des tissus non cérébraux). Le script est conçu de manière
"résiliente" : il détecte les fichiers déjà traités et ne reprend que
ceux qui manquent en cas d'interruption.

``` bash
python brain_extractor.py
```

## 🏗️ Prochaines Étapes (Entraînement)

Une fois ce pipeline exécuté, les images générées dans le dossier de
sortie (ADNI_SKULL_STRIPPED) sont prêtes à être ingérées par le modèle
de Deep Learning volumétrique (ResNet 3D avec stratégie Inflate-1).

Le modèle utilise un échantillonneur intelligent (Grouped Train-Test
Split) basé sur l'identifiant du patient pour garantir l'absence totale
de fuite de données (Data Leakage).

Projet réalisé dans le cadre d'un Projet de Fin d'Année (PFA)
d'ingénierie en Intelligence Artificielle.
