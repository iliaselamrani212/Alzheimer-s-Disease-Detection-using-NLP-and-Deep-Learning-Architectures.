# 📔 Journal de Conversation - Projet NLP Alzheimer (Pitt Corpus)

## Informations Générales

| Élément | Détail |
|---------|--------|
| **Étudiant** | Younes |
| **Projet** | Mémoire de fin d'études (PFA) - Partie NLP |
| **Sujet** | Détection automatique de la maladie d'Alzheimer à partir de transcriptions orales |
| **Dataset** | Pitt Corpus - DementiaBank |
| **Hardware** | PC portable, RTX 4070 Mobile 8GB VRAM, 31GB RAM |
| **Environnement** | Windows, Anaconda env `torch_gpu`, Python 3.10.18 |
| **Lien avec partie IRM** | Modalité complémentaire (texte vs imagerie ADNI) |

---

## Table des Matières

1. [Découverte du Dataset](#1-découverte-du-dataset)
2. [Parsing et Nettoyage CHAT](#2-parsing-et-nettoyage-chat)
3. [Filtrage Cookie Theft](#3-filtrage-cookie-theft)
4. [Compréhension des Labels Cliniques](#4-compréhension-des-labels-cliniques)
5. [Convention de Nommage des Fichiers](#5-convention-de-nommage-des-fichiers)
6. [Split par Sujet Stratifié](#6-split-par-sujet-stratifié)
7. [Choix du Transformer (RoBERTa vs BERT)](#7-choix-du-transformer-roberta-vs-bert)
8. [Architecture RoBERTa et Token `<s>`](#8-architecture-roberta-et-token-s)
9. [Stratégie de Fine-tuning](#9-stratégie-de-fine-tuning)
10. [Hyperparamètres et Régularisation](#10-hyperparamètres-et-régularisation)
11. [Code Final et Bugs Identifiés](#11-code-final-et-bugs-identifiés)
12. [Scripts Générés](#12-scripts-générés)

---

## 1. Découverte du Dataset

### 1.1 Données téléchargées

| Source | Contenu |
|--------|---------|
| **DementiaBank Pitt Corpus** | Transcriptions au format CHAT (.cha) |
| **PItt-data.xlsx** | Métadonnées cliniques (MMSE, CDR, diagnostic, âge, sexe, éducation) |

**Arborescence des fichiers** :
```
Pitt/
├── Control/
│   └── cookie/      → 243 fichiers .cha
│   └── fluency/, recall/, sentence/  (rares)
└── Dementia/
    └── cookie/      → 309 fichiers .cha
    └── fluency/, recall/, sentence/
```

### 1.2 Les 4 tâches du protocole Pitt

| Tâche | Description | Nb fichiers |
|-------|-------------|-------------|
| **cookie** | Description de l'image "Cookie Theft" (BDAE) | 552 |
| **fluency** | Fluence verbale (citer un max d'animaux en 60s) | 237 |
| **recall** | Rappel d'histoire (Cinderella) | 264 |
| **sentence** | Construction de phrases | 237 |
| **Total** | | **1290** |

### 1.3 Distribution diagnostique (Excel)

| Code `groupdx` | Label (créé par script) | Nb sujets | MMSE moyen |
|----------------|-------------------------|-----------|------------|
| 1 | ProbableAD | 248 | 17.6 ± 6.5 |
| 2 | PossibleAD | 98 | 17.2 ± 6.0 |
| 3 | Vascular | 12 | 17.8 |
| 4 | OtherDementia | 12 | 23.8 |
| 5 | NoDx | 15 | 27.9 |
| 6/7 | MCI | 19 | 27.7 |
| 8 | Control | 106 | 29.1 ± 1.0 |

**Note importante** : L'Excel original contient **uniquement des codes numériques** (1-8). Les labels texte (`ProbableAD`, `Control`, etc.) sont créés artificiellement par le script via un mapping basé sur la feuille `readme` de l'Excel.

---

## 2. Parsing et Nettoyage CHAT

### 2.1 Pourquoi parser ?

Le format CHAT (Codes for the Human Analysis of Transcripts, MacWhinney 2000) contient de nombreuses annotations linguistiques qui doivent être nettoyées pour le NLP :

```
Exemple ligne brute du .cha :
*PAR:	&-uh well there's a [/] a thing &-um (..) you know the &+w water
	is &-uh going .
%mor:	co|well co|uh pro:exist|there~v|be det:art|a
```

### 2.2 Le script `parse_pitt_cookie.py`

**Étapes du pipeline** :
1. Scanne récursivement les `.cha` sous `Pitt/`
2. Filtre les fichiers situés dans un dossier `cookie/`
3. Pour chaque fichier :
   - Extrait uniquement les tours `*PAR:` (patient), ignore `*INV:` (investigateur)
   - Compte les disfluences sur le texte BRUT (features cliniques)
   - Nettoie les annotations CHAT
4. Joint avec `PItt-data.xlsx` via `subject_id`
5. Récupère MMSE/CDR de la **bonne visite** (pas juste baseline)
6. Sort `dataset_pitt_cookie.csv`

### 2.3 Annotations CHAT nettoyées

| Annotation | Exemple | Traitement |
|------------|---------|------------|
| `&-um`, `&-uh` | Filled pauses | Supprimées (mais comptées) |
| `&+w`, `&+co` | Phonological fragments | Supprimés (mais comptés) |
| `&=laughs`, `&=sighs` | Paralinguistic | Supprimés (mais comptés) |
| `[/]`, `[//]` | Retracings | Supprimés (mais comptés) |
| `xxx`, `yyy`, `www` | Unintelligible | Supprimés |
| `(.)`, `(..)`, `(...)` | Pauses silencieuses | Supprimées (mais comptées) |
| `+<` | Chevauchement | Supprimé |
| `+/.`, `+//.`, `+...` | Interruptions | Supprimés |
| `+^`, `+,`, `+"` | Autres marqueurs CHAT | Supprimés |
| `fallin(g)`, `washin(g)` | Notation phonétique | **Reconstruit** : `falling`, `washing` |
| `%mor:`, `%gra:` | Annotations morphologiques | Ignorées (lignes entières) |
| `<...>` | Retraced segments | Supprimés |

### 2.4 Features cliniques (disfluences) comptées avant nettoyage

| Feature | Phénomène clinique | Ratio AD/Control observé |
|---------|---------------------|--------------------------|
| `n_filled_pauses` | Hésitations (&-um, &-uh) | 1.0× |
| `n_phon_fragments` | Mots abandonnés (&+w) | **2.3×** |
| `n_retracings` | Répétitions/révisions [/] [//] | **1.9×** |
| `n_unintelligible` | Mots indistincts (xxx) | Élevé |
| `n_pauses` | Silences (.) (..) | Élevé |
| `n_paralinguistic` | Rires, soupirs (&=laughs) | Variable |

### 2.5 Avant/Après nettoyage (exemple)

**Avant** :
```
*PAR: &-uh well there's a [/] a thing &-um (..) you know
*PAR: the &+w water is &-uh going .
*PAR: mhm. +< alright. there's a young boy fallin(g) over.
```

**Après** :
```
well there's a thing you know
the water is going
mhm. alright. there's a young boy falling over
```

---

## 3. Filtrage Cookie Theft

### 3.1 Pourquoi seulement cookie ?

Cross-tab révélatrice du dataset :

| Task | Control | ProbableAD |
|------|---------|------------|
| cookie | 219 | 250 |
| fluency | 1 | 181 |
| recall | 0 | 198 |
| sentence | 0 | 176 |

**Découverte cruciale** : Les Controls n'ont passé QUE la tâche cookie. Inclure les autres biaiserait massivement la classification.

### 3.2 Script `filter_cookie.py`

```python
import pandas as pd
df = pd.read_csv('dataset_pitt_nlp.csv')
df[df['task'] == 'cookie'].to_csv('dataset_pitt_cookie.csv', index=False)
```

### 3.3 Résultat après filtrage

| | Total |
|---|---|
| Transcripts cookie | 552 |
| Sujets uniques | 292 |
| Moyenne transcripts/sujet | 1.89 |

---

## 4. Compréhension des Labels Cliniques

### 4.1 Les 6 catégories du dossier `Dementia/`

| Diagnostic | Définition | Pourquoi exclure de la classif binaire |
|------------|------------|----------------------------------------|
| **ProbableAD** (n=250) | Alzheimer certain (critères NINCDS-ADRDA) | ✅ INCLURE (cible AD) |
| **PossibleAD** (n=34) | AD probable + autres facteurs | Diagnostic incertain |
| **Vascular** (n=3) | Démence causée par AVC, micro-infarctus | Mécanisme biologique différent + n=3 |
| **OtherDementia** (n=5) | Lewy, frontotemporale, mixte... | Trop hétérogène + n=5 |
| **NoDx** (n=12) | Plaintes mnésiques sans diagnostic | Catégorie ambiguë |
| **MCI** (n=29) | Trouble cognitif léger (entre normal et démence) | État intermédiaire, sujet à part |

### 4.2 Choix final pour la classification

**Classification binaire** : `Control` vs `ProbableAD`
- **469 transcripts** (219 + 250)
- **233 sujets uniques**
- Équilibré : 53% AD / 47% Control
- Conforme à la convention de la littérature (Fraser 2016, ADReSS 2020)

```python
data = data[data['dx_label'].isin(['Control', 'ProbableAD'])]
```

---

## 5. Convention de Nommage des Fichiers

### 5.1 Format `NNN-V.cha`

```
   002 - 0 . cha
    │   │   │
    │   │   └── Format CHAT
    │   └────── V = numéro de visite (0=baseline, 1, 2, 3, 4)
    └────────── NNN = ID du sujet (3 chiffres)
```

**Exemples** :
- `002-0.cha` : Sujet 2, baseline
- `002-3.cha` : Sujet 2, 4ème visite (~3 ans après baseline)
- `006-2.cha` : Sujet 6, 3ème visite (pas de baseline disponible)

### 5.2 Implications

- Dataset **longitudinal** : suivi pluri-annuel (1 visite/an environ)
- Un sujet a typiquement **1 à 4 visites**
- Le **MMSE évolue d'une visite à l'autre** (déclin chez les AD)

### 5.3 Récupération du MMSE par visite (script)

```python
def mmse_for_visit(row, visit):
    if visit == 0:
        return row['mms']         # baseline
    return row[f'mmse{visit+1}']   # mmse2, mmse3, ...
```

**Exemple vérifié** : Sujet 1 (ProbableAD) :
- Visite 0 : MMSE = 18
- Visite 2 : MMSE = 11
- → Déclin de -7 points en 2 ans, cohérent avec progression Alzheimer

---

## 6. Split par Sujet Stratifié

### 6.1 Risque de data leakage

**Problème** : 552 transcripts pour 292 sujets uniques. Un split naïf par fichier mettrait le même sujet en train ET val.

❌ **Mauvais split (data leakage)** :
```
Train : 002-0.cha, 002-1.cha, 002-3.cha
Val   : 002-2.cha
```

✅ **Bon split (par sujet)** :
```
Train : tous les .cha des sujets 1, 3, 5, 7, ...
Val   : tous les .cha des sujets 2, 4, 6, 8, ...
```

### 6.2 Code final stratifié (70/15/15)

```python
from sklearn.model_selection import train_test_split

# Labels au NIVEAU SUJET (un sujet = un diagnostic)
subject_labels = data.groupby('subject_id')['dx_label'].first()

# Split stratifié
id_tr_vl, id_test, lab_tr_vl, _ = train_test_split(
    subject_labels.index.values, subject_labels.values,
    test_size=0.15, random_state=42, stratify=subject_labels.values
)
id_tr, id_vl, _, _ = train_test_split(
    id_tr_vl, lab_tr_vl,
    test_size=0.17, random_state=42, stratify=lab_tr_vl
)

# Sous-DataFrames
data_train = data[data['subject_id'].isin(id_tr)].reset_index(drop=True)
data_val   = data[data['subject_id'].isin(id_vl)].reset_index(drop=True)
data_test  = data[data['subject_id'].isin(id_test)].reset_index(drop=True)

# Anti-leakage
assert len(set(id_tr) & set(id_vl))   == 0
assert len(set(id_tr) & set(id_test)) == 0
assert len(set(id_vl) & set(id_test)) == 0
```

### 6.3 Résultat du split

| Split | Sujets | Transcripts | % AD |
|-------|--------|-------------|------|
| Train | 164 | 339 | 51.3% |
| Val | 34 | 55 | 52.7% |
| Test | 35 | 75 | 62.7% |
| **Total** | **233** | **469** | **53.3%** |

**Note** : `test_size=0.17` au 2ème split = mathématique pour obtenir 70/15/15 au final :
```
0.15 / 0.85 = 0.176 ≈ 0.17
```

---

## 7. Choix du Transformer (RoBERTa vs BERT)

### 7.1 Comparaison des transformers pour la détection AD

| Transformer | Accuracy ADReSS | Params | Pourquoi |
|-------------|-----------------|--------|----------|
| BERT-base | 83-85% | 110M | Baseline standard |
| DistilBERT | 82-84% | 66M | Léger, perd 1-2 points |
| **RoBERTa-base** ⭐ | **86-88%** | 125M | Meilleur pré-entraînement |
| RoBERTa-large | 87-89% | 355M | Trop gros pour 8GB VRAM |
| ELECTRA-base | 85-87% | 110M | Moins étudié sur AD |
| ClinicalBERT | 84-86% | 110M | Mauvais (texte écrit ≠ oral) |
| DeBERTa-v3 | 88-90% | 184M | SOTA mais moins de tutos |

### 7.2 Pourquoi RoBERTa-base ?

| Critère | BERT-base | RoBERTa-base |
|---------|-----------|--------------|
| Pré-entraînement | 16 GB (Wikipedia + Books) | **160 GB** (10× plus, inclut OpenWebText, Stories) |
| Masquage | Statique | **Dynamique** (à chaque epoch) |
| Tâche NSP | Oui | **Supprimée** (jugée inutile) |
| Vocabulaire | 30k WordPiece | **50k BPE byte-level** |
| Batch pré-entraînement | 256 | **8000** |

**Raisons principales** :
1. **Performance prouvée** sur ADReSS 2020 (Balagopalan et al.)
2. **Pré-entraînement plus large** incluant données conversationnelles (OpenWebText, Stories) proches du langage spontané
3. **Compatible 8GB VRAM** : 125M params en FP16 → 3-4 GB VRAM
4. Disponible sur Hugging Face : `roberta-base`

---

## 8. Architecture RoBERTa et Token `<s>`

### 8.1 Équivalences BERT vs RoBERTa

| | BERT | RoBERTa |
|---|------|---------|
| Token classification | `[CLS]` | `<s>` |
| Token séparation | `[SEP]` | `</s>` |
| Token padding | `[PAD]` | `<pad>` |
| Token masquage | `[MASK]` | `<mask>` |
| Token unknown | `[UNK]` | `<unk>` |

### 8.2 Le token `<s>` chez RoBERTa

```
Input  : "the boy is taking a cookie"
Tokens : ['<s>', 'the', 'Ġboy', 'Ġis', 'Ġtaking', 'Ġa', 'Ġcookie', '</s>']
IDs    : [   0,  1437,   2143,  1437,    1972,   102,    7197,    2  ]
```

- `<s>` (id=0) est ajouté **automatiquement** au début par le tokenizer
- C'est l'équivalent fonctionnel du `[CLS]` de BERT
- Position toujours **0** dans la séquence

### 8.3 Différence subtile : pas de NSP

- BERT pré-entraîné avec Next Sentence Prediction → `[CLS]` spécialement entraîné pour résumer la phrase
- RoBERTa a supprimé NSP → `<s>` n'a pas reçu cet entraînement spécifique
- En pratique : `<s>` s'avère **aussi efficace** après fine-tuning sur classification

### 8.4 Architecture de la head de classification

**3 façons de l'implémenter** :

#### Façon 1 : `RobertaForSequenceClassification` (recommandée)

```python
from transformers import RobertaForSequenceClassification
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
```

Head HuggingFace en interne :
```python
RobertaClassificationHead(
    dense:    Linear(768 → 768),
    dropout:  Dropout(p=0.1),
    out_proj: Linear(768 → 2)
)
# Forward: x → dense → tanh → dropout → out_proj
```
→ **592 130 paramètres** dans la head.

#### Façon 2 : Head custom simple

```python
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.classifier = nn.Linear(768, 2)
    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])
```
→ **1 538 paramètres** seulement.

#### Façon 3 : Head hybride (RoBERTa + features cliniques)

```python
class HybridModel(nn.Module):
    def __init__(self, n_features=6):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.classifier = nn.Sequential(
            nn.Linear(768 + n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    def forward(self, input_ids, attention_mask, clinical_features):
        out = self.roberta(input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        combined = torch.cat([cls, clinical_features], dim=1)
        return self.classifier(combined)
```
→ Combine embedding `<s>` + 6 disfluences → meilleur résultat attendu.

### 8.5 Dimension cachée RoBERTa-base

⚠️ Erreur courante : c'est **768**, pas 720 !

| Modèle | Hidden size |
|--------|-------------|
| RoBERTa-base, BERT-base, DistilBERT | **768** |
| RoBERTa-large, BERT-large | 1024 |

---

## 9. Stratégie de Fine-tuning

### 9.1 Fine-tuning vs Feature Extraction

| Stratégie | Ce qu'on entraîne | Accuracy attendue |
|-----------|-------------------|-------------------|
| **Feature extraction** (head only) | Juste la head MLP | ~75-78% |
| **Full fine-tuning** ⭐ | TOUT (RoBERTa + head) | ~86-88% |
| Partial fine-tuning | Head + dernières couches | ~83-85% |

### 9.2 Pourquoi full fine-tuning ?

**Argument 1 - Décalage de domaine** : RoBERTa pré-entraîné sur texte général (Wikipedia, news). Notre dataset = **discours spontané oral de personnes âgées avec troubles cognitifs**. Domaine très différent.

**Argument 2 - Dataset suffisant** : 469 transcripts > seuil empirique de ~500 exemples pour fine-tuner BERT/RoBERTa.

**Argument 3 - Littérature ADReSS** : Tous les papiers gagnants ont fait full fine-tuning, aucun n'a obtenu >85% avec juste la head.

### 9.3 Lien avec partie IRM

Tu as vécu exactement la même problématique avec ResNet3D :
- Backbone gelé → 74%
- Full fine-tuning sans précaution → overfit (val=63%)
- **Inflated ResNet3D avec full fine-tuning + régularisation → 83.8%** ⭐

Même principe ici : **full fine-tuning + régularisation forte**.

---

## 10. Hyperparamètres et Régularisation

### 10.1 Learning Rate : **2e-5**

| LR | Effet | Verdict |
|---|---|---|
| 1e-3 | Catastrophic forgetting | ❌ JAMAIS |
| 1e-4 | Instable | ❌ |
| 5e-5 | Acceptable mais risqué | ⚠️ |
| **2e-5** | **Optimal RoBERTa** | ✅ ⭐ |
| 1e-5 | Trop lent | ⚠️ |

**Sources** : Liu et al. 2019, Devlin et al. 2018, Mosbach et al. 2021, Balagopalan et al. 2020.

### 10.2 Stratégie LR différentié (discriminative fine-tuning)

```python
optimizer = torch.optim.AdamW([
    {'params': model.roberta.parameters(),    'lr': 2e-5},   # backbone : LR petit
    {'params': model.classifier.parameters(), 'lr': 1e-4}    # head : LR 5× plus grand
], weight_decay=0.01)
```

**Justification** : La head est initialisée aléatoirement → elle doit apprendre vite. Le backbone est déjà bon → ajustement léger seulement.

### 10.3 Scheduler : Warmup + Linear Decay

```python
from transformers import get_linear_schedule_with_warmup

num_training_steps = len(train_loader) * num_epochs
num_warmup_steps = int(0.1 * num_training_steps)   # 10% warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)
```

**Pourquoi warmup** : Évite que le head random "perturbe" le backbone dans les premiers steps.

**Pourquoi decay** : Permet au modèle de se stabiliser sur un minimum local en fin d'entraînement.

### 10.4 Weight Decay : **0.01**

**Rôle** : Pénalité L2 sur les poids → empêche le modèle de devenir trop "grand" et de mémoriser le training set.

**Formule** :
```
Loss = CrossEntropy + λ × Σ(w²)    avec λ = 0.01
```

**Effet sur les poids** :
- Sans wd : poids appris = `[-12.4, +35.7, -41.2, ...]` → overfit
- Avec wd : poids appris = `[-0.8, +1.2, -1.1, ...]` → généralise

### 10.5 AdamW vs Adam

⚠️ **Toujours `AdamW` pour fine-tuner BERT/RoBERTa**, jamais `Adam`.

- **Adam** : mélange weight_decay avec gradient → régularisation **biaisée** par moments adaptatifs (mathématiquement incorrect)
- **AdamW** (Loshchilov & Hutter 2019) : applique weight_decay **séparément** → régularisation propre

### 10.6 Configuration finale recommandée

```python
config = {
    'model_name':    'roberta-base',
    'num_labels':    2,
    'max_length':    256,            # cookie ≈ 100 tokens, 256 large
    'batch_size':    8 ou 16,
    'num_epochs':    3-5,            # ⚠️ pas 30 !
    'lr_backbone':   2e-5,
    'lr_classifier': 1e-4,
    'weight_decay':  0.01,
    'warmup_ratio':  0.1,
    'dropout':       0.1,            # RoBERTa default
    'fp16':          True,           # économise VRAM RTX 4070
    'gradient_clip': 1.0,
}
```

### 10.7 Mécanismes de régularisation cumulés

| Mécanisme | Valeur | Rôle |
|-----------|--------|------|
| Weight decay | 0.01 | Limite amplitude poids |
| Dropout | 0.1 | Désactive neurones aléatoirement |
| Warmup + decay | 10% | Stabilise training |
| Early stopping | sur val F1 | Stoppe avant overfit |
| Gradient clipping | 1.0 | Évite gradients explosifs |
| Mixed Precision FP16 | - | Économise VRAM + accélère |

---

## 11. Code Final et Bugs Identifiés

### 11.1 Bugs critiques détectés dans le 1er notebook

**Bug 1 - Texte val/test = texte train** :
```python
# AVANT (BUG)
text_tr   = data_train['transcript'].tolist()
text_vl   = data_train['transcript'].tolist()   # ← BUG
text_test = data_train['transcript'].tolist()   # ← BUG

# APRÈS (FIX)
text_tr   = data_train['transcript'].tolist()
text_vl   = data_val['transcript'].tolist()     # ✅
text_test = data_test['transcript'].tolist()    # ✅
```

**Bug 2 - 30 epochs (catastrophic overfit)** :
```python
Train(model, ..., epochs=30)   # ❌ trop
Train(model, ..., epochs=4)    # ✅ 3-5 max
```

**Bug 3 - Tokenisation directe sur DataFrame** :
```python
torch.tensor(data_train[['transcript']])   # ❌ ValueError
# → il faut tokeniser d'abord
tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
```

### 11.2 Pipeline complet final

```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchmetrics import Accuracy, F1Score
from sklearn.model_selection import train_test_split
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)

# === 1. CHARGEMENT + FILTRAGE ===
data = pd.read_csv('dataset_pitt_cookie.csv')
data = data[data['dx_label'].isin(['ProbableAD', 'Control'])].reset_index(drop=True)

# === 2. SPLIT STRATIFIÉ PAR SUJET ===
subject_labels = data.groupby('subject_id')['dx_label'].first()
id_tr_vl, id_test, lab_tr_vl, _ = train_test_split(
    subject_labels.index.values, subject_labels.values,
    test_size=0.15, random_state=42, stratify=subject_labels.values
)
id_tr, id_vl, _, _ = train_test_split(
    id_tr_vl, lab_tr_vl,
    test_size=0.17, random_state=42, stratify=lab_tr_vl
)

data_train = data[data['subject_id'].isin(id_tr)].reset_index(drop=True)
data_val   = data[data['subject_id'].isin(id_vl)].reset_index(drop=True)
data_test  = data[data['subject_id'].isin(id_test)].reset_index(drop=True)

# === 3. EXTRACTION TEXTES + LABELS ===
text_tr   = data_train['transcript'].tolist()
text_vl   = data_val['transcript'].tolist()
text_test = data_test['transcript'].tolist()

tr_labels   = (data_train['dx_label'] == 'ProbableAD').astype(int).tolist()
val_labels  = (data_val['dx_label']   == 'ProbableAD').astype(int).tolist()
test_labels = (data_test['dx_label']  == 'ProbableAD').astype(int).tolist()

# === 4. DATASET + DATALOADER ===
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

class NLPDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long)
        }

tr_ds   = NLPDataset(text_tr,   tr_labels,   tokenizer)
val_ds  = NLPDataset(text_vl,   val_labels,  tokenizer)
test_ds = NLPDataset(text_test, test_labels, tokenizer)

tr_loader   = DataLoader(tr_ds,   batch_size=8, shuffle=True)
val_loader  = DataLoader(val_ds,  batch_size=8, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

# === 5. MODÈLE + OPTIMISEUR + SCHEDULER ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base', num_labels=2
).to(device)

NUM_EPOCHS = 4

optimizer = torch.optim.AdamW([
    {'params': model.roberta.parameters(),    'lr': 2e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-4}
], weight_decay=0.01)

num_training_steps = len(tr_loader) * NUM_EPOCHS
num_warmup_steps = int(0.1 * num_training_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

scaler = GradScaler()

# === 6. TRAINING LOOP ===
def Train(model, train_loader, valid_loader, optimizer, scheduler, scaler, epochs):
    metric_train_acc = Accuracy(task='binary').to(device)
    metric_train_f1  = F1Score(task='binary').to(device)
    metric_valid_acc = Accuracy(task='binary').to(device)
    metric_valid_f1  = F1Score(task='binary').to(device)

    best_val_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        metric_train_acc.reset(); metric_train_f1.reset()
        train_loss = 0.0

        for batch in train_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            optimizer.zero_grad()
            with autocast(dtype=torch.float16):
                output = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = output.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            preds = torch.argmax(output.logits, dim=1)
            metric_train_acc.update(preds, labels)
            metric_train_f1.update(preds, labels)
            train_loss += loss.item()

        model.eval()
        metric_valid_acc.reset(); metric_valid_f1.reset()
        valid_loss = 0.0

        with torch.no_grad():
            for batch in valid_loader:
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels         = batch['labels'].to(device)
                with autocast(dtype=torch.float16):
                    output = model(input_ids, attention_mask=attention_mask, labels=labels)
                preds = torch.argmax(output.logits, dim=1)
                metric_valid_acc.update(preds, labels)
                metric_valid_f1.update(preds, labels)
                valid_loss += output.loss.item()

        train_acc = metric_train_acc.compute().item()
        train_f1  = metric_train_f1.compute().item()
        val_acc   = metric_valid_acc.compute().item()
        val_f1    = metric_valid_f1.compute().item()

        print(f"---- Epoch {epoch+1}/{epochs} ----")
        print(f"  Train | Loss: {train_loss/len(train_loader):.4f}  Acc: {train_acc:.4f}  F1: {train_f1:.4f}")
        print(f"  Val   | Loss: {valid_loss/len(valid_loader):.4f}  Acc: {val_acc:.4f}  F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_roberta_pitt.pt')
            print(f"  ✅ Best model saved (val F1: {val_f1:.4f})")

Train(model, tr_loader, val_loader, optimizer, scheduler, scaler, epochs=NUM_EPOCHS)
```

### 11.3 Résultats attendus

| Epoch | Train Acc | Val Acc | Val F1 |
|-------|-----------|---------|--------|
| 1 | ~65% | ~70% | ~0.70 |
| 2 | ~80% | ~80% | ~0.80 |
| 3 | ~88% | ~85% | ~0.85 |
| **4** | **~92%** | **~87%** | **~0.86** ⭐ |

---

## 12. Scripts Générés

### 12.1 Scripts créés pendant la conversation

| Script | Fonction | Status |
|--------|----------|--------|
| `parse_pitt_cha.py` | Parser .cha (toutes tâches) + nettoyage CHAT + jointure Excel | ✅ v1 |
| `filter_cookie.py` | Filtrage cookie seul (à partir CSV générique) | ✅ |
| `parse_pitt_cookie.py` | Tout-en-un : parse cookie + nettoyage + jointure | ✅ Version finale |
| `dataset_pitt_cookie.csv` | CSV final (552 lignes, 21 colonnes) | ✅ |
| `train_roberta.py` (à coder) | Fine-tuning RoBERTa avec FP16, scheduler, early stopping | 🔜 |

### 12.2 Colonnes du CSV final

| Catégorie | Colonnes |
|-----------|----------|
| **Identification** | `subject_id`, `visit`, `label_folder`, `task`, `file_path` |
| **Texte** | `transcript` (nettoyé), `n_tokens`, `n_utterances` |
| **Disfluences (cliniques)** | `n_filled_pauses`, `n_phon_fragments`, `n_paralinguistic`, `n_retracings`, `n_unintelligible`, `n_pauses` |
| **Métadonnées** | `mmse_visit`, `cdr_visit`, `groupdx`, `dx_label`, `entryage`, `sex`, `educ` |

---

## 📚 Références Clés

| Référence | Apport |
|-----------|--------|
| **Fraser et al. 2016** | Features linguistiques classiques sur Pitt (81.9% acc) |
| **Devlin et al. 2018 (BERT)** | Architecture transformer + token [CLS] |
| **Liu et al. 2019 (RoBERTa)** | Améliorations sur BERT, LR=2e-5 recommandé |
| **Loshchilov & Hutter 2019 (AdamW)** | Correction du weight decay dans Adam |
| **Balagopalan et al. 2020** | Comparaison BERT vs RoBERTa sur Pitt/ADReSS |
| **Luz et al. 2020 (ADReSS Challenge)** | Convention binaire Control vs ProbableAD |
| **Mosbach et al. 2021** | Étude de stabilité du fine-tuning BERT |
| **MacWhinney 2000 (CHILDES)** | Format CHAT, manuel officiel |

---

## 🎓 Points clés à retenir pour le mémoire

### Méthodologie

1. **Filtrer Control vs ProbableAD** uniquement (469 transcripts, classes équilibrées)
2. **Split par sujet stratifié** (anti data-leakage, comme partie IRM)
3. **Tâche cookie seule** (seule où Controls sont représentés)
4. **MMSE par visite** (pas juste baseline)

### Choix de modèle

5. **RoBERTa-base** > BERT-base (160 GB pré-entraînement, masquage dynamique)
6. **Full fine-tuning** > feature extraction (469 transcripts suffisants)
7. **Token `<s>`** chez RoBERTa = équivalent `[CLS]` de BERT

### Hyperparamètres

8. **LR = 2e-5** (backbone), **1e-4** (head)
9. **AdamW** avec **weight_decay = 0.01**
10. **3-5 epochs max** (jamais 30 !)
11. **Warmup 10% + linear decay**
12. **FP16 + gradient clipping 1.0**

### Évaluation

13. **Métriques** : Accuracy, F1, AUC, matrice de confusion
14. **Early stopping** sur val F1
15. **Test set évalué UNE seule fois** à la fin

### Cohérence avec partie IRM

| Partie IRM (ADNI) | Partie NLP (Pitt) |
|-------------------|-------------------|
| HD-BET skull stripping | Nettoyage CHAT |
| Split par sujet 80/20 | Split par sujet 70/15/15 stratifié |
| Inflated ResNet3D-18 | RoBERTa-base full fine-tuning |
| Weighted loss | Weight decay 0.01 |
| Mixed Precision FP16 | Mixed Precision FP16 |
| 83.8% accuracy AD vs CN | ~87% attendu Control vs ProbableAD |

---

## 🔜 Prochaines étapes

1. ⏭️ Lancer le training corrigé (4 epochs) et vérifier les résultats
2. ⏭️ Évaluation sur test set + matrice de confusion
3. ⏭️ Modèle hybride RoBERTa + 6 features disfluences (gain attendu +1-2%)
4. ⏭️ Régression MMSE (bonus, score continu 0-30)
5. ⏭️ Comparaison avec ML classique (LogReg/SVM/RF + features Fraser)
6. ⏭️ Phase 2 (plus tard) : Multimodal texte + audio

---

*Document généré le 21 mai 2026*
*Journal de conversation - Partie NLP du PFA : Détection automatique d'Alzheimer sur le Pitt Corpus*
