# Alzheimer's Disease Detection — MRI and Speech

Bimodal detection of Alzheimer's disease: a 3D convolutional network on
brain MRI volumes, and a transformer on spontaneous speech transcripts.

End-of-year engineering project (PFA), AI engineering degree.

---

## Two modalities, two datasets

| Branch | Data | Signal |
|--------|------|--------|
| Imaging | ADNI — 3D T1 MRI volumes | structural atrophy |
| Language | DementiaBank Pitt Corpus — Cookie Theft transcripts | lexical and disfluency markers |

Each branch is independent. They classify the same condition from
unrelated evidence.

---

## Imaging branch

### Preprocessing

Raw ADNI archives are turned into a clean, labelled, skull-stripped
dataset in three steps, to be run in order.

**1. `extract_zips_fast.py`** — Parallel extraction of the ADNI archives
with a thread pool, throttled to avoid saturating disk I/O.

**2. `create_csv_v2.py`** — Walks the extracted `.nii` files, recovers
each image identifier by regex, and joins it with ADNI clinical records
to produce `dataset_preprocessed.csv` (filename, subject ID, AD/CN
label, age, sex).

**3. `brain_extractor.py`** — GPU skull-stripping with HD-BET, removing
skull, neck and non-brain tissue. Resumable: already-processed files are
detected and skipped, so an interrupted run continues where it stopped.

### Model

**`model.py`** — ResNet3D-18 using the *Inflate-1* strategy: a 2D
ResNet-18 pretrained on ImageNet, with its kernels inflated to 3D so the
pretrained features transfer to volumetric data. Training includes tight
cropping to the brain, focal loss for class imbalance, volumetric
augmentation, and test-time augmentation.

**`st_2_multimodal.py`** — Late-fusion variant combining the 3D backbone
with an MLP over clinical features (age, sex), and an automatic
comparison against the MRI-only model. Age and sex are normalised on the
training split only.

**`gradcam_3axes.py`** — 3D Grad-CAM rendered across axial, sagittal and
coronal planes, several slices each, to show which regions drive a
prediction.

### Avoiding data leakage

Patients contribute several scans across visits. Splitting on images
would place the same patient on both sides and inflate accuracy. The
split is grouped by subject identifier, so no patient appears in both
training and test sets. The same rule applies to the speech branch,
split by subject and stratified by label.

---

## Language branch — `partie nlp/`

**`parse_pitt_cookie.py`** builds the dataset from the Pitt Corpus:

- Keeps only Cookie Theft recordings, and only the participant's turns
  (the investigator's are discarded).
- Strips CHAT annotation markup to recover plain transcripts.
- Counts disfluencies *before* cleaning — filled pauses, phonological
  fragments, retracings, unintelligible segments, paralinguistic events
  — since those counts are clinical features in their own right.
- Joins with Pitt metadata: MMSE, CDR, diagnosis, age, sex, education.

Output: `dataset_pitt_cookie.csv`, 552 transcripts with 21 columns.

`Pitt_NLP_Conversation_Journal.md` is the working log for this branch —
dataset discovery, labelling conventions, split strategy, and the
RoBERTa-versus-BERT decision.

---

## Stack

| Layer | Technology |
|-------|------------|
| Deep learning | PyTorch, torchvision |
| Medical imaging | nibabel, HD-BET, scipy.ndimage |
| NLP | transformers, pandas |
| Evaluation | scikit-learn — F1, ROC-AUC, confusion matrices |
| Explainability | Grad-CAM 3D |

Trained on a single RTX 4070 Mobile (8 GB VRAM), which constrained batch
size and input resolution throughout.

## Requirements

```bash
pip install torch torchvision pandas nibabel scipy tqdm scikit-learn matplotlib
```

HD-BET must be installed separately for skull-stripping, and requires a
CUDA-capable GPU.

## Running the imaging pipeline

```bash
python extract_zips_fast.py     # 1. extract ADNI archives
python create_csv_v2.py         # 2. build the labelled dataset
python brain_extractor.py       # 3. skull-strip
python model.py                 # 4. train
python gradcam_3axes.py         # 5. visualise attributions
```

---

**Ilias EL AMRANI** · [LinkedIn](https://linkedin.com/in/iliaselamrani)
