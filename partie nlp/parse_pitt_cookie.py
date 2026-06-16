"""
parse_pitt_cookie.py
--------------------
Script tout-en-un pour le Pitt corpus (DementiaBank) :

  1. Scanne SEULEMENT les .cha de la tâche "cookie" (Cookie Theft picture)
  2. Extrait uniquement les tours du PARticipant (pas l'investigateur)
  3. Nettoie complètement les annotations CHAT (voir détails ci-dessous)
  4. Compte les disfluences AVANT nettoyage (features cliniques)
  5. Joint avec les métadonnées (MMSE, dx, age, sex, educ) depuis PItt-data.xlsx
  6. Sauvegarde un CSV unique : dataset_pitt_cookie.csv

Usage:
    python parse_pitt_cookie.py --pitt_root Pitt --metadata PItt-data.xlsx
"""
import argparse
import re
from pathlib import Path
import pandas as pd
import numpy as np


# =================================================================
# 1. NETTOYAGE CHAT
# =================================================================

# Compteurs de disfluences (appliqués AVANT nettoyage)
DISFLUENCY_PATTERNS = {
    'n_filled_pauses':   re.compile(r'&-\w+'),          # &-um  &-uh
    'n_phon_fragments':  re.compile(r'&\+\w+'),         # &+w  &+co  (mot abandonné)
    'n_paralinguistic':  re.compile(r'&=\w+'),          # &=laughs  &=sighs
    'n_retracings':      re.compile(r'\[/+\]'),         # [/]  [//]  (répétition / révision)
    'n_unintelligible':  re.compile(r'\bxxx\b|\byyy\b'),
    'n_pauses':          re.compile(r'\(\.+\)'),        # (.)  (..)
}

def count_disfluencies(raw_text: str) -> dict:
    return {k: len(p.findall(raw_text)) for k, p in DISFLUENCY_PATTERNS.items()}


# Regex de nettoyage CHAT (ordre important : spécifique → général)
CHAT_CLEANERS = [
    (re.compile(r'\x15\d+_\d+\x15'),         ' '),   # marqueurs temps audio
    (re.compile(r'\[.*?\]'),                 ' '),   # [/], [//], [* ...], [+ exc], etc.
    (re.compile(r'\(\.+\)'),                 ' '),   # (.) (..) (...) pauses
    (re.compile(r'&[+\-=][\w:]+'),           ' '),   # &-um  &=laughs  &+w
    (re.compile(r'&[\w:]+'),                 ' '),   # &word
    (re.compile(r'@[a-z:]+'),                ' '),   # @s:eng (code-switch)
    (re.compile(r'<[^>]*>'),                 ' '),   # <...> retracings
    (re.compile(r'\bxxx\b|\byyy\b|\bwww\b'), ' '),   # unintelligible
    (re.compile(r'[↑↓‡„‹›«»]'),              ' '),   # prosodiques
]


def clean_text(u: str) -> str:
    """
    Nettoyage CHAT complet :
      - Supprime annotations CHAT (&-um, [/], xxx, etc.)
      - Supprime marqueurs +xxx (+<, +^, +/., +//., +..., +,, +", etc.)
      - Reconstruit les mots phonétiques : fallin(g) -> falling
      - Supprime ponctuation terminale parasites
      - Normalise les espaces
    """
    if not u:
        return ''
    u = u.strip()
    # Supprimer terminator final
    u = re.sub(r'[.?!]+\s*$', '', u)
    # Marqueurs CHAT commençant par '+': +< +^ +, +" +/. +//. +/? +...
    u = re.sub(r'\+[<\^,"/\\.?!]+', ' ', u)
    # Reconstruction phonétique : fallin(g) -> falling
    u = re.sub(r'(\w+)\(([a-zA-Z]+)\)', r'\1\2', u)
    # Application des regex de nettoyage CHAT
    for rx, repl in CHAT_CLEANERS:
        u = rx.sub(repl, u)
    # '+' isolés résiduels
    u = re.sub(r'\s\+\s|\s\+$|^\+\s', ' ', u)
    # Ponctuation collée
    u = re.sub(r'\s+([,.;:!?])', r'\1', u)
    # Normalisation espaces
    u = re.sub(r'\s+', ' ', u)
    return u.strip()


# =================================================================
# 2. PARSE D'UN FICHIER .cha
# =================================================================

UTT_LINE_RE  = re.compile(r'^\*([A-Z]{3}):\s*(.*)$')   # *PAR: ...
CONT_LINE_RE = re.compile(r'^\t(.+)$')                  # continuation (TAB)
TIER_LINE_RE = re.compile(r'^[%@]\w+:')                 # %mor: %gra: @ID: etc.


def parse_cha_file(path: Path, speaker: str = 'PAR') -> dict:
    """
    Lit un .cha, extrait les tours du locuteur `speaker`, retourne :
        transcript_raw : texte brut concaténé (pour compter disfluences)
        transcript     : texte propre prêt pour NLP
        n_utterances   : nombre de tours
        disfluencies   : dict de comptages
    """
    utts_raw, utts_clean = [], []
    current_speaker = None
    current_text = []

    try:
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    except Exception:
        lines = path.read_text(encoding='latin-1', errors='replace').splitlines()

    def flush():
        nonlocal current_text, current_speaker
        if current_speaker == speaker and current_text:
            raw = ' '.join(current_text)
            cleaned = clean_text(raw)
            utts_raw.append(raw)
            if cleaned:
                utts_clean.append(cleaned)
        current_text = []

    for line in lines:
        # Tier dépendant (%mor, %gra, @xxx) : ferme l'utterance courante
        stripped = line.lstrip('\t ').rstrip()
        if TIER_LINE_RE.match(stripped):
            flush()
            current_speaker = None
            continue

        m_utt = UTT_LINE_RE.match(line)
        if m_utt:
            flush()
            current_speaker = m_utt.group(1)
            current_text = [m_utt.group(2)]
            continue

        m_cont = CONT_LINE_RE.match(line)
        if m_cont and current_speaker is not None:
            current_text.append(m_cont.group(1))
            continue

        if not line.strip():
            flush()
            current_speaker = None

    flush()

    transcript_raw = ' '.join(utts_raw)
    transcript = ' '.join(utts_clean)
    return {
        'transcript_raw': transcript_raw,
        'transcript':     transcript,
        'n_utterances':   len(utts_clean),
        'disfluencies':   count_disfluencies(transcript_raw),
    }


# =================================================================
# 3. PARSING DU NOM DE FICHIER (ex: '001-0.cha' -> id=1, visit=0)
# =================================================================

FNAME_RE = re.compile(r'^(\d{3})-(\d)\.cha$', re.IGNORECASE)

def parse_filename(fname: str):
    m = FNAME_RE.match(fname)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


# =================================================================
# 4. MÉTADONNÉES (PItt-data.xlsx)
# =================================================================

DX_LABELS = {
    1: 'ProbableAD', 2: 'PossibleAD', 3: 'Vascular',
    4: 'OtherDementia', 5: 'NoDx', 6: 'MCI', 7: 'MCI',
    8: 'Control', 9: 'Olivopontocerebellar',
}

def load_metadata(xlsx_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, sheet_name='data', header=None)
    cols = raw.iloc[2].tolist()
    md = raw.iloc[3:].copy()
    md.columns = cols
    md = md.reset_index(drop=True)

    numeric_cols = ['id', 'basedx', 'groupdx', 'mms', 'cdrfs',
                    'entryage', 'sex', 'educ',
                    'mmse2', 'mmse3', 'mmse4', 'mmse5', 'mmse6', 'mmse7',
                    'cdr2', 'cdr3', 'cdr4', 'cdr5', 'cdr6', 'cdr7']
    for c in numeric_cols:
        if c in md.columns:
            md[c] = pd.to_numeric(md[c], errors='coerce')

    md['dx_label'] = md['groupdx'].map(
        lambda x: DX_LABELS.get(int(x), 'Unknown') if pd.notna(x) else 'Unknown'
    )
    return md


def mmse_for_visit(row: pd.Series, visit: int):
    """visit=0 -> mms (baseline); visit=1 -> mmse2; visit=2 -> mmse3 ..."""
    return row.get('mms', np.nan) if visit == 0 else row.get(f'mmse{visit+1}', np.nan)

def cdr_for_visit(row: pd.Series, visit: int):
    return row.get('cdrfs', np.nan) if visit == 0 else row.get(f'cdr{visit+1}', np.nan)


# =================================================================
# 5. MAIN
# =================================================================

def main():
    ap = argparse.ArgumentParser(
        description='Parse Pitt corpus Cookie Theft files + clean text + join metadata.'
    )
    ap.add_argument('--pitt_root', required=True, type=Path,
                    help='Dossier racine contenant Control/ et Dementia/')
    ap.add_argument('--metadata', required=True, type=Path,
                    help='Chemin vers PItt-data.xlsx')
    ap.add_argument('--output', default='dataset_pitt_cookie.csv', type=Path)
    args = ap.parse_args()

    print(f'[+] Chargement des métadonnées : {args.metadata}')
    md = load_metadata(args.metadata)
    print(f'    {len(md)} participants dans l\'Excel')

    print(f'[+] Scan des .cha (tâche=cookie uniquement) sous : {args.pitt_root}')
    # On ne garde QUE les .cha situés sous un dossier "cookie"
    all_cha = list(args.pitt_root.rglob('*.cha'))
    cookie_files = [p for p in all_cha
                    if 'cookie' in [part.lower() for part in p.relative_to(args.pitt_root).parts]]
    print(f'    {len(all_cha)} .cha au total, {len(cookie_files)} dans cookie/')

    rows = []
    skipped_filename = 0
    skipped_no_metadata = 0
    skipped_empty = 0

    for cha in cookie_files:
        parts = cha.relative_to(args.pitt_root).parts
        if len(parts) < 2:
            continue
        label_folder = parts[0]   # 'Control' / 'Dementia'

        sid, visit = parse_filename(cha.name)
        if sid is None:
            skipped_filename += 1
            continue

        md_row = md[md['id'] == sid]
        if len(md_row) == 0:
            skipped_no_metadata += 1
            continue
        md_row = md_row.iloc[0]

        parsed = parse_cha_file(cha, speaker='PAR')
        transcript = parsed['transcript']
        n_tokens = len(transcript.split())

        if n_tokens == 0:
            skipped_empty += 1
            continue

        disf = parsed['disfluencies']
        rows.append({
            'subject_id':       sid,
            'visit':            visit,
            'label_folder':     label_folder,
            'task':             'cookie',
            'file_path':        str(cha),
            'transcript':       transcript,
            'n_tokens':         n_tokens,
            'n_utterances':     parsed['n_utterances'],
            # disfluences (comptées sur le texte BRUT)
            'n_filled_pauses':  disf['n_filled_pauses'],
            'n_phon_fragments': disf['n_phon_fragments'],
            'n_paralinguistic': disf['n_paralinguistic'],
            'n_retracings':     disf['n_retracings'],
            'n_unintelligible': disf['n_unintelligible'],
            'n_pauses':         disf['n_pauses'],
            # métadonnées cliniques
            'mmse_visit':       mmse_for_visit(md_row, visit),
            'cdr_visit':        cdr_for_visit(md_row, visit),
            'groupdx':          md_row.get('groupdx', np.nan),
            'dx_label':         md_row.get('dx_label', 'Unknown'),
            'entryage':         md_row.get('entryage', np.nan),
            'sex':              md_row.get('sex', np.nan),
            'educ':             md_row.get('educ', np.nan),
        })

    out = pd.DataFrame(rows).sort_values(['subject_id', 'visit']).reset_index(drop=True)
    out.to_csv(args.output, index=False, encoding='utf-8')

    # =================================================================
    # RÉSUMÉ
    # =================================================================
    print()
    print('=' * 60)
    print('RÉSUMÉ')
    print('=' * 60)
    print(f'  Fichiers parsés     : {len(out)}')
    print(f'  Sujets uniques      : {out["subject_id"].nunique()}')
    print(f'  Skipped (nom)       : {skipped_filename}')
    print(f'  Skipped (no meta)   : {skipped_no_metadata}')
    print(f'  Skipped (vide)      : {skipped_empty}')
    print()
    print('Distribution par dossier :')
    print(out['label_folder'].value_counts().to_string())
    print()
    print('Distribution par diagnostic :')
    print(out['dx_label'].value_counts().to_string())
    print()
    print('MMSE par groupe :')
    print(out.groupby('dx_label')['mmse_visit'].agg(['count', 'mean', 'std']).round(2).to_string())
    print()
    print('Tokens par groupe :')
    print(out.groupby('dx_label')['n_tokens'].agg(['mean', 'std', 'min', 'max']).round(1).to_string())
    print()
    print(f'[+] Sauvegardé -> {args.output}')


if __name__ == '__main__':
    main()
