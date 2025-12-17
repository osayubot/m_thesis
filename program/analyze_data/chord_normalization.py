"""
コードとキーの正規化関数
"""
from __future__ import annotations
import re
from typing import Optional

# Unicode ♭♯ to ascii
UNICODE_REPL = {
    "♭": "b",
    "♯": "#",
}

# Prefer flats internally: convert sharps to enharmonic flats
SHARP_TO_FLAT = {
    "A#": "Bb",
    "C#": "Db",
    "D#": "Eb",
    "F#": "Gb",
    "G#": "Ab",
}

ROOTS_12 = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
ROOT_TO_IDX = {r: i for i, r in enumerate(ROOTS_12)}
IDX = ROOT_TO_IDX  # Alias for consistency

# Parse chord root like: Bb, C#, F, etc.
ROOT_RE = re.compile(r"^([A-G])([b#])?")

def norm_unicode(s: str) -> str:
    for k, v in UNICODE_REPL.items():
        s = s.replace(k, v)
    return s

def sharp_to_flat(root: str) -> str:
    return SHARP_TO_FLAT.get(root, root)

def normalize_root(root: str) -> Optional[str]:
    """Return root in flats among ROOTS_12, else None."""
    m = ROOT_RE.match(root)
    if not m:
        return None
    letter = m.group(1)
    acc = m.group(2) or ""
    r = f"{letter}{acc}"
    r = sharp_to_flat(r)
    # normalize weird cases if ever appear (e.g., E#) -> ignore
    return r if r in ROOT_TO_IDX else None

def normalize_chord(ch: str) -> Optional[str]:
    """
    Task-specific chord normalization for key detection (Bag-of-Chords):
    - Drop slash bass: D/F# -> D
    - Normalize unicode accidentals; prefer flats
    - Keep only qualities: major, minor, dominant7, dim, sus
    - Drop add/maj7/m7/9/11/13/6 etc. (collapse to base)
    """
    if not ch:
        return None
    ch = norm_unicode(ch).strip()
    if ch in {"N.C.", "NC", "N/C", "-", "?"}:
        return None

    # drop slash bass
    ch = ch.split("/")[0].strip()

    # extract root
    m = ROOT_RE.match(ch)
    if not m:
        return None
    root_raw = (m.group(1) + (m.group(2) or ""))
    root = sharp_to_flat(root_raw)
    if root not in ROOT_TO_IDX:
        return None

    rest = ch[len(root_raw):]  # keep original acc length
    rest = rest.strip()

    # unify minor spellings
    # treat "min", "-" as minor
    is_minor = False
    if rest.startswith(("m", "min", "-")):
        # BUT exclude "maj" (e.g., maj7)
        if rest.startswith("maj"):
            is_minor = False
        else:
            is_minor = True

    # diminished
    if "dim" in rest or "o" == rest[:1]:
        return f"{root}dim"

    # sus
    if "sus" in rest:
        return f"{root}sus"

    # dominant 7: keep only plain "7" (not maj7)
    # examples: G7, A7(omit tensions), B7
    # If "maj7" or "M7" -> treat as major triad
    if re.search(r"(maj7|M7)", rest):
        return root

    if re.search(r"(^|[^a-zA-Z])7([^0-9]|$)", rest) or rest.endswith("7"):
        # m7 should collapse to minor, not dominant7
        if is_minor:
            return f"{root}m"
        return f"{root}7"

    # Everything else: collapse to major/minor triad
    return f"{root}m" if is_minor else root

def normalize_key_label(k: str) -> Optional[str]:
    """Normalize key labels to 24-class strings like 'C', 'Bbm', etc."""
    if not k:
        return None
    k = norm_unicode(k).strip()
    # Some datasets use "Dm" etc. We'll keep 'm' suffix for minor.
    # Drop anything after whitespace
    k = k.split()[0]

    # Handle forms like "C#m", "A#", "Bb", "CM" (rare)
    # Extract root then minor flag
    m = ROOT_RE.match(k)
    if not m:
        return None
    root_raw = (m.group(1) + (m.group(2) or ""))
    root = sharp_to_flat(root_raw)
    if root not in ROOT_TO_IDX:
        return None

    rest = k[len(root_raw):]
    rest = rest.strip()

    is_minor = False
    if rest.startswith(("m", "min", "-")) and not rest.startswith("maj"):
        is_minor = True

    return f"{root}m" if is_minor else root

def split_key_label(k: str):
    """'Em' -> ('E', True), 'C' -> ('C', False)"""
    if not k:
        return None, None
    k = normalize_key_label(k)
    if not k:
        return None, None
    if k.endswith("m"):
        return k[:-1], True
    return k, False

def parse_norm_token(tok: str):
    """
    tok is output of normalize_chord(): root + optional suffix among {m,7,dim,sus}
    returns: (root:str, quality:str)
      quality in {"maj","min","dom7","dim","sus","min7"}
    """
    if tok.endswith("dim"):
        return tok[:-3], "dim"
    if tok.endswith("sus"):
        return tok[:-3], "sus"
    if tok.endswith("7"):
        root = tok[:-1]
        # might be "Am7" style from normalize_chord? (it returns root+"m" for m7, not "m7")
        # But earlier normalize_chord returns "rootm" for minor triad/m7 collapse, and "root7" for dominant.
        return root, "dom7"
    if tok.endswith("m"):
        return tok[:-1], "min"
    return tok, "maj"

