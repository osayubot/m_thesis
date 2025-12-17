"""
ローマ数字変換関数
"""
from __future__ import annotations
from typing import Optional
from .chord_normalization import (
    ROOT_TO_IDX, normalize_chord, split_key_label, parse_norm_token
)

MAJOR_DEGREE = {
    0: ("I", 0),
    2: ("II", 0),
    4: ("III", 0),
    5: ("IV", 0),
    7: ("V", 0),
    9: ("VI", 0),
    11: ("VII", 0),
}

MINOR_DEGREE = {
    0: ("I", 0),
    2: ("II", 0),
    3: ("III", 0),
    5: ("IV", 0),
    7: ("V", 0),
    8: ("VI", 0),
    10: ("VII", 0),
}

# chromatic fallback (use b/# against nearest diatonic degree)
MAJOR_CHROM = {
    1: ("b", "II"),   # bII
    3: ("b", "III"),  # bIII
    6: ("#", "IV"),   # #IV (tritone-ish)
    8: ("b", "VI"),   # bVI
    10: ("b", "VII"), # bVII
}

MINOR_CHROM = {
    1: ("b", "II"),
    4: ("#", "III"),  # #III (rare)
    6: ("#", "IV"),
    9: ("#", "VI"),
    11: ("#", "VII"),
}

def degree_in_key(root: str, key_root: str, is_minor_key: bool):
    """
    returns a roman base like 'IV' or 'bVII' (no case/quality yet)
    """
    if root not in ROOT_TO_IDX or key_root not in ROOT_TO_IDX:
        return None
    semi = (ROOT_TO_IDX[root] - ROOT_TO_IDX[key_root]) % 12

    if not is_minor_key:
        if semi in MAJOR_DEGREE:
            deg, _ = MAJOR_DEGREE[semi]
            return deg
        if semi in MAJOR_CHROM:
            acc, deg = MAJOR_CHROM[semi]
            return f"{acc}{deg}"
        # last resort: label as chromatic degree by nearest (rare)
        return f"chrom{semi}"
    else:
        if semi in MINOR_DEGREE:
            deg, _ = MINOR_DEGREE[semi]
            return deg
        if semi in MINOR_CHROM:
            acc, deg = MINOR_CHROM[semi]
            return f"{acc}{deg}"
        return f"chrom{semi}"

def chord_to_roman(chord_raw: str, key_label: str):
    """
    chord_raw: original chord string (e.g., 'G', 'D/F#', 'Em7')
    key_label: section key like 'G' or 'Em'
    returns: roman token like 'V', 'vi', 'V7', 'bVII', 'ivsus'
    """
    key_root, is_minor_key = split_key_label(key_label)
    if not key_root:
        return None

    tok = normalize_chord(chord_raw)
    if not tok:
        return None
    root, qual = parse_norm_token(tok)

    deg = degree_in_key(root, key_root, is_minor_key)
    if not deg:
        return None

    # apply case by chord quality
    # major-like: uppercase, minor: lowercase, dim: lowercase + o
    if qual == "min":
        base = deg.lower()
    elif qual == "dim":
        base = deg.lower() + "o"  # dim marker
    else:
        base = deg  # uppercase

    # keep some quality tags (helps islands separate: V vs V7 vs IVsus)
    if qual == "dom7":
        base = base + "7"
    elif qual == "sus":
        base = base + "sus"

    return base

def section_to_roman_progression(section: dict, fallback_key: str = None):
    """
    section dict contains chord_progression and optionally key.
    returns list of roman tokens
    """
    k = section.get("key") or fallback_key
    if not k:
        return []
    out = []
    for ch in section.get("chord_progression", []):
        r = chord_to_roman(ch, k)
        if r:
            out.append(r)
    return out

