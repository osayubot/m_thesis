"""
移調関連関数
"""
from __future__ import annotations
from typing import List, Optional
from collections import Counter
from .chord_normalization import (
    normalize_root, normalize_key_label, split_key_label, 
    ROOT_TO_IDX, ROOTS_12
)

def first_valid_root(chords: List[str]):
    """Get first valid root from chord list."""
    for c in chords:
        r = normalize_root(c)
        if r is not None:
            return r
    return None

def semitone_diff(r1: str, r2: str) -> Optional[int]:
    """Calculate semitone difference from r1 to r2 (returns -11 to +11, or None)."""
    if r1 not in ROOT_TO_IDX or r2 not in ROOT_TO_IDX:
        return None
    diff = (ROOT_TO_IDX[r2] - ROOT_TO_IDX[r1]) % 12
    # Convert to signed range (-11 to +11)
    if diff > 6:
        diff -= 12
    return diff

def transpose_key(key: str, semitone: int) -> Optional[str]:
    """Transpose key by semitone amount."""
    if not key or semitone is None:
        return None
    root, is_minor = split_key_label(key)
    if not root:
        return None
    if root not in ROOT_TO_IDX:
        return None
    new_root_idx = (ROOT_TO_IDX[root] + semitone) % 12
    new_root = ROOTS_12[new_root_idx]
    return f"{new_root}m" if is_minor else new_root

def lyric_initial_match(jt_sec: dict, uf_sec: dict) -> bool:
    """
    Check if the first character of lyrics matches between jtotal and ufret sections.
    Used as a quality score for data integration.
    
    Args:
        jt_sec: Section dict from jtotal
        uf_sec: Section dict from ufret
    
    Returns:
        True if first characters match (and both are non-empty), False otherwise
    """
    jt = jt_sec.get("lyric", "").strip()
    uf = uf_sec.get("lyric", "").strip()
    if not jt or not uf:
        return False
    return jt[0] == uf[0]

def estimate_transposition_shift(jt_chords: List[str], uf_chords: List[str], N: int = 12) -> Optional[int]:
    """
    Estimate transposition shift between jtotal and ufret chords.
    Uses most common semitone difference from first N chords.
    
    Args:
        jt_chords: Normalized chords from jtotal
        uf_chords: Normalized chords from ufret
        N: Number of initial chords to compare
    
    Returns:
        Estimated semitone shift (None if cannot estimate)
    """
    diffs = []
    min_len = min(len(jt_chords), len(uf_chords), N)
    
    for i in range(min_len):
        r1 = normalize_root(jt_chords[i])
        r2 = normalize_root(uf_chords[i])
        if r1 and r2:
            diff = semitone_diff(r1, r2)
            if diff is not None:
                diffs.append(diff)
    
    if not diffs:
        return None
    
    # Use most common difference
    return Counter(diffs).most_common(1)[0][0]

