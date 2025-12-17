"""
データ抽出関数
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
from .chord_normalization import normalize_chord, normalize_root, normalize_key_label, ROOT_TO_IDX
from .transposition import transpose_key

def extract_jtotal_chords(song: dict) -> List[str]:
    """Extract and normalize chords from jtotal_chord_progressions_and_lyrics."""
    chords: List[str] = []
    for sec in song.get("jtotal_chord_progressions_and_lyrics", []):
        for c in sec.get("chord_progression", []):
            nc = normalize_chord(c)
            if nc:
                chords.append(nc)
    return chords

def extract_jtotal_chords_with_section_spans(song: dict):
    """
    Extract normalized chords with section spans based on normalized indices.
    
    Returns:
      chords_norm: List[str] (flattened normalized chords)
      section_spans: List[tuple[int,int]] (start,end) indices in chords_norm for each section
      section_norm_counts: List[int] number of kept chords per section
    """
    chords_norm = []
    section_spans = []
    section_norm_counts = []
    cur = 0
    
    for sec in song.get("jtotal_chord_progressions_and_lyrics", []):
        start = cur
        kept = 0
        for c in sec.get("chord_progression", []):
            nc = normalize_chord(c)
            if nc:
                chords_norm.append(nc)
                kept += 1
                cur += 1
        end = cur
        section_spans.append((start, end))
        section_norm_counts.append(kept)
    
    return chords_norm, section_spans, section_norm_counts

def extract_ufret_chords(song: dict) -> List[str]:
    """Extract and normalize chords from ufret_chord_progressions_and_lyrics."""
    chords: List[str] = []
    for sec in song.get("ufret_chord_progressions_and_lyrics", []):
        for c in sec.get("chord_progression", []):
            nc = normalize_chord(c)
            if nc:
                chords.append(nc)
    return chords

def extract_ufret_chords_with_section_spans(song: dict):
    """
    Extract normalized chords with section spans based on normalized indices for ufret.
    
    Returns:
      chords_norm: List[str] (flattened normalized chords)
      section_spans: List[tuple[int,int]] (start,end) indices in chords_norm for each section
      section_norm_counts: List[int] number of kept chords per section
    """
    chords_norm = []
    section_spans = []
    section_norm_counts = []
    cur = 0
    
    for sec in song.get("ufret_chord_progressions_and_lyrics", []):
        start = cur
        kept = 0
        for c in sec.get("chord_progression", []):
            nc = normalize_chord(c)
            if nc:
                chords_norm.append(nc)
                kept += 1
                cur += 1
        end = cur
        section_spans.append((start, end))
        section_norm_counts.append(kept)
    
    return chords_norm, section_spans, section_norm_counts

def root_hist_12(norm_chords: List[str]) -> np.ndarray:
    """Calculate root histogram (12 semitones)."""
    v = np.zeros(12, dtype=np.float32)
    for c in norm_chords:
        r = normalize_root(c)
        if r is not None:
            v[ROOT_TO_IDX[r]] += 1.0
    s = v.sum()
    if s > 0:
        v /= s
    return v

def load_dataset(json_dir: str, recursive: bool = False) -> Tuple[List[str], List[str], np.ndarray, List[str]]:
    """
    Load dataset from JSON files.
    
    Returns:
      texts_all: "C Am F G ..." full sequence
      texts_last: last N tokens as text
      X_root: (n,12) root histogram
      y: key labels
    """
    json_path = Path(json_dir)
    if not json_path.exists():
        raise RuntimeError(f"JSON directory does not exist: {json_path.absolute()}")
    
    print(f"Loading dataset from: {json_path.absolute()}")
    
    if recursive:
        paths = sorted(json_path.rglob("*.json"))
    else:
        paths = sorted(json_path.glob("*.json"))
    
    print(f"Found {len(paths)} JSON files")
    
    texts_all: List[str] = []
    texts_last: List[str] = []
    roots: List[np.ndarray] = []
    y: List[str] = []
    
    # Debug counters
    total_songs = 0
    skipped_no_jtotal_key = 0
    skipped_no_base_key = 0
    skipped_no_calculated_key = 0
    skipped_no_final_key = 0
    skipped_too_few_chords = 0

    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: Failed to parse {p}: {e}")
            continue
        
        # Handle both list and dict formats
        songs = data if isinstance(data, list) else [data]
        
        for song in songs:
            if not isinstance(song, dict):
                continue
            
            total_songs += 1
            
            # Calculate key from jtotal_original_key + ufret_capo
            jtotal_original_key = song.get("jtotal_original_key")
            ufret_capo = song.get("ufret_capo")
            
            # Handle None case for ufret_capo (default to 0)
            if ufret_capo is None:
                ufret_capo = 0
            
            if not jtotal_original_key:
                skipped_no_jtotal_key += 1
                continue
            
            # Normalize jtotal_original_key first
            base_key = normalize_key_label(jtotal_original_key)
            if not base_key:
                skipped_no_base_key += 1
                continue
            
            # Transpose by ufret_capo (capo offset in semitones)
            calculated_key = transpose_key(base_key, ufret_capo)
            if not calculated_key:
                skipped_no_calculated_key += 1
                continue
            
            key = normalize_key_label(calculated_key)
            if not key:
                skipped_no_final_key += 1
                continue

            # Use ufret_chord_progressions_and_lyrics instead of jtotal
            chords = extract_ufret_chords(song)
            # basic filter: require enough info
            if len(chords) < 12:
                skipped_too_few_chords += 1
                continue

            texts_all.append(" ".join(chords))
            lastN = chords[-16:] if len(chords) >= 16 else chords
            texts_last.append(" ".join(lastN))
            roots.append(root_hist_12(chords))
            y.append(key)

    print(f"\nDataset loading statistics:")
    print(f"  Total songs processed: {total_songs}")
    print(f"  Skipped (no jtotal_original_key): {skipped_no_jtotal_key}")
    print(f"  Skipped (base_key normalization failed): {skipped_no_base_key}")
    print(f"  Skipped (calculated_key transpose failed): {skipped_no_calculated_key}")
    print(f"  Skipped (final key normalization failed): {skipped_no_final_key}")
    print(f"  Skipped (too few chords < 12): {skipped_too_few_chords}")
    print(f"  Successfully loaded: {len(texts_all)}")

    if not texts_all:
        raise RuntimeError(f"No samples loaded. Check json_dir and field names.\n"
                          f"Total songs: {total_songs}, Skipped: {skipped_no_jtotal_key + skipped_no_base_key + skipped_no_calculated_key + skipped_no_final_key + skipped_too_few_chords}")
    X_root = np.vstack(roots)
    return texts_all, texts_last, X_root, y

