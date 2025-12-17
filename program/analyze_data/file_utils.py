"""
ファイル操作関数
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from .chord_normalization import normalize_key_label, normalize_chord, normalize_root, split_key_label, ROOT_TO_IDX, ROOTS_12
from .transposition import transpose_key, semitone_diff
from .key_assignment import (
    assign_keys_to_ufret_with_transposition, assign_keys_with_probabilities,
    assign_keys_to_jtotal_sections, assign_keys_to_ufret_sections
)
from .data_extraction import (
    extract_jtotal_chords, extract_jtotal_chords_with_section_spans,
    extract_ufret_chords, extract_ufret_chords_with_section_spans
)

def remove_key_info_from_sections(secs):
    """Remove key-related fields from sections."""
    key_fields = ["key", "key_confidence", "key_method", "key_span"]
    for sec in secs:
        for field in key_fields:
            sec.pop(field, None)

def save_song_with_keys(song_json, output_dir: str, create_subdirs: bool = True):
    """
    Save song JSON with analyzed_chord_progressions_and_lyrics to output directory.
    Removes jtotal_chord_progressions_and_lyrics, ufret_chord_progressions_and_lyrics,
    jtotal_original_key, jtotal_original_play_key, ufret_original_key, ufret_capo,
    and key_span/key_method from sections.
    Adds ufret_play_key calculated from jtotal_original_key + ufret_capo.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_json = song_json.copy()
    
    # Calculate ufret_play_key from jtotal_original_key + ufret_capo
    jtotal_original_key = song_json.get("jtotal_original_key")
    ufret_capo = song_json.get("ufret_capo")
    if ufret_capo is None:
        ufret_capo = 0
    
    if jtotal_original_key:
        base_key = normalize_key_label(jtotal_original_key)
        if base_key:
            calculated_key = transpose_key(base_key, ufret_capo)
            if calculated_key:
                output_json["ufret_play_key"] = calculated_key
    
    # Remove jtotal and ufret chord progression fields
    output_json.pop("jtotal_chord_progressions_and_lyrics", None)
    output_json.pop("ufret_chord_progressions_and_lyrics", None)
    
    # Remove key fields from top level
    output_json.pop("jtotal_original_key", None)
    output_json.pop("jtotal_original_play_key", None)
    output_json.pop("ufret_original_key", None)
    output_json.pop("ufret_capo", None)
    
    # Remove key_span and key_method from analyzed sections, and add normalized_chord_progression
    if "analyzed_chord_progressions_and_lyrics" in output_json:
        for sec in output_json["analyzed_chord_progressions_and_lyrics"]:
            sec.pop("key_span", None)
            sec.pop("key_method", None)
            
            # Add normalized_chord_progression (transposed to C or Am)
            chord_progression = sec.get("chord_progression", [])
            key = sec.get("key")
            
            if chord_progression and key:
                # Determine target key: C for major, Am for minor
                root, is_minor = split_key_label(key)
                if root:
                    target_root = "Am" if is_minor else "C"
                    target_root_only = "A" if is_minor else "C"
                    
                    # Calculate semitone difference from current key root to target
                    if root in ROOT_TO_IDX and target_root_only in ROOT_TO_IDX:
                        shift = semitone_diff(root, target_root_only)
                        if shift is not None:
                            normalized_chords = []
                            for chord in chord_progression:
                                # Normalize chord first
                                normalized = normalize_chord(chord)
                                if normalized:
                                    # Extract root and quality
                                    chord_root = normalize_root(normalized)
                                    if chord_root and chord_root in ROOT_TO_IDX:
                                        # Transpose root
                                        new_root_idx = (ROOT_TO_IDX[chord_root] + shift) % 12
                                        new_root = ROOTS_12[new_root_idx]
                                        
                                        # Keep quality (m, 7, dim, sus, etc.)
                                        quality = normalized[len(chord_root):]
                                        transposed_chord = new_root + quality
                                        normalized_chords.append(transposed_chord)
                                    else:
                                        normalized_chords.append(normalized)
                                else:
                                    # If normalization fails, try to transpose original
                                    chord_root = normalize_root(chord)
                                    if chord_root and chord_root in ROOT_TO_IDX:
                                        new_root_idx = (ROOT_TO_IDX[chord_root] + shift) % 12
                                        new_root = ROOTS_12[new_root_idx]
                                        quality = chord[len(chord_root):]
                                        transposed_chord = new_root + quality
                                        normalized_chords.append(transposed_chord)
                                    else:
                                        normalized_chords.append(chord)
                            
                            if normalized_chords:
                                sec["normalized_chord_progression"] = normalized_chords
                            else:
                                # Fallback: just normalize without transposition
                                normalized_chords = []
                                for chord in chord_progression:
                                    normalized = normalize_chord(chord)
                                    normalized_chords.append(normalized if normalized else chord)
                                sec["normalized_chord_progression"] = normalized_chords
                        else:
                            # Fallback: just normalize without transposition
                            normalized_chords = []
                            for chord in chord_progression:
                                normalized = normalize_chord(chord)
                                normalized_chords.append(normalized if normalized else chord)
                            sec["normalized_chord_progression"] = normalized_chords
                    else:
                        # Fallback: just normalize without transposition
                        normalized_chords = []
                        for chord in chord_progression:
                            normalized = normalize_chord(chord)
                            normalized_chords.append(normalized if normalized else chord)
                        sec["normalized_chord_progression"] = normalized_chords
                else:
                    # No valid key, just normalize
                    normalized_chords = []
                    for chord in chord_progression:
                        normalized = normalize_chord(chord)
                        normalized_chords.append(normalized if normalized else chord)
                    sec["normalized_chord_progression"] = normalized_chords
            elif chord_progression:
                # No key, just normalize
                normalized_chords = []
                for chord in chord_progression:
                    normalized = normalize_chord(chord)
                    normalized_chords.append(normalized if normalized else chord)
                sec["normalized_chord_progression"] = normalized_chords
    
    # Determine filename from spotify_id (priority)
    spotify_id = output_json.get("spotify_id")
    if spotify_id:
        filename = f"{spotify_id}.json"
    else:
        jtotal_path = output_json.get("jtotal_path")
        if jtotal_path:
            filename = f"{jtotal_path.replace('/', '-')}.json"
        else:
            title = output_json.get("title", "unknown")
            artist = output_json.get("artist", "unknown")
            safe_title = re.sub(r'[^\w\-_\.]', '_', title)[:50]
            safe_artist = re.sub(r'[^\w\-_\.]', '_', artist)[:50]
            filename = f"{safe_artist}_{safe_title}.json"
    
    output_file = output_path / filename
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)
    
    return output_file

def process_and_save_songs_with_keys(input_dir: str, output_dir: str, vec_all, clf, 
                                     W=16, H=4, switch_penalty=4.0, 
                                     min_chords=12, max_songs=None, use_jtotal_new=False, use_ufret_transposed=False, use_ufret=True):
    """
    Process all songs in input directory, assign keys to sections, and save to output directory.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    stats = {"processed": 0, "skipped": 0, "errors": 0}
    paths = sorted(input_path.rglob("*.json"))
    
    if max_songs:
        paths = paths[:max_songs]
    
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            songs = data if isinstance(data, list) else [data]
            
            for song in songs:
                if not isinstance(song, dict):
                    continue
                
                if use_ufret_transposed:
                    jt_chords = extract_jtotal_chords(song)
                    uf_chords = extract_ufret_chords(song)
                    if len(jt_chords) < min_chords or len(uf_chords) < min_chords:
                        stats["skipped"] += 1
                        continue
                    analyzed_sections = assign_keys_to_ufret_with_transposition(
                        song.copy(), vec_all, clf,
                        W=W, H=H, switch_penalty=switch_penalty
                    )
                elif use_ufret:
                    chords_norm, _, _ = extract_ufret_chords_with_section_spans(song)
                    if len(chords_norm) < min_chords:
                        stats["skipped"] += 1
                        continue
                    analyzed_sections = assign_keys_to_ufret_sections(
                        song.copy(), vec_all, clf,
                        W=W, H=H, switch_penalty=switch_penalty
                    )
                elif use_jtotal_new:
                    chords_norm, _, _ = extract_jtotal_chords_with_section_spans(song)
                    if len(chords_norm) < min_chords:
                        stats["skipped"] += 1
                        continue
                    analyzed_sections = assign_keys_to_jtotal_sections(
                        song.copy(), vec_all, clf,
                        W=W, H=H, switch_penalty=switch_penalty
                    )
                else:
                    chords = extract_jtotal_chords(song)
                    if len(chords) < min_chords:
                        stats["skipped"] += 1
                        continue
                    analyzed_sections = assign_keys_with_probabilities(
                        song.copy(), vec_all, clf,
                        W=W, H=H, switch_penalty=switch_penalty,
                        use_ufret=False
                    )
                
                if not analyzed_sections:
                    stats["skipped"] += 1
                    continue
                
                song_copy = song.copy()
                song_copy["analyzed_chord_progressions_and_lyrics"] = analyzed_sections
                
                save_song_with_keys(song_copy, output_dir)
                stats["processed"] += 1
                
        except Exception as e:
            print(f"Error processing {p}: {e}")
            stats["errors"] += 1
    
    return stats

def print_modulation_log(song_json, analysis_result, context=6, use_ufret=True):
    """
    Print human-readable modulation detection log.
    """
    title = song_json.get("title", "(unknown)")
    artist = song_json.get("artist", "(unknown)")
    jtotal_original_key = song_json.get("jtotal_original_key", "(unknown)")
    ufret_capo = song_json.get("ufret_capo", 0)
    if jtotal_original_key != "(unknown)":
        base_key = normalize_key_label(jtotal_original_key)
        if base_key:
            calculated_key = transpose_key(base_key, ufret_capo)
            play_key = calculated_key
        else:
            play_key = "(unknown)"
    else:
        play_key = song_json.get("jtotal_original_play_key", "(unknown)")
    print(f"\n=== {title} / {artist}  (label key={play_key}, from jtotal_original_key={jtotal_original_key} + ufret_capo={ufret_capo}) ===")
    
    from .modulation import build_chord_index_map
    mapping = build_chord_index_map(song_json, use_ufret=use_ufret)
    flat_len = len(mapping)
    
    mods = analysis_result.get("modulations", [])
    if not mods:
        print("転調検出: なし（キー列はほぼ一定）")
        return
    
    for m in mods:
        idx = m["at_chord_index"]
        idx = max(0, min(idx, flat_len - 1))
        si, ci, raw_chord, lyric = mapping[idx]
        
        s = max(0, idx - context)
        e = min(flat_len, idx + context + 1)
        ctx = " ".join([mapping[i][2] for i in range(s, e)])
        
        print("\n" + "-" * 40)
        print(f"ここから転調!!  {m['from_key']}  →  {m['to_key']}")
        print(f"位置: flat_chord_index={idx} / section={si} / chord_in_section={ci} / chord={raw_chord}")
        if lyric:
            print(f"歌詞(セクション): {lyric[:40]}{'...' if len(lyric) > 40 else ''}")
        print(f"周辺コード({context}前後): {ctx}")

