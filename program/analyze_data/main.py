"""
データ分析のメイン処理
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .chord_normalization import normalize_key_label
from .transposition import transpose_key
from .data_extraction import (
    load_dataset, extract_ufret_chords, extract_ufret_chords_with_section_spans
)
from .evaluation import evaluate_with_cv
from .modulation import (
    modulation_analysis_for_song, build_chord_index_map
)
from .key_assignment import assign_keys_to_ufret_sections
from .file_utils import (
    save_song_with_keys, process_and_save_songs_with_keys, print_modulation_log
)

def main():
    # point this to your folder with many *.json files
    # Use data/combined directory which contains both jtotal and ufret data
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    # Go up to root directory: program/analyze_data -> program -> root
    root_dir = script_dir.parent.parent
    JSON_DIR = str(root_dir / "data" / "combined")

    texts_all, texts_last, X_root, y = load_dataset(JSON_DIR, recursive=True)
    print(f"Loaded {len(y)} songs")

    # Run cross-validation (n_splits=3 to handle classes with only 2 members)
    evaluate_with_cv(texts_all, texts_last, X_root, y, n_splits=3, random_state=42)
    
    # Train a model for modulation analysis (using all + root only, matching make_window_features)
    print("\n" + "="*60)
    print("Training model for modulation analysis...")
    vec_all_full = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    
    X_all_full = vec_all_full.fit_transform(texts_all)
    X_root_sp_full = csr_matrix(X_root)
    
    # For modulation analysis, we use X_all + X_root (matching make_window_features)
    X_mod_full = hstack([X_all_full, X_root_sp_full]).tocsr()
    
    clf_full = LogisticRegression(
        solver="saga",
        max_iter=2000,
        n_jobs=-1,
        C=4.0,
    )
    clf_full.fit(X_mod_full, y)
    print("Model trained for modulation analysis (X_all + X_root)")
    
    # Example: Analyze modulation for songs with modulations
    print("\n" + "="*60)
    print("Example: Modulation analysis - finding songs with key changes")
    print("="*60)
    
    paths = sorted(Path(JSON_DIR).rglob("*.json"))
    found_count = 0
    max_examples = 3
    
    if paths:
        # Try to find songs with modulations
        for test_path in paths:
            data = json.loads(test_path.read_text(encoding="utf-8"))
            songs = data if isinstance(data, list) else [data]
            
            for song in songs:
                if not isinstance(song, dict):
                    continue
                chords = extract_ufret_chords(song)
                if len(chords) >= 40:  # Enough chords for modulation analysis
                    res = modulation_analysis_for_song(
                        song, vec_all_full, clf_full,
                        W=16, H=4,
                        switch_penalty=4.0,
                        min_run_windows=3,
                        use_ufret=True
                    )
                    if "error" not in res and len(res.get("modulations", [])) > 0:
                        print_modulation_log(song, res, context=6, use_ufret=True)
                        found_count += 1
                        if found_count >= max_examples:
                            break
            if found_count >= max_examples:
                break
    
    if found_count == 0:
        print("転調が検出された曲が見つかりませんでした。")
        print("（転調が少ない曲が多かったか、パラメータ調整が必要かもしれません）")
    
    # Example: Assign keys to sections
    print("\n" + "="*60)
    print("Example: Assigning keys to sections")
    print("="*60)
    
    if paths:
        # Test on first song with enough chords
        for test_path in paths[:3]:
            data = json.loads(test_path.read_text(encoding="utf-8"))
            songs = data if isinstance(data, list) else [data]
            
            for song in songs:
                if not isinstance(song, dict):
                    continue
                # Check if ufret data exists with enough chords
                chords_norm, _, _ = extract_ufret_chords_with_section_spans(song)
                if len(chords_norm) >= 40:
                    # Assign keys to ufret sections using normalized index spans
                    analyzed_sections = assign_keys_to_ufret_sections(
                        song.copy(), vec_all_full, clf_full,
                        W=16, H=4, switch_penalty=4.0
                    )
                    
                    if analyzed_sections:
                        # Add analyzed_chord_progressions_and_lyrics to song
                        song_copy = song.copy()
                        song_copy["analyzed_chord_progressions_and_lyrics"] = analyzed_sections
                        
                        # Show first few sections with keys
                        print(f"\n{song_copy.get('title', 'Unknown')} / {song_copy.get('artist', 'Unknown')}")
                        # Calculate key from jtotal_original_key + ufret_capo
                        jtotal_original_key = song_copy.get("jtotal_original_key", "N/A")
                        ufret_capo = song_copy.get("ufret_capo", 0)
                        if jtotal_original_key != "N/A":
                            base_key = normalize_key_label(jtotal_original_key)
                            if base_key:
                                calculated_key = transpose_key(base_key, ufret_capo)
                                print(f"Label key: {calculated_key} (from jtotal_original_key={jtotal_original_key} + ufret_capo={ufret_capo})")
                            else:
                                print(f"Label key: N/A")
                        else:
                            print(f"Label key: N/A")
                        for i, sec in enumerate(analyzed_sections[:5]):
                            key_info = f"key={sec.get('key', 'N/A')}"
                            conf_info = f"conf={sec.get('key_confidence', 0):.2f}"
                            chord_preview = " ".join(sec.get("chord_progression", [])[:4])
                            print(f"  Section {i}: {key_info} ({conf_info}) | {chord_preview}...")
                        
                        # Save example
                        OUTPUT_DIR = str(root_dir / "data" / "analyzed")
                        saved_file = save_song_with_keys(song_copy, OUTPUT_DIR)
                        print(f"\nSaved to: {saved_file}")
                        break
            else:
                continue
            break
    
    # Process songs and save to data/analyzed
    print("\n" + "="*60)
    print("Processing songs and saving to data/analyzed...")
    print("="*60)
    OUTPUT_DIR = str(root_dir / "data" / "analyzed")
    stats = process_and_save_songs_with_keys(
        JSON_DIR, OUTPUT_DIR,
        vec_all_full, clf_full,
        W=16, H=4, switch_penalty=4.0,
        min_chords=12, max_songs=None, use_jtotal_new=False, use_ufret_transposed=False, use_ufret=True
    )
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Processed: {stats['processed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

