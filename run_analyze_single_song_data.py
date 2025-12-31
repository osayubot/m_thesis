#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Re-analyze ONE song (by spotify_id) and write to data/analyzed/<spotify_id>.json.

This regenerates `analyzed_chord_progressions_and_lyrics` including:
- key / key_confidence
- emotion (koshin2001/Japanese-to-emotions)

Usage:
  python run_analyze_single_song.py <spotify_id>

Options:
  --input_dir data/combined --output_dir data/analyzed
  --W 16 --H 4 --switch_penalty 4.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from program.analyze_data.data_extraction import load_dataset
from program.analyze_data.file_utils import save_song_with_keys
from program.analyze_data.key_assignment import assign_keys_to_ufret_sections


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spotify_id", help="Target spotify_id (e.g., 5XURwUMd9vo0YoFJXQ0feh)")
    ap.add_argument("--input_dir", default="data/combined", help="Directory containing combined JSONs")
    ap.add_argument("--output_dir", default="data/analyzed", help="Directory to write analyzed JSON")
    ap.add_argument("--W", type=int, default=16)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--switch_penalty", type=float, default=4.0)
    args = ap.parse_args()

    root_dir = Path(__file__).parent
    input_dir = (root_dir / args.input_dir).resolve()
    output_dir = (root_dir / args.output_dir).resolve()
    song_path = input_dir / f"{args.spotify_id}.json"

    if not song_path.exists():
        raise SystemExit(f"Not found: {song_path}")

    # Train the same model used in analyze_data/main.py (X_all + X_root).
    texts_all, _texts_last, X_root, y = load_dataset(str(input_dir), recursive=True)
    vec_all = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    X_all = vec_all.fit_transform(texts_all)
    X = hstack([X_all, csr_matrix(X_root)]).tocsr()

    clf = LogisticRegression(
        solver="saga",
        max_iter=2000,
        n_jobs=-1,
        C=4.0,
    )
    clf.fit(X, y)

    song = json.loads(song_path.read_text(encoding="utf-8"))
    analyzed_sections = assign_keys_to_ufret_sections(
        song.copy(),
        vec_all,
        clf,
        W=args.W,
        H=args.H,
        switch_penalty=args.switch_penalty,
    )
    if not analyzed_sections:
        raise SystemExit("No analyzed sections were produced (too few chords / missing data?)")

    song_out = song.copy()
    song_out["analyzed_chord_progressions_and_lyrics"] = analyzed_sections
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_song_with_keys(song_out, str(output_dir), create_subdirs=False)
    print(f"Wrote: {out_file}")


if __name__ == "__main__":
    main()


