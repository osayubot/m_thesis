"""
データ読み込みと前処理
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter, defaultdict
import logging

from .config import Config, DataProcessingConfig
from .utils import setup_logging

logger = setup_logging()


def load_json_files(
    input_dir: str | Path,
    max_files: Optional[int] = None,
    use_multiprocessing: bool = True
) -> List[Dict[str, Any]]:
    """
    JSONファイルを読み込む
    
    Args:
        input_dir: 入力ディレクトリ
        max_files: 最大ファイル数（Noneなら無制限）
        use_multiprocessing: マルチプロセッシングを使用するか
    
    Returns:
        楽曲データのリスト
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    json_files = list(input_dir.glob("*.json"))
    if max_files:
        json_files = json_files[:max_files]
    
    logger.info(f"Loading {len(json_files)} JSON files from {input_dir}")
    
    songs = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 1ファイルに複数曲が含まれる場合と1曲の場合の両方に対応
            if isinstance(data, list):
                songs.extend(data)
            elif isinstance(data, dict):
                # 単一の曲データ
                songs.append(data)
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")
            continue
    
    logger.info(f"Loaded {len(songs)} songs")
    return songs


def extract_phrases(
    songs: List[Dict[str, Any]],
    song_id_key: str = "ufret_id"
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    楽曲データからフレーズを抽出
    
    Args:
        songs: 楽曲データのリスト
        song_id_key: 曲IDのキー名（ufret_id, jtotal_path等）
    
    Returns:
        (phrases, song_metadata)
        phrases: フレーズデータのリスト（各フレーズにsong_id, phrase_id, text, metadataを含む）
        song_metadata: {song_id: {title, artist, ...}}
    """
    phrases = []
    song_metadata = {}
    
    for song_idx, song in enumerate(songs):
        # 曲IDを取得（複数のキーを試す）
        song_id = None
        for key in [song_id_key, "ufret_id", "jtotal_path", "spotify_id", "id"]:
            if key in song and song[key]:
                song_id = str(song[key])
                break
        
        if not song_id:
            song_id = f"song_{song_idx}"
        
        # 曲メタデータを保存
        song_metadata[song_id] = {
            'title': song.get('title', ''),
            'artist': song.get('artist', ''),
            'lyricist': song.get('lyricist', ''),
            'composer': song.get('composer', ''),
            'spotify_id': song.get('spotify_id', ''),
            'release_date': song.get('release_date', ''),
        }
        
        # フレーズを抽出
        analyzed = song.get('analyzed_chord_progressions_and_lyrics', [])
        for phrase_idx, section in enumerate(analyzed):
            lyric = section.get('lyric', '').strip()
            if not lyric:
                continue
            
            phrase_data = {
                'song_id': song_id,
                'phrase_id': f"{song_id}_phrase_{phrase_idx}",
                'phrase_idx': phrase_idx,
                'text': lyric,
                'metadata': {
                    'chord_progression': section.get('chord_progression', []),
                    'normalized_chord_progression': section.get('normalized_chord_progression', []),
                    'key': section.get('key'),
                    'emotion': section.get('emotion', {}),
                }
            }
            phrases.append(phrase_data)
    
    logger.info(f"Extracted {len(phrases)} phrases from {len(songs)} songs")
    return phrases, song_metadata


def remove_duplicates(
    phrases: List[Dict[str, Any]],
    config: DataProcessingConfig
) -> List[Dict[str, Any]]:
    """
    曲内で重複するフレーズを間引く
    
    Args:
        phrases: フレーズデータのリスト
        config: データ処理設定
    
    Returns:
        間引き後のフレーズリスト
    """
    # 曲ごとにグループ化
    song_phrases = defaultdict(list)
    for phrase in phrases:
        song_phrases[phrase['song_id']].append(phrase)
    
    filtered_phrases = []
    removed_count = 0
    
    for song_id, song_phrase_list in song_phrases.items():
        # テキストごとにカウント
        text_counter = Counter(p['text'] for p in song_phrase_list)
        
        # 各テキストについて、max_duplicate_countを超える場合は間引く
        text_indices = defaultdict(list)
        for idx, phrase in enumerate(song_phrase_list):
            text_indices[phrase['text']].append(idx)
        
        kept_indices = set()
        for text, indices in text_indices.items():
            count = text_counter[text]
            if count > config.max_duplicate_count:
                # 最初のmax_duplicate_count個だけ残す
                kept_indices.update(indices[:config.max_duplicate_count])
                removed_count += count - config.max_duplicate_count
            else:
                kept_indices.update(indices)
        
        # 保持するフレーズを追加
        for idx in sorted(kept_indices):
            filtered_phrases.append(song_phrase_list[idx])
    
    logger.info(f"Removed {removed_count} duplicate phrases")
    logger.info(f"Remaining phrases: {len(filtered_phrases)}")
    return filtered_phrases


def combine_short_phrases(
    phrases: List[Dict[str, Any]],
    config: DataProcessingConfig
) -> List[Dict[str, Any]]:
    """
    短すぎるフレーズを隣接フレーズと結合
    
    Args:
        phrases: フレーズデータのリスト
        config: データ処理設定
    
    Returns:
        結合後のフレーズリスト
    """
    # 曲ごとにグループ化
    song_phrases = defaultdict(list)
    for phrase in phrases:
        song_phrases[phrase['song_id']].append(phrase)
    
    combined_phrases = []
    
    for song_id, song_phrase_list in song_phrases.items():
        # インデックス順にソート
        song_phrase_list.sort(key=lambda x: x['phrase_idx'])
        
        i = 0
        while i < len(song_phrase_list):
            phrase = song_phrase_list[i]
            text = phrase['text']
            
            # 短すぎる場合は結合を試みる
            if len(text) < config.min_phrase_length:
                # 前後のフレーズを取得
                window_start = max(0, i - config.combine_window)
                window_end = min(len(song_phrase_list), i + config.combine_window + 1)
                window_phrases = song_phrase_list[window_start:window_end]
                
                # 短いフレーズを含む範囲で結合
                combined_text = " ".join(p['text'] for p in window_phrases)
                
                # 結合後のテキストが適切な長さになったら採用
                if len(combined_text) >= config.min_phrase_length:
                    # 結合されたフレーズを作成
                    combined_phrase = {
                        'song_id': song_id,
                        'phrase_id': f"{song_id}_combined_{i}",
                        'phrase_idx': phrase['phrase_idx'],
                        'text': combined_text,
                        'metadata': phrase['metadata'].copy(),  # 最初のフレーズのメタデータを使用
                    }
                    combined_phrases.append(combined_phrase)
                    # 結合に使用したフレーズをスキップ
                    i = window_end
                    continue
            
            # 結合不要な場合はそのまま追加
            combined_phrases.append(phrase)
            i += 1
    
    logger.info(f"Combined short phrases: {len(phrases)} -> {len(combined_phrases)}")
    return combined_phrases


def clean_phrases(
    phrases: List[Dict[str, Any]],
    config: DataProcessingConfig
) -> List[Dict[str, Any]]:
    """
    空文字や1文字トークンを除去
    
    Args:
        phrases: フレーズデータのリスト
        config: データ処理設定
    
    Returns:
        クリーンアップ後のフレーズリスト
    """
    cleaned = []
    
    for phrase in phrases:
        text = phrase['text']
        
        # 空文字除去
        if config.remove_empty and not text.strip():
            continue
        
        # 1文字トークン除去（分割由来の""など）
        if config.remove_single_char and len(text.strip()) <= 1:
            continue
        
        cleaned.append(phrase)
    
    removed = len(phrases) - len(cleaned)
    if removed > 0:
        logger.info(f"Removed {removed} empty/single-char phrases")
    
    return cleaned


def preprocess_data(
    input_dir: str | Path,
    config: Config,
    max_files: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    データの読み込みと前処理を実行
    
    Args:
        input_dir: 入力ディレクトリ
        config: 設定
        max_files: 最大ファイル数
    
    Returns:
        (phrases, song_metadata)
    """
    logger.info("Loading data...")
    songs = load_json_files(input_dir, max_files)
    
    logger.info("Extracting phrases...")
    phrases, song_metadata = extract_phrases(songs)
    
    logger.info("Removing duplicates...")
    phrases = remove_duplicates(phrases, config.data_processing)
    
    logger.info("Combining short phrases...")
    phrases = combine_short_phrases(phrases, config.data_processing)
    
    logger.info("Cleaning phrases...")
    phrases = clean_phrases(phrases, config.data_processing)
    
    logger.info(f"Preprocessing completed: {len(phrases)} phrases from {len(song_metadata)} songs")
    
    return phrases, song_metadata

