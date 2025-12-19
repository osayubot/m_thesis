"""
感情ごとにコード進行データを抽出するモジュール
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# 8つの感情
EMOTIONS = ['JOY', 'TRUST', 'SADNESS', 'ANGER', 'FEAR', 'DISGUST', 'ANTICIPATION', 'SURPRISE']


def load_analyzed_data(data_dir: str, max_files: Optional[int] = None) -> List[Dict]:
    """
    分析済みデータを読み込む
    
    Args:
        data_dir: データディレクトリ
        max_files: 最大ファイル数（Noneの場合は全て）
    
    Returns:
        楽曲データのリスト
    """
    data_path = Path(data_dir)
    json_files = list(data_path.glob("*.json"))
    
    if max_files:
        json_files = json_files[:max_files]
    
    all_songs = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_songs.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    return all_songs


def extract_emotion_based_data(songs: List[Dict]) -> Dict[str, List[Dict]]:
    """
    感情ごとにコード進行データをグループ化
    
    Args:
        songs: 楽曲データのリスト
    
    Returns:
        感情名をキー、コード進行データのリストを値とする辞書
    """
    emotion_data = {emotion: [] for emotion in EMOTIONS}
    
    for song in songs:
        song_info = {
            'title': song.get('title', ''),
            'artist': song.get('artist', ''),
            'lyricist': song.get('lyricist', ''),
            'composer': song.get('composer', ''),
            'spotify_id': song.get('spotify_id', ''),
            'jtotal_path': song.get('jtotal_path', ''),
            'release_date': song.get('release_date', ''),
            'spotify_popularity': song.get('spotify_popularity', None)
        }
        
        analyzed = song.get('analyzed_chord_progressions_and_lyrics', [])
        
        for section in analyzed:
            # コード進行を取得
            chord_prog = section.get('normalized_chord_progression', [])
            if not chord_prog or chord_prog == ['N.C']:
                continue
            
            # 歌詞を取得
            lyric = section.get('lyric', '').strip()
            if not lyric:
                continue
            
            # 感情データを取得
            emotion = section.get('emotion', {})
            if not emotion:
                continue
            
            # 各感情について、0.5以上の値を持つものを対象の感情グループに追加
            # ただし、最大値を持つ感情を優先的に割り当てる（1つのデータが複数の感情グループに含まれないようにする）
            max_emotion_value = max(emotion.values()) if emotion.values() else 0.0
            if max_emotion_value < 0.5:
                continue
            
            # 最大値を持つ感情を特定（これを主感情とする）
            dominant_emotion = max(emotion.items(), key=lambda x: x[1])[0]
            if dominant_emotion not in EMOTIONS:
                continue
            
            # 主感情の値が0.5以上であることを確認
            if emotion.get(dominant_emotion, 0.0) < 0.5:
                continue
            
            # typical_chord_distanceを取得
            typical_dist = section.get('typical_chord_distance', {})
            if not typical_dist or 'odo' not in typical_dist or 'komuro' not in typical_dist or 'marusa' not in typical_dist:
                continue
            
            # コード進行のキー情報
            key = section.get('key', '')
            
            # データを追加
            emotion_data[dominant_emotion].append({
                'chord_progression': section.get('chord_progression', []),
                'normalized_chord_progression': chord_prog,
                'lyric': lyric,
                'emotion': emotion,
                'key': key,
                'typical_chord_distance': {
                    'odo': typical_dist['odo'],
                    'komuro': typical_dist['komuro'],
                    'marusa': typical_dist['marusa']
                },
                'song_info': song_info,
                'roman_progression': section.get('normalized_chord_progression', [])  # ローマ数字変換は後で必要なら追加
            })
    
    return emotion_data


def get_emotion_statistics(emotion_data: Dict[str, List[Dict]]) -> Dict[str, int]:
    """
    各感情のデータ数を取得
    
    Args:
        emotion_data: 感情ごとのデータ辞書
    
    Returns:
        感情名をキー、データ数を値とする辞書
    """
    return {emotion: len(data) for emotion, data in emotion_data.items()}
