"""
感情ごとの可視化データをJSON形式で出力するモジュール
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

from .emotion_data_extractor import EMOTIONS
from .dimension_reduction import (
    compute_mds_coordinates,
    compute_umap_coordinates,
    compute_tsne_coordinates,
    UMAP_AVAILABLE
)

# 感情から色へのマッピング
EMOTION_COLORS = {
    'JOY': '#FFFF73',
    'SADNESS': '#5150F8',
    'ANTICIPATION': '#F3AB63',
    'SURPRISE': '#74BBF9',
    'ANGER': '#E93323',
    'FEAR': '#429429',
    'DISGUST': '#EB60F8',
    'TRUST': '#88FC6E',
}


def normalize_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """
    座標を0-100の範囲に正規化
    
    Args:
        coordinates: 座標配列（n×2）
    
    Returns:
        正規化された座標配列
    """
    if coordinates.shape[0] == 0:
        return coordinates
    
    # 各次元ごとに最小値と最大値を計算
    min_vals = coordinates.min(axis=0)
    max_vals = coordinates.max(axis=0)
    
    # 範囲を計算
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1  # ゼロ除算を防ぐ
    
    # 正規化
    normalized = ((coordinates - min_vals) / ranges) * 100
    
    return normalized


def generate_json_data(
    emotion_data: Dict[str, List[Dict]],
    method: str,
    output_dir: str,
    max_items_per_emotion: Optional[int] = 10000
) -> None:
    """
    感情ごとにJSONファイルを生成
    
    Args:
        emotion_data: 感情ごとのデータ辞書
        method: 次元削減手法（'mds', 'umap', 'tsne'）
        output_dir: 出力ディレクトリ
        max_items_per_emotion: 感情ごとの最大アイテム数（メモリ節約のため）
    """
    import random
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for emotion in EMOTIONS:
        data_list = emotion_data[emotion]
        
        if len(data_list) == 0:
            print(f"Warning: No data for emotion {emotion}, skipping...")
            continue
        
        # データが多い場合はサンプリング（メモリ節約）
        original_count = len(data_list)
        if max_items_per_emotion and len(data_list) > max_items_per_emotion:
            print(f"  Warning: {emotion} has {len(data_list)} items, sampling to {max_items_per_emotion} for {method.upper()}...")
            random.seed(42)  # 再現性のため
            data_list = random.sample(data_list, max_items_per_emotion)
        
        print(f"Processing {emotion} ({len(data_list)} items{'(sampled from ' + str(original_count) + ')' if original_count > len(data_list) else ''}) with {method.upper()}...")
        
        # 次元削減を実行
        if method.lower() == 'mds':
            coordinates = compute_mds_coordinates(data_list, random_state=42)
        elif method.lower() == 'umap':
            if not UMAP_AVAILABLE:
                print(f"Warning: UMAP is not available, skipping {emotion}...")
                continue
            coordinates = compute_umap_coordinates(data_list, random_state=42)
            if coordinates is None:
                print(f"Warning: Failed to compute UMAP coordinates for {emotion}, skipping...")
                continue
        elif method.lower() == 'tsne':
            coordinates = compute_tsne_coordinates(data_list, random_state=42)
        else:
            print(f"Error: Unknown method {method}, skipping {emotion}...")
            continue
        
        # 座標を正規化
        coordinates = normalize_coordinates(coordinates)
        
        # ポイントデータを構築
        points = []
        for i, (data, coord) in enumerate(zip(data_list, coordinates)):
            # 感情分布を計算（pieチャート用）
            emotion_dict = data['emotion']
            total = sum(emotion_dict.values()) if emotion_dict.values() else 1.0
            pie = []
            
            # このポイントの歌詞を取得
            lyric_text = data.get('lyric', '')
            song_info = data.get('song_info', {})
            
            # 最大値の感情を特定（このポイントが属する感情）
            max_emo = max(emotion_dict.items(), key=lambda x: x[1])[0] if emotion_dict else None
            
            for emo in EMOTIONS:
                value = emotion_dict.get(emo, 0.0) / total * 100 if total > 0 else 0.0
                
                # 最大値の感情のセグメントにのみ歌詞を追加
                lyrics = []
                if emo == max_emo and lyric_text:
                    lyrics.append({
                        'lyric': lyric_text,
                        'song': song_info
                    })
                
                pie.append({
                    'label': emo,
                    'v': round(value, 2),
                    'c': EMOTION_COLORS.get(emo, '#808080'),
                    'lyrics': lyrics
                })
            
            # コード進行を文字列化
            chord_prog = data.get('normalized_chord_progression', [])
            progression_str = ' - '.join(chord_prog) if chord_prog else 'N/A'
            
            # typical_chord_distanceから最も近い基準進行を決定（色付け用）
            typical_dist = data['typical_chord_distance']
            odo_dist = typical_dist.get('odo', float('inf'))
            komuro_dist = typical_dist.get('komuro', float('inf'))
            marusa_dist = typical_dist.get('marusa', float('inf'))
            
            # 最小距離を持つ基準進行を決定
            min_dist = min(odo_dist, komuro_dist, marusa_dist)
            if min_dist == odo_dist:
                reference_color = '#0000FF'  # 青（王道進行）
                reference_name = 'odo'
            elif min_dist == marusa_dist:
                reference_color = '#00FF00'  # 緑（マルサ進行）
                reference_name = 'marusa'
            else:
                reference_color = '#FF0000'  # 赤（小室進行）
                reference_name = 'komuro'
            
            # ポイントデータ
            point = {
                'x': float(coord[0]),
                'y': float(coord[1]),
                'r': 6,  # デフォルト半径
                'progression': progression_str,
                'chord_progression': chord_prog,
                'pie': pie,  # ツールチップ表示用
                'emotion': emotion_dict,  # 感情データ全体（ツールチップ用）
                'dominant_emotion': max_emo,  # 主感情
                'reference_color': reference_color,  # 基準進行に基づく色
                'reference_name': reference_name,  # 基準進行名
                'typical_chord_distance': data['typical_chord_distance'],
                'song_info': data['song_info'],
                'lyric': lyric_text  # 歌詞
            }
            
            points.append(point)
        
        # JSONデータを構築
        json_data = {
            'emotion': emotion,
            'method': method.upper(),
            'points': points,
            'total_points': len(points)
        }
        
        # ファイル名を生成（例: JOY_mds.json）
        filename = f"{emotion}_{method.lower()}.json"
        filepath = output_path / filename
        
        # JSONファイルに保存
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"  Saved {filename} ({len(points)} points)")


def main(
    emotion_data: Dict[str, List[Dict]],
    output_dir: str,
    methods: List[str] = ['umap', 'tsne', 'mds'],
    max_items_per_emotion: Optional[int] = 10000
) -> None:
    """
    メイン処理
    
    Args:
        emotion_data: 感情ごとのデータ辞書
        output_dir: 出力ディレクトリ
        methods: 使用する次元削減手法のリスト
        max_items_per_emotion: 感情ごとの最大アイテム数（メモリ節約のため、Noneの場合は制限なし）
    """
    for method in methods:
        if method.lower() == 'umap' and not UMAP_AVAILABLE:
            print(f"Warning: UMAP is not available, skipping...")
            continue
        
        generate_json_data(emotion_data, method, output_dir, max_items_per_emotion)
