"""
出力データの生成
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from .config import Config, MANUAL_TOPICS
from .topic_model import TopicModelPipeline
from .mapping import TopicMapper
from .utils import setup_logging, calculate_entropy

logger = setup_logging()


def create_phrase_level_output(
    phrases: List[Dict[str, Any]],
    free_topics: List[int],
    free_topic_probs: List[Dict[int, float]],
    manual_probs: np.ndarray,
    output_path: Path
):
    """
    フレーズレベルの出力を作成
    
    Args:
        phrases: フレーズデータのリスト
        free_topics: 各フレーズの自由トピックID
        free_topic_probs: 各フレーズの自由トピック確率（上位k）
        manual_probs: 各フレーズの手動トピック確率 (n_phrases, n_manual)
        output_path: 出力パス
    """
    logger.info("Creating phrase-level output...")
    
    # データフレームを作成
    data = []
    
    for idx, phrase in enumerate(phrases):
        row = {
            'song_id': phrase['song_id'],
            'phrase_id': phrase['phrase_id'],
            'phrase_idx': phrase['phrase_idx'],
            'text': phrase['text'],
            'free_topic_id': free_topics[idx] if idx < len(free_topics) else -1,
        }
        
        # 自由トピック確率（上位k）を文字列として保存
        if idx < len(free_topic_probs):
            free_probs = free_topic_probs[idx]
            # JSON形式の文字列として保存
            import json
            row['free_topic_probs'] = json.dumps(free_probs)
        else:
            row['free_topic_probs'] = "{}"
        
        # 手動トピック確率を各カラムとして保存
        if idx < len(manual_probs):
            manual_prob_row = manual_probs[idx]
            for manual_id in sorted(MANUAL_TOPICS.keys()):
                col_name = f"manual_prob_{manual_id}_{MANUAL_TOPICS[manual_id]['name']}"
                row[col_name] = float(manual_prob_row[manual_id])
        
        # メタデータをJSON文字列として保存
        import json
        row['metadata'] = json.dumps(phrase.get('metadata', {}))
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Parquet形式で保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine='pyarrow')
    
    logger.info(f"Saved phrase-level output: {output_path} ({len(df)} phrases)")


def create_topic_info_output(
    topic_model_pipeline: TopicModelPipeline,
    topic_mapper: Optional[TopicMapper],
    output_path: Path
):
    """
    トピック情報の出力を作成
    
    Args:
        topic_model_pipeline: トピックモデルパイプライン
        topic_mapper: トピックマッパー（オプション）
        output_path: 出力パス
    """
    logger.info("Creating topic info output...")
    
    topic_info = topic_model_pipeline.get_topic_info()
    
    data = []
    for topic_id, info in topic_info.items():
        row = {
            'topic_id': topic_id,
            'size': info['size'],
            'top_words': ', '.join(info['words'][:10]),
            'representative_docs': ' | '.join(info['representative_docs'][:3]),
        }
        
        # マッピング情報があれば追加
        if topic_mapper:
            mapping_info = topic_mapper.get_mapping_info()
            if topic_id in mapping_info:
                map_info = mapping_info[topic_id]
                row['top_manual_topics'] = ', '.join(
                    f"{m['manual_topic_name']}({m['prob']:.2f})"
                    for m in map_info['top_manual_topics']
                )
                row['mapping_entropy'] = map_info['entropy']
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # CSV形式で保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    logger.info(f"Saved topic info output: {output_path} ({len(df)} topics)")


def create_song_level_output(
    phrases: List[Dict[str, Any]],
    free_topics: List[int],
    manual_probs: np.ndarray,
    song_metadata: Dict[str, Any],
    output_path: Path
):
    """
    曲レベルの出力を作成
    
    Args:
        phrases: フレーズデータのリスト
        free_topics: 各フレーズの自由トピックID
        manual_probs: 各フレーズの手動トピック確率 (n_phrases, n_manual)
        song_metadata: 曲メタデータ
        output_path: 出力パス
    """
    logger.info("Creating song-level output...")
    
    # 曲ごとにグループ化
    from collections import defaultdict
    song_phrases = defaultdict(list)
    
    for idx, phrase in enumerate(phrases):
        song_id = phrase['song_id']
        song_phrases[song_id].append({
            'phrase_idx': idx,
            'free_topic_id': free_topics[idx] if idx < len(free_topics) else -1,
            'manual_probs': manual_probs[idx] if idx < len(manual_probs) else None,
        })
    
    # 曲ごとに集約
    data = []
    for song_id, phrase_list in song_phrases.items():
        # メタデータを取得
        metadata = song_metadata.get(song_id, {})
        
        # 手動トピック確率を平均（重み付き平均も可能）
        if phrase_list and phrase_list[0]['manual_probs'] is not None:
            manual_probs_list = [p['manual_probs'] for p in phrase_list if p['manual_probs'] is not None]
            if manual_probs_list:
                avg_manual_probs = np.mean(manual_probs_list, axis=0)
            else:
                avg_manual_probs = np.ones(len(MANUAL_TOPICS)) / len(MANUAL_TOPICS)
        else:
            avg_manual_probs = np.ones(len(MANUAL_TOPICS)) / len(MANUAL_TOPICS)
        
        # 主要なトピックを取得
        dominant_manual_topic_id = int(np.argmax(avg_manual_probs))
        dominant_manual_topic_name = MANUAL_TOPICS[dominant_manual_topic_id]['name']
        
        # 自由トピックの分布
        free_topic_ids = [p['free_topic_id'] for p in phrase_list if p['free_topic_id'] != -1]
        if free_topic_ids:
            from collections import Counter
            free_topic_counter = Counter(free_topic_ids)
            dominant_free_topic_id = free_topic_counter.most_common(1)[0][0]
        else:
            dominant_free_topic_id = -1
        
        row = {
            'song_id': song_id,
            'title': metadata.get('title', ''),
            'artist': metadata.get('artist', ''),
            'n_phrases': len(phrase_list),
            'dominant_free_topic_id': dominant_free_topic_id,
            'dominant_manual_topic_id': dominant_manual_topic_id,
            'dominant_manual_topic_name': dominant_manual_topic_name,
        }
        
        # 手動トピック確率を各カラムとして保存
        for manual_id in sorted(MANUAL_TOPICS.keys()):
            col_name = f"manual_prob_{manual_id}_{MANUAL_TOPICS[manual_id]['name']}"
            row[col_name] = float(avg_manual_probs[manual_id])
        
        # エントロピーを計算
        row['entropy'] = float(calculate_entropy(avg_manual_probs))
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Parquet形式で保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine='pyarrow')
    
    logger.info(f"Saved song-level output: {output_path} ({len(df)} songs)")


def create_evaluation_output(
    manual_probs: np.ndarray,
    free_topics: List[int],
    output_path: Path
):
    """
    評価用の出力を作成
    
    Args:
        manual_probs: 各フレーズの手動トピック確率
        free_topics: 各フレーズの自由トピックID
        output_path: 出力パス
    """
    logger.info("Creating evaluation output...")
    
    # エントロピー分布
    entropies = calculate_entropy(manual_probs, axis=1)
    
    # -1率
    noise_ratio = np.sum(np.array(free_topics) == -1) / len(free_topics)
    
    # 統計情報
    stats = {
        'n_phrases': len(manual_probs),
        'noise_ratio': float(noise_ratio),
        'entropy_mean': float(np.mean(entropies)),
        'entropy_std': float(np.std(entropies)),
        'entropy_min': float(np.min(entropies)),
        'entropy_max': float(np.max(entropies)),
    }
    
    # 各手動トピックの平均確率
    avg_probs = np.mean(manual_probs, axis=0)
    for manual_id in sorted(MANUAL_TOPICS.keys()):
        stats[f"avg_prob_{MANUAL_TOPICS[manual_id]['name']}"] = float(avg_probs[manual_id])
    
    # JSON形式で保存
    import json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved evaluation output: {output_path}")
    logger.info(f"  Noise ratio: {noise_ratio:.2%}")
    logger.info(f"  Mean entropy: {np.mean(entropies):.3f}")

