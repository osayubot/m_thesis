"""
自由トピックから手動定義トピックへのマッピング
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import logging

from .config import Config, MANUAL_TOPICS
from .embedding import EmbeddingCalculator
from .topic_model import TopicModelPipeline
from .utils import setup_logging, softmax

logger = setup_logging()


class TopicMapper:
    """トピックマッパー"""
    
    def __init__(
        self,
        config: Config,
        embedding_calculator: EmbeddingCalculator,
        topic_model_pipeline: TopicModelPipeline
    ):
        self.config = config
        self.embedding_calculator = embedding_calculator
        self.topic_model_pipeline = topic_model_pipeline
        self.free_to_manual_matrix: Optional[np.ndarray] = None  # (n_free_topics, n_manual_topics)
    
    def _get_topic_representative_text(
        self,
        topic_id: int
    ) -> str:
        """
        自由トピックの代表テキストを取得
        
        Args:
            topic_id: 自由トピックID
        
        Returns:
            代表テキスト（top words + representative docs）
        """
        topic_info = self.topic_model_pipeline.get_topic_info()
        
        if topic_id not in topic_info:
            return ""
        
        info = topic_info[topic_id]
        
        # Top wordsを結合
        words = " ".join(info.get('words', [])[:10])
        
        # Representative docsを結合
        docs = " ".join(info.get('representative_docs', [])[:3])
        
        # 結合
        representative_text = f"{words} {docs}".strip()
        
        return representative_text
    
    def compute_mapping_matrix(
        self,
        free_topic_ids: List[int]
    ) -> np.ndarray:
        """
        自由トピック→手動トピックのマッピング行列を計算
        
        Args:
            free_topic_ids: 自由トピックIDのリスト（-1を除く）
        
        Returns:
            マッピング行列 (n_free_topics, n_manual_topics)
        """
        logger.info("Computing free topic → manual topic mapping matrix...")
        
        # 各自由トピックの代表テキストを取得
        free_topic_texts = []
        valid_topic_ids = []
        
        for topic_id in free_topic_ids:
            if topic_id == -1:
                continue
            text = self._get_topic_representative_text(topic_id)
            if text:
                free_topic_texts.append(text)
                valid_topic_ids.append(topic_id)
        
        if not free_topic_texts:
            logger.warning("No valid free topic texts found")
            return np.zeros((len(free_topic_ids), len(MANUAL_TOPICS)))
        
        # 各手動トピックの説明文を取得
        manual_topic_descriptions = [
            MANUAL_TOPICS[i]['description']
            for i in sorted(MANUAL_TOPICS.keys())
        ]
        
        # Embeddingを計算
        logger.info("Computing embeddings for free topic representatives...")
        free_embeddings = self.embedding_calculator.compute_embeddings(
            free_topic_texts,
            use_cache=False
        )
        
        logger.info("Computing embeddings for manual topic descriptions...")
        manual_embeddings = self.embedding_calculator.compute_embeddings(
            manual_topic_descriptions,
            use_cache=False
        )
        
        # コサイン類似度を計算
        # free_embeddings: (n_free_topics, dim)
        # manual_embeddings: (n_manual_topics, dim)
        similarities = np.dot(free_embeddings, manual_embeddings.T)  # (n_free_topics, n_manual_topics)
        
        # Softmaxで確率に変換
        if self.config.mapping.use_softmax:
            mapping_probs = softmax(
                similarities,
                temperature=self.config.mapping.temperature,
                axis=1
            )
        else:
            # 単純に正規化
            mapping_probs = similarities - similarities.min(axis=1, keepdims=True)
            mapping_probs = mapping_probs / (mapping_probs.sum(axis=1, keepdims=True) + 1e-10)
        
        # 全トピックID（-1含む）に対応する行列を作成
        all_topic_ids = sorted(set(free_topic_ids))
        full_matrix = np.zeros((len(all_topic_ids), len(MANUAL_TOPICS)))
        
        topic_id_to_idx = {tid: idx for idx, tid in enumerate(all_topic_ids)}
        for valid_id, valid_idx in zip(valid_topic_ids, range(len(valid_topic_ids))):
            if valid_id in topic_id_to_idx:
                full_matrix[topic_id_to_idx[valid_id]] = mapping_probs[valid_idx]
        
        self.free_to_manual_matrix = full_matrix
        
        logger.info(f"Computed mapping matrix: shape {full_matrix.shape}")
        
        return full_matrix
    
    def map_phrase_probs(
        self,
        free_topic_probs: List[Dict[int, float]],
        free_topics: List[int]
    ) -> np.ndarray:
        """
        フレーズの自由トピック確率を手動トピック確率に変換
        
        Args:
            free_topic_probs: 各フレーズの自由トピック確率 [{topic_id: prob}, ...]
            free_topics: 各フレーズの自由トピックID
        
        Returns:
            手動トピック確率配列 (n_phrases, n_manual_topics)
        """
        if self.free_to_manual_matrix is None:
            raise ValueError("Mapping matrix must be computed first")
        
        n_phrases = len(free_topic_probs)
        n_manual = len(MANUAL_TOPICS)
        manual_probs = np.zeros((n_phrases, n_manual))
        
        # 全自由トピックIDのリストを作成
        all_free_topic_ids = sorted(set(t for t in free_topics if t != -1))
        topic_id_to_idx = {tid: idx for idx, tid in enumerate(all_free_topic_ids)}
        
        for phrase_idx, (free_probs_dict, free_topic_id) in enumerate(
            zip(free_topic_probs, free_topics)
        ):
            if free_topic_id == -1 or free_topic_id not in topic_id_to_idx:
                # ノイズの場合は均等分布
                manual_probs[phrase_idx] = np.ones(n_manual) / n_manual
                continue
            
            # 自由トピックIDのインデックス
            free_idx = topic_id_to_idx[free_topic_id]
            
            # マッピング行列から該当行を取得
            mapping_row = self.free_to_manual_matrix[free_idx]  # (n_manual,)
            
            # 自由トピック確率で重み付け
            # free_probs_dictには上位k個の確率が入っている
            # 主要なトピック（free_topic_id）の確率を1.0として扱う
            # または、free_probs_dictの確率を重みとして使用
            
            # 簡易版: 主要トピックのマッピングをそのまま使用
            # より正確には、free_probs_dictの各トピックのマッピングを重み付け平均
            weighted_mapping = np.zeros(n_manual)
            total_weight = 0.0
            
            for topic_id, prob in free_probs_dict.items():
                if topic_id in topic_id_to_idx:
                    topic_idx = topic_id_to_idx[topic_id]
                    topic_mapping = self.free_to_manual_matrix[topic_idx]
                    weighted_mapping += prob * topic_mapping
                    total_weight += prob
            
            if total_weight > 0:
                manual_probs[phrase_idx] = weighted_mapping / total_weight
            else:
                # フォールバック: 主要トピックのマッピングを使用
                manual_probs[phrase_idx] = mapping_row
        
        # 正規化（合計が1になるように）
        row_sums = manual_probs.sum(axis=1, keepdims=True)
        manual_probs = manual_probs / (row_sums + 1e-10)
        
        return manual_probs
    
    def get_mapping_info(self) -> Dict[int, Dict[str, Any]]:
        """マッピング情報を取得"""
        if self.free_to_manual_matrix is None:
            return {}
        
        all_free_topic_ids = sorted(
            set(self.topic_model_pipeline.free_topics)
            if self.topic_model_pipeline.free_topics
            else []
        )
        all_free_topic_ids = [tid for tid in all_free_topic_ids if tid != -1]
        
        mapping_info = {}
        topic_id_to_idx = {tid: idx for idx, tid in enumerate(all_free_topic_ids)}
        
        for topic_id in all_free_topic_ids:
            if topic_id not in topic_id_to_idx:
                continue
            
            idx = topic_id_to_idx[topic_id]
            mapping_row = self.free_to_manual_matrix[idx]
            
            # 上位3つの手動トピックを取得
            top_manual_indices = np.argsort(mapping_row)[-3:][::-1]
            top_manual = [
                {
                    'manual_topic_id': int(manual_idx),
                    'manual_topic_name': MANUAL_TOPICS[manual_idx]['name'],
                    'prob': float(mapping_row[manual_idx])
                }
                for manual_idx in top_manual_indices
                if mapping_row[manual_idx] > 0.01
            ]
            
            mapping_info[topic_id] = {
                'top_manual_topics': top_manual,
                'entropy': float(-np.sum(mapping_row * np.log(mapping_row + 1e-10)))
            }
        
        return mapping_info

