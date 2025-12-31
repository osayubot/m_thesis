"""
Embedding計算とキャッシュ管理
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import List, Optional
import hashlib
import logging
import pickle
import pyarrow.parquet as pq
import pyarrow as pa
from sentence_transformers import SentenceTransformer
import torch

from .config import Config, EmbeddingConfig
from .utils import setup_logging, detect_device

logger = setup_logging()


def get_cache_path(texts: List[str], config: EmbeddingConfig, cache_dir: Path) -> Path:
    """キャッシュファイルのパスを生成"""
    # テキストのハッシュを計算（最初の1000文字と総文字数で簡易ハッシュ）
    text_hash = hashlib.md5(
        f"{len(texts)}_{sum(len(t) for t in texts[:100])}".encode()
    ).hexdigest()
    
    model_name_safe = config.model_name.replace("/", "_").replace("-", "_")
    cache_file = cache_dir / f"embeddings_{model_name_safe}_{text_hash}.parquet"
    return cache_file


def load_embeddings_from_cache(cache_path: Path) -> Optional[np.ndarray]:
    """キャッシュからembeddingを読み込む"""
    if not cache_path.exists():
        return None
    
    try:
        table = pq.read_table(cache_path)
        embeddings = table['embedding'].to_numpy()
        # リストのリストをnumpy配列に変換
        embeddings = np.array([np.array(e) for e in embeddings])
        logger.info(f"Loaded embeddings from cache: {cache_path}")
        return embeddings
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")
        return None


def save_embeddings_to_cache(embeddings: np.ndarray, cache_path: Path):
    """Embeddingをキャッシュに保存"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # numpy配列をリストのリストに変換（Parquet互換性のため）
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        table = pa.Table.from_arrays(
            [embeddings_list],
            names=['embedding']
        )
        pq.write_table(table, cache_path)
        logger.info(f"Saved embeddings to cache: {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")


class EmbeddingCalculator:
    """Embedding計算クラス"""
    
    def __init__(self, config: EmbeddingConfig, cache_dir: Path):
        self.config = config
        self.cache_dir = cache_dir
        self.model = None
        self.device = config.device or detect_device()
        
        # キャッシュディレクトリを作成
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_model(self):
        """モデルを読み込む（遅延読み込み）"""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.config.model_name}")
            self.model = SentenceTransformer(
                self.config.model_name,
                device=self.device,
                cache_folder=self.config.cache_dir
            )
            logger.info(f"Model loaded on device: {self.device}")
    
    def compute_embeddings(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Embeddingを計算（キャッシュを使用可能）
        
        Args:
            texts: テキストのリスト
            use_cache: キャッシュを使用するか
        
        Returns:
            Embedding配列 (n_samples, n_dim)
        """
        if not texts:
            return np.array([])
        
        # キャッシュを確認
        if use_cache:
            cache_path = get_cache_path(texts, self.config, self.cache_dir)
            cached_emb = load_embeddings_from_cache(cache_path)
            if cached_emb is not None:
                if len(cached_emb) == len(texts):
                    return cached_emb
                else:
                    logger.warning("Cache size mismatch, recomputing...")
        
        # モデルを読み込む
        self._load_model()
        
        # E5 prefixを適用
        if self.config.use_e5_prefix:
            # E5モデルの場合、query prefixを追加
            if "e5" in self.config.model_name.lower():
                texts = [f"query: {text}" for text in texts]
        
        logger.info(f"Computing embeddings for {len(texts)} texts...")
        
        # バッチ処理でembeddingを計算
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True
        )
        
        # キャッシュに保存
        if use_cache:
            save_embeddings_to_cache(embeddings, cache_path)
        
        logger.info(f"Computed embeddings: shape {embeddings.shape}")
        return embeddings
    
    def compute_embedding_single(self, text: str) -> np.ndarray:
        """単一テキストのembeddingを計算"""
        return self.compute_embeddings([text], use_cache=False)[0]

