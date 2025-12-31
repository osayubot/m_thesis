"""
BERTopicモデルの構築と管理
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer

from .config import Config
from .embedding import EmbeddingCalculator
from .utils import setup_logging

logger = setup_logging()


class TopicModelPipeline:
    """BERTopicパイプライン"""
    
    def __init__(self, config: Config, embedding_calculator: EmbeddingCalculator):
        self.config = config
        self.embedding_calculator = embedding_calculator
        self.topic_model: Optional[BERTopic] = None
        self.free_topics: Optional[List[int]] = None
        self.free_topic_probs: Optional[np.ndarray] = None
    
    def _create_umap_model(self) -> UMAP:
        """UMAPモデルを作成"""
        return UMAP(
            n_neighbors=self.config.umap.n_neighbors,
            n_components=self.config.umap.n_components,
            min_dist=self.config.umap.min_dist,
            metric=self.config.umap.metric,
            random_state=self.config.umap.random_state,
            low_memory=False,  # メモリに余裕がある場合
        )
    
    def _create_hdbscan_model(self) -> HDBSCAN:
        """HDBSCANモデルを作成"""
        return HDBSCAN(
            min_cluster_size=self.config.hdbscan.min_cluster_size,
            min_samples=self.config.hdbscan.min_samples,
            cluster_selection_method=self.config.hdbscan.cluster_selection_method,
            prediction_data=self.config.hdbscan.prediction_data,
        )
    
    def _create_vectorizer(self) -> CountVectorizer:
        """CountVectorizerを作成"""
        min_df = self.config.vectorizer.min_df
        max_df = self.config.vectorizer.max_df
        
        # min_dfが整数で大きすぎる場合、相対値に変換
        # これにより、トピックごとのドキュメント数が少ない場合でも動作する
        if isinstance(min_df, int) and min_df > 5:
            # 相対値に変換（最小1%のドキュメントに出現する単語）
            min_df = max(0.01, min_df / 1000.0)
            logger.info(f"Converting min_df from absolute to relative value: {min_df}")
        
        return CountVectorizer(
            ngram_range=self.config.vectorizer.ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words=self.config.vectorizer.stop_words if self.config.vectorizer.stop_words else None,
        )
    
    def _create_ctfidf_model(self) -> ClassTfidfTransformer:
        """c-TF-IDFモデルを作成"""
        return ClassTfidfTransformer(
            reduce_frequent_words=self.config.ctfidf.reduce_frequent_words
        )
    
    def fit(
        self,
        texts: List[str],
        embeddings: Optional[np.ndarray] = None
    ) -> Tuple[List[int], Optional[np.ndarray]]:
        """
        BERTopicモデルを学習
        
        Args:
            texts: テキストのリスト
            embeddings: 事前計算済みのembedding（Noneの場合は計算）
        
        Returns:
            (topics, probs)
            topics: 各テキストのトピックID
            probs: 各テキストのトピック確率（calculate_probabilities=Trueの場合）
        """
        logger.info(f"Fitting BERTopic model on {len(texts)} texts...")
        
        # Embeddingを計算または使用
        if embeddings is None:
            embeddings = self.embedding_calculator.compute_embeddings(texts)
        
        # モデルコンポーネントを作成
        umap_model = self._create_umap_model()
        hdbscan_model = self._create_hdbscan_model()
        vectorizer = self._create_vectorizer()
        ctfidf_model = self._create_ctfidf_model()
        
        # BERTopicモデルを作成
        self.topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            ctfidf_model=ctfidf_model,
            calculate_probabilities=self.config.topic_model.calculate_probabilities,
            verbose=True,
        )
        
        # 学習（embeddingsを外部から渡す）
        # エラーハンドリング: max_dfとmin_dfの不整合が発生した場合、より緩い設定で再試行
        try:
            topics, probs = self.topic_model.fit_transform(texts, embeddings=embeddings)
        except ValueError as e:
            if "max_df corresponds to < documents than min_df" in str(e):
                logger.warning("Vectorizer error detected. Retrying with relaxed min_df...")
                # より緩い設定で再試行
                relaxed_vectorizer = CountVectorizer(
                    ngram_range=self.config.vectorizer.ngram_range,
                    min_df=0.01,  # 相対値1%
                    max_df=self.config.vectorizer.max_df,
                    stop_words=self.config.vectorizer.stop_words if self.config.vectorizer.stop_words else None,
                )
                # BERTopicモデルを再作成
                self.topic_model = BERTopic(
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    vectorizer_model=relaxed_vectorizer,
                    ctfidf_model=ctfidf_model,
                    calculate_probabilities=self.config.topic_model.calculate_probabilities,
                    verbose=True,
                )
                topics, probs = self.topic_model.fit_transform(texts, embeddings=embeddings)
            else:
                raise
        
        # ノイズ（-1）の割合を計算
        noise_ratio = np.sum(np.array(topics) == -1) / len(topics)
        n_topics = len(set(topics)) - (1 if -1 in topics else 0)
        
        logger.info(f"Initial topics: {n_topics} (noise ratio: {noise_ratio:.2%})")
        
        self.free_topics = list(topics)
        self.free_topic_probs = probs
        
        return topics, probs
    
    def reduce_topics(
        self,
        texts: List[str],
        target_nr_topics: Optional[int] = None
    ) -> Tuple[List[int], Optional[np.ndarray]]:
        """
        トピック数を削減
        
        Args:
            texts: テキストのリスト
            target_nr_topics: 目標トピック数（Noneの場合はconfigから取得）
        
        Returns:
            (reduced_topics, reduced_probs)
        """
        if self.topic_model is None:
            raise ValueError("Model must be fitted before reducing topics")
        
        if target_nr_topics is None:
            target_nr_topics = self.config.topic_model.target_free_topics
        
        # nr_topicsが整数であることを確認
        if not isinstance(target_nr_topics, int):
            target_nr_topics = int(target_nr_topics)
        
        logger.info(f"Reducing topics to {target_nr_topics}...")
        
        # reduce_topicsを実行
        # BERTopicのバージョンによって引数の扱いが異なる可能性があるため、
        # 複数の呼び出し方法を試す
        # 注意: BERTopicのreduce_topicsは通常、docsのみを受け取り、内部でtopicsを使用する
        # または、docsとtopicsを受け取り、nr_topicsをキーワード引数として受け取る
        
        # reduce_topicsを実行
        # BERTopicのバージョンによって、reduce_topicsは戻り値を返さない場合がある
        # その場合、in-placeで動作し、その後fit_transformを再度呼び出す必要がある
        
        # 方法1: docsとtopicsを位置引数として渡し、nr_topicsをキーワード引数として渡す
        try:
            result = self.topic_model.reduce_topics(
                texts,
                self.free_topics,
                nr_topics=target_nr_topics
            )
            # 戻り値がタプルの場合
            if isinstance(result, tuple) and len(result) == 2:
                reduced_topics, reduced_probs = result
            else:
                # 戻り値がNoneまたはBERTopicオブジェクトの場合、fit_transformを再度呼び出す
                reduced_topics, reduced_probs = self.topic_model.fit_transform(texts)
        except (TypeError, ValueError) as e:
            error_msg = str(e)
            logger.debug(f"Method 1 (docs + topics, nr_topics kwarg) failed: {error_msg}")
            
            # 方法2: nr_topicsを位置引数として渡す（topicsも位置引数）
            try:
                result = self.topic_model.reduce_topics(
                    texts,
                    self.free_topics,
                    target_nr_topics
                )
                # 戻り値がタプルの場合
                if isinstance(result, tuple) and len(result) == 2:
                    reduced_topics, reduced_probs = result
                else:
                    # 戻り値がNoneまたはBERTopicオブジェクトの場合、fit_transformを再度呼び出す
                    reduced_topics, reduced_probs = self.topic_model.fit_transform(texts)
            except (TypeError, ValueError) as e2:
                error_msg2 = str(e2)
                logger.debug(f"Method 2 (docs + topics + nr_topics all positional) failed: {error_msg2}")
                
                # 方法3: docsのみで、nr_topicsをキーワード引数として渡す
                try:
                    result = self.topic_model.reduce_topics(
                        texts,
                        nr_topics=target_nr_topics
                    )
                    # 戻り値がタプルの場合
                    if isinstance(result, tuple) and len(result) == 2:
                        reduced_topics, reduced_probs = result
                    else:
                        # 戻り値がNoneまたはBERTopicオブジェクトの場合、fit_transformを再度呼び出す
                        reduced_topics, reduced_probs = self.topic_model.fit_transform(texts)
                except (TypeError, ValueError) as e3:
                    error_msg3 = str(e3)
                    logger.debug(f"Method 3 (docs only, nr_topics kwarg) failed: {error_msg3}")
                    
                    # 方法4: docsのみで、nr_topicsを位置引数として渡す
                    try:
                        result = self.topic_model.reduce_topics(
                            texts,
                            target_nr_topics
                        )
                        # 戻り値がタプルの場合
                        if isinstance(result, tuple) and len(result) == 2:
                            reduced_topics, reduced_probs = result
                        else:
                            # 戻り値がNoneまたはBERTopicオブジェクトの場合、fit_transformを再度呼び出す
                            reduced_topics, reduced_probs = self.topic_model.fit_transform(texts)
                    except (TypeError, ValueError) as e4:
                        error_msg4 = str(e4)
                        logger.error(f"All methods failed. Last error: {error_msg4}")
                        raise ValueError(f"Could not call reduce_topics with any method. Last error: {error_msg4}")
        
        # ノイズの割合を計算
        noise_ratio = np.sum(np.array(reduced_topics) == -1) / len(reduced_topics)
        n_topics = len(set(reduced_topics)) - (1 if -1 in reduced_topics else 0)
        
        logger.info(f"Reduced topics: {n_topics} (noise ratio: {noise_ratio:.2%})")
        
        self.free_topics = list(reduced_topics)
        self.free_topic_probs = reduced_probs
        
        return reduced_topics, reduced_probs
    
    def get_topic_info(self) -> Dict[int, Dict[str, Any]]:
        """トピック情報を取得"""
        if self.topic_model is None:
            return {}
        
        try:
            topic_info_df = self.topic_model.get_topic_info()
        except Exception as e:
            logger.warning(f"Failed to get topic info: {e}")
            return {}
        
        topic_info = {}
        for _, row in topic_info_df.iterrows():
            topic_id = int(row['Topic'])
            if topic_id == -1:
                continue
            
            # トピックの代表語を取得
            try:
                topic_words = self.topic_model.get_topic(topic_id)
                if topic_words is not None:
                    if isinstance(topic_words, list):
                        words = [w[0] for w in topic_words[:10]]
                        scores = [w[1] for w in topic_words[:10]]
                    else:
                        # DataFrameの場合
                        words = topic_words['Word'].tolist()[:10]
                        scores = topic_words['Score'].tolist()[:10]
                else:
                    words = []
                    scores = []
            except Exception:
                words = []
                scores = []
            
            # 代表ドキュメントを取得
            try:
                representative_docs = self.topic_model.get_representative_docs(topic_id)
                if representative_docs is None:
                    representative_docs = []
            except Exception:
                representative_docs = []
            
            topic_info[topic_id] = {
                'size': int(row.get('Count', 0)),
                'words': words,
                'word_scores': scores,
                'representative_docs': representative_docs[:5],  # 上位5件
            }
        
        return topic_info
    
    def get_top_k_probs(
        self,
        probs: Optional[np.ndarray],
        k: Optional[int] = None
    ) -> List[Dict[int, float]]:
        """
        各テキストの上位k個のトピック確率を取得
        
        Args:
            probs: 確率配列 (n_samples, n_topics)
            k: 上位k個（Noneの場合はconfigから取得）
        
        Returns:
            各テキストの{topic_id: prob}のリスト
        """
        if probs is None:
            return [{} for _ in range(len(self.free_topics))]
        
        if k is None:
            k = self.config.topic_model.top_k_free_topics
        
        top_k_probs = []
        for prob_vec in probs:
            # 上位k個のインデックスを取得
            top_k_indices = np.argsort(prob_vec)[-k:][::-1]
            top_k_dict = {
                int(idx): float(prob_vec[idx])
                for idx in top_k_indices
                if prob_vec[idx] > 0
            }
            top_k_probs.append(top_k_dict)
        
        return top_k_probs

