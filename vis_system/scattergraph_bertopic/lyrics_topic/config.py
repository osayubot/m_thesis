"""
設定ファイルの読み込みと管理
"""
from __future__ import annotations
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field


# 手動定義トピック（8カテゴリ）
MANUAL_TOPICS = {
    0: {
        'name': '恋愛',
        'description': '恋愛、好き、愛、恋、想い、あなた、君、告白、デート、付き合う、恋人、彼氏、彼女、片思い、両想い'
    },
    1: {
        'name': '別れ／未練',
        'description': '別れ、別れる、さよなら、未練、後悔、涙、泣く、つらい、悲しい、終わり、去る、離れる、破局'
    },
    2: {
        'name': '夢・未来・応援',
        'description': '夢、未来、希望、応援、頑張る、努力、目標、夢中、叶う、チャレンジ、前進、成長、エール'
    },
    3: {
        'name': '日常／等身大',
        'description': '日常、普通、平凡、生活、朝、昼、夕方、何気ない、ありふれた、リアル、現実、日常会話'
    },
    4: {
        'name': '夜・都会',
        'description': '夜、都会、街、夜景、ネオン、都市、ビル、夜更かし、夜道、繁華街、都会の生活、夜の街'
    },
    5: {
        'name': '季節（夏・冬）',
        'description': '夏、冬、季節、暑い、寒い、雪、雨、太陽、花火、祭り、クリスマス、夏休み、冬休み'
    },
    6: {
        'name': '内省／孤独',
        'description': '孤独、一人、内省、考える、静か、寂しい、思い出す、過去、記憶、自分、心、内面、独り'
    },
    7: {
        'name': '前進／決意',
        'description': '前進、決意、決める、変わる、新しい、スタート、始まり、挑戦、覚悟、決断、進む、歩く'
    }
}


@dataclass
class EmbeddingConfig:
    """Embedding設定"""
    model_name: str = "intfloat/multilingual-e5-base"
    normalize_embeddings: bool = True
    use_e5_prefix: bool = True
    batch_size: int = 32
    device: Optional[str] = None  # Noneなら自動判定
    cache_dir: Optional[str] = None


@dataclass
class UMAPConfig:
    """UMAP設定"""
    n_neighbors: int = 20
    n_components: int = 5
    min_dist: float = 0.1
    metric: str = "cosine"
    random_state: int = 42


@dataclass
class HDBSCANConfig:
    """HDBSCAN設定"""
    min_cluster_size: int = 120
    min_samples: int = 5
    cluster_selection_method: str = "eom"
    prediction_data: bool = True


@dataclass
class VectorizerConfig:
    """Vectorizer設定"""
    ngram_range: tuple = (1, 2)
    min_df: Union[int, float] = 2  # 絶対値（int）または相対値（float 0.0-1.0）
    max_df: float = 0.7
    stop_words: list = field(default_factory=lambda: [
        'yeah', 'oh', 'ah', 'woo', 'na', 'lalala', 'ららら', 'ラララ'
    ])


@dataclass
class CTFIDFConfig:
    """c-TF-IDF設定"""
    reduce_frequent_words: bool = True


@dataclass
class DataProcessingConfig:
    """データ前処理設定"""
    min_phrase_length: int = 12  # これより短いフレーズは結合を試みる
    max_duplicate_count: int = 2  # 曲内で同一lyricがこれより多い場合は間引く
    combine_window: int = 3  # 短いフレーズ結合時の前後ウィンドウ
    remove_empty: bool = True
    remove_single_char: bool = True


@dataclass
class TopicModelConfig:
    """トピックモデル設定"""
    calculate_probabilities: bool = True
    top_k_free_topics: int = 3  # 保存する上位k個の自由トピック確率
    target_free_topics: int = 20  # reduce_topics後の目標トピック数
    reduce_topics: bool = True


@dataclass
class MappingConfig:
    """マッピング設定"""
    use_softmax: bool = True
    temperature: float = 1.0  # softmaxの温度パラメータ


@dataclass
class Config:
    """全体設定"""
    # 基本設定
    random_seed: int = 42
    log_level: str = "INFO"
    
    # 各コンポーネント設定
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    umap: UMAPConfig = field(default_factory=UMAPConfig)
    hdbscan: HDBSCANConfig = field(default_factory=HDBSCANConfig)
    vectorizer: VectorizerConfig = field(default_factory=VectorizerConfig)
    ctfidf: CTFIDFConfig = field(default_factory=CTFIDFConfig)
    data_processing: DataProcessingConfig = field(default_factory=DataProcessingConfig)
    topic_model: TopicModelConfig = field(default_factory=TopicModelConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    
    # パス設定
    cache_dir: str = "cache"
    output_dir: str = "output"
    
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> Config:
        """YAMLファイルから設定を読み込む"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # ネストされた設定を構築
        config = cls()
        
        if 'random_seed' in data:
            config.random_seed = data['random_seed']
        if 'log_level' in data:
            config.log_level = data['log_level']
        if 'cache_dir' in data:
            config.cache_dir = data['cache_dir']
        if 'output_dir' in data:
            config.output_dir = data['output_dir']
        
        # Embedding設定
        if 'embedding' in data:
            emb_data = data['embedding']
            config.embedding = EmbeddingConfig(**emb_data)
        
        # UMAP設定
        if 'umap' in data:
            umap_data = data['umap']
            config.umap = UMAPConfig(**umap_data)
        
        # HDBSCAN設定
        if 'hdbscan' in data:
            hdbscan_data = data['hdbscan']
            config.hdbscan = HDBSCANConfig(**hdbscan_data)
        
        # Vectorizer設定
        if 'vectorizer' in data:
            vec_data = data['vectorizer']
            if 'ngram_range' in vec_data:
                vec_data['ngram_range'] = tuple(vec_data['ngram_range'])
            config.vectorizer = VectorizerConfig(**vec_data)
        
        # c-TF-IDF設定
        if 'ctfidf' in data:
            ctfidf_data = data['ctfidf']
            config.ctfidf = CTFIDFConfig(**ctfidf_data)
        
        # データ処理設定
        if 'data_processing' in data:
            dp_data = data['data_processing']
            config.data_processing = DataProcessingConfig(**dp_data)
        
        # トピックモデル設定
        if 'topic_model' in data:
            tm_data = data['topic_model']
            config.topic_model = TopicModelConfig(**tm_data)
        
        # マッピング設定
        if 'mapping' in data:
            map_data = data['mapping']
            config.mapping = MappingConfig(**map_data)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書に変換"""
        return {
            'random_seed': self.random_seed,
            'log_level': self.log_level,
            'cache_dir': self.cache_dir,
            'output_dir': self.output_dir,
            'embedding': {
                'model_name': self.embedding.model_name,
                'normalize_embeddings': self.embedding.normalize_embeddings,
                'use_e5_prefix': self.embedding.use_e5_prefix,
                'batch_size': self.embedding.batch_size,
                'device': self.embedding.device,
                'cache_dir': self.embedding.cache_dir,
            },
            'umap': {
                'n_neighbors': self.umap.n_neighbors,
                'n_components': self.umap.n_components,
                'min_dist': self.umap.min_dist,
                'metric': self.umap.metric,
                'random_state': self.umap.random_state,
            },
            'hdbscan': {
                'min_cluster_size': self.hdbscan.min_cluster_size,
                'min_samples': self.hdbscan.min_samples,
                'cluster_selection_method': self.hdbscan.cluster_selection_method,
                'prediction_data': self.hdbscan.prediction_data,
            },
            'vectorizer': {
                'ngram_range': list(self.vectorizer.ngram_range),
                'min_df': self.vectorizer.min_df,
                'max_df': self.vectorizer.max_df,
                'stop_words': self.vectorizer.stop_words,
            },
            'ctfidf': {
                'reduce_frequent_words': self.ctfidf.reduce_frequent_words,
            },
            'data_processing': {
                'min_phrase_length': self.data_processing.min_phrase_length,
                'max_duplicate_count': self.data_processing.max_duplicate_count,
                'combine_window': self.data_processing.combine_window,
                'remove_empty': self.data_processing.remove_empty,
                'remove_single_char': self.data_processing.remove_single_char,
            },
            'topic_model': {
                'calculate_probabilities': self.topic_model.calculate_probabilities,
                'top_k_free_topics': self.topic_model.top_k_free_topics,
                'target_free_topics': self.topic_model.target_free_topics,
                'reduce_topics': self.topic_model.reduce_topics,
            },
            'mapping': {
                'use_softmax': self.mapping.use_softmax,
                'temperature': self.mapping.temperature,
            },
        }

