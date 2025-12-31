"""
BERTopicによるトピック分析と基準進行との関係計算
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# BERTopicのインポート
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    print("Warning: bertopic is not installed. Please install it with: pip install bertopic sentence-transformers")

from ..visualize_scattergraph_data.musical_distance import circular_distance
from ..visualize_scattergraph_data.common import (
    REFERENCE_PROGRESSIONS,
    REFERENCE_COLORS,
    load_analyzed_data,
    extract_chord_progressions_with_lyrics
)

# 手動定義トピック
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
        'description': '夏、冬、季節、暑い、寒い、雪、雨、太陽、太陽、花火、祭り、クリスマス、夏休み、冬休み'
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


def extract_lyrics_for_topic_modeling(songs: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """
    トピックモデリング用に歌詞を抽出
    
    Args:
        songs: 楽曲データのリスト
    
    Returns:
        (lyrics_list, metadata_list)
        lyrics_list: 歌詞テキストのリスト
        metadata_list: 各歌詞に対応するメタデータ（コード進行、曲情報など）のリスト
    """
    lyrics_list = []
    metadata_list = []
    
    for song in songs:
        song_info = {
            'title': song.get('title', ''),
            'artist': song.get('artist', ''),
            'lyricist': song.get('lyricist', ''),
            'composer': song.get('composer', ''),
            'spotify_id': song.get('spotify_id', ''),
            'release_date': song.get('release_date', ''),
        }
        
        analyzed = song.get('analyzed_chord_progressions_and_lyrics', [])
        
        for section in analyzed:
            lyric = section.get('lyric', '').strip()
            if not lyric:
                continue
            
            # コード進行を取得
            chord_prog = section.get('normalized_chord_progression', [])
            if not chord_prog or chord_prog == ['N.C']:
                continue
            
            lyrics_list.append(lyric)
            metadata_list.append({
                **song_info,
                'chord_progression': section.get('chord_progression', []),
                'normalized_chord_progression': chord_prog,
                'key': section.get('key'),
                'emotion': section.get('emotion', {}),
            })
    
    return lyrics_list, metadata_list


def compute_topic_reference_distances(
    topic_progressions: Dict[int, List[str]],
    reference_progressions: Dict[str, List[str]] = None
) -> Dict[int, Dict[str, float]]:
    """
    各トピックの代表コード進行から基準進行への距離を計算
    
    Args:
        topic_progressions: {topic_id: [roman_progression, ...]} 各トピックに属するコード進行のリスト
        reference_progressions: 基準進行の辞書
    
    Returns:
        {topic_id: {reference_name: distance, ...}, ...}
    """
    if reference_progressions is None:
        reference_progressions = REFERENCE_PROGRESSIONS
    
    topic_distances = {}
    
    for topic_id, progressions in topic_progressions.items():
        if not progressions:
            continue
        
        # 各基準進行への距離を計算
        distances = {}
        for ref_name, ref_prog in reference_progressions.items():
            # トピック内のすべてのコード進行から基準進行への距離の平均を計算
            topic_distances_list = []
            for prog in progressions:
                if len(prog) == 4:  # 4コード進行のみ
                    dist = circular_distance(prog, ref_prog)
                    topic_distances_list.append(dist)
            
            if topic_distances_list:
                distances[ref_name] = np.mean(topic_distances_list)
            else:
                distances[ref_name] = float('inf')
        
        topic_distances[topic_id] = distances
    
    return topic_distances


def classify_lyrics_to_manual_topics(
    lyrics_list: List[str],
    metadata_list: List[Dict],
    topics_def: Dict[int, Dict[str, str]] = None
) -> Tuple[List[int], Dict[int, List[int]], Dict[int, List[List[str]]]]:
    """
    歌詞を手動定義されたトピックに分類
    
    Args:
        lyrics_list: 歌詞テキストのリスト
        metadata_list: 各歌詞のメタデータ
        topics_def: トピック定義の辞書（Noneの場合はMANUAL_TOPICSを使用）
    
    Returns:
        (topics, topic_docs, topic_progressions)
        topics: [topic_id, ...] 各歌詞に対応するトピックIDのリスト
        topic_docs: {topic_id: [doc_index, ...]} 各トピックに属するドキュメントのインデックス
        topic_progressions: {topic_id: [roman_progression, ...]} 各トピックに属するコード進行
    """
    if topics_def is None:
        topics_def = MANUAL_TOPICS
    
    # SentenceTransformerモデルを使用してエンベディングを計算
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    except ImportError:
        raise ImportError("sentence-transformers is required. Install it with: pip install sentence-transformers")
    
    print(f"Classifying {len(lyrics_list)} lyrics to {len(topics_def)} manual topics...")
    
    # 各トピックの説明文をエンベディング化
    topic_descriptions = [topics_def[topic_id]['description'] for topic_id in sorted(topics_def.keys())]
    topic_embeddings = model.encode(topic_descriptions, show_progress_bar=True)
    
    # 歌詞をエンベディング化
    print("Encoding lyrics...")
    lyrics_embeddings = model.encode(lyrics_list, show_progress_bar=True, batch_size=32)
    
    # 各歌詞を最も類似度が高いトピックに分類（コサイン類似度）
    from sklearn.metrics.pairwise import cosine_similarity
    topics = []
    topic_docs = defaultdict(list)
    topic_progressions = defaultdict(list)
    
    topic_ids = sorted(topics_def.keys())
    
    print("Classifying lyrics...")
    for doc_idx, lyric_emb in enumerate(lyrics_embeddings):
        # 各トピックとの類似度を計算
        similarities = cosine_similarity([lyric_emb], topic_embeddings)[0]
        # 最も類似度が高いトピックに分類
        best_topic_idx = similarities.argmax()
        topic_id = topic_ids[best_topic_idx]
        
        topics.append(topic_id)
        topic_docs[topic_id].append(doc_idx)
        
        # コード進行をローマ数字に変換
        metadata = metadata_list[doc_idx]
        chord_prog = metadata.get('normalized_chord_progression', [])
        key = metadata.get('key')
        if key and chord_prog:
            from ..analyze_data.roman_numeral import section_to_roman_progression
            section_for_roman = {
                'chord_progression': chord_prog,
                'key': key
            }
            roman_prog = section_to_roman_progression(section_for_roman, key)
            if roman_prog and len(roman_prog) == 4:
                topic_progressions[topic_id].append(roman_prog)
    
    print(f"Classification completed:")
    for topic_id in topic_ids:
        count = len(topic_docs[topic_id])
        print(f"  Topic {topic_id} ({topics_def[topic_id]['name']}): {count} lyrics")
    
    return topics, dict(topic_docs), dict(topic_progressions)


def perform_topic_modeling(
    lyrics_list: List[str],
    metadata_list: List[Dict],
    min_cluster_size: int = 30,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    use_manual_topics: bool = False
) -> Tuple[Optional[object], Dict[int, List[int]], Dict[int, List[List[str]]], List[int]]:
    """
    トピックモデリングを実行（BERTopicまたは手動トピック分類）
    
    Args:
        lyrics_list: 歌詞テキストのリスト
        metadata_list: 各歌詞のメタデータ
        min_cluster_size: HDBSCANの最小クラスタサイズ（BERTopic使用時のみ）
        n_neighbors: UMAPの近傍数（BERTopic使用時のみ）
        min_dist: UMAPの最小距離（BERTopic使用時のみ）
        use_manual_topics: Trueの場合は手動定義トピックを使用
    
    Returns:
        (topic_model, topic_docs, topic_progressions, topics)
        topic_model: BERTopicモデル（手動トピックの場合はNone）
        topic_docs: {topic_id: [doc_index, ...]} 各トピックに属するドキュメントのインデックス
        topic_progressions: {topic_id: [roman_progression, ...]} 各トピックに属するコード進行
        topics: [topic_id, ...] 各歌詞に対応するトピックIDのリスト
    """
    if use_manual_topics:
        # 手動定義トピックを使用
        topics, topic_docs, topic_progressions = classify_lyrics_to_manual_topics(
            lyrics_list, metadata_list
        )
        return None, topic_docs, topic_progressions, topics
    
    # BERTopicを使用
    if not BERTOPIC_AVAILABLE:
        raise ImportError("bertopic is not installed. Please install it with: pip install bertopic sentence-transformers")
    
    print(f"Performing topic modeling with min_cluster_size={min_cluster_size}...")
    print(f"Total lyrics: {len(lyrics_list)}")
    
    # 日本語用のSentenceTransformerモデルを使用
    # より良いモデルがあれば変更可能
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # BERTopicモデルの初期化
    # HDBSCANのパラメータを調整してトピック数を制御
    from hdbscan import HDBSCAN
    from umap import UMAP
    
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=5,
        min_dist=min_dist,
        metric='cosine',
        random_state=42
    )
    
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=10,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        language='japanese',
        verbose=True
    )
    
    # トピックモデリングを実行
    topics, probs = topic_model.fit_transform(lyrics_list)
    
    print(f"Found {len(set(topics)) - (1 if -1 in topics else 0)} topics (excluding noise)")
    
    # 各トピックに属するドキュメントのインデックスを集計
    topic_docs = defaultdict(list)
    topic_progressions = defaultdict(list)
    
    # コード進行をローマ数字に変換するための関数
    from ..analyze_data.roman_numeral import section_to_roman_progression
    
    for doc_idx, (topic_id, metadata) in enumerate(zip(topics, metadata_list)):
        if topic_id == -1:  # ノイズは除外
            continue
        
        topic_docs[topic_id].append(doc_idx)
        
        # コード進行をローマ数字に変換
        chord_prog = metadata.get('normalized_chord_progression', [])
        key = metadata.get('key')
        if key and chord_prog:
            section_for_roman = {
                'chord_progression': chord_prog,
                'key': key
            }
            roman_prog = section_to_roman_progression(section_for_roman, key)
            if roman_prog and len(roman_prog) == 4:
                topic_progressions[topic_id].append(roman_prog)
    
    return topic_model, dict(topic_docs), dict(topic_progressions), topics.tolist() if hasattr(topics, 'tolist') else list(topics)


def analyze_topics_for_multiple_cluster_sizes(
    lyrics_list: List[str],
    metadata_list: List[Dict],
    cluster_sizes: List[int] = [20, 30, 40],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    use_manual_topics: bool = False
) -> Dict[int, Dict]:
    """
    複数のmin_cluster_sizeでトピック分析を実行、または手動トピック分類を実行
    
    Args:
        lyrics_list: 歌詞テキストのリスト
        metadata_list: 各歌詞のメタデータ
        cluster_sizes: 試すmin_cluster_sizeのリスト（手動トピックの場合は無視される）
        n_neighbors: UMAPの近傍数（手動トピックの場合は無視される）
        min_dist: UMAPの最小距離（手動トピックの場合は無視される）
        use_manual_topics: Trueの場合は手動定義トピックを使用
    
    Returns:
        {min_cluster_size: {
            'topic_model': BERTopicモデル（手動トピックの場合はNone）,
            'topic_docs': {topic_id: [doc_index, ...]},
            'topic_progressions': {topic_id: [roman_progression, ...]},
            'topic_distances': {topic_id: {ref_name: distance, ...}},
            'topic_info': {topic_id: {words: [...], size: int, ...}},
            'topic_words': {topic_id: {words: [...], scores: [...], size: int}},
            'topics': [topic_id, ...] (ドキュメントごとのトピックID)
        }, ...}
    """
    results = {}
    
    if use_manual_topics:
        # 手動トピックを使用（cluster_sizesは無視、1回だけ実行）
        print(f"\n{'='*60}")
        print(f"Using manual topics")
        print(f"{'='*60}")
        
        try:
            topic_model, topic_docs, topic_progressions, topics = perform_topic_modeling(
                lyrics_list,
                metadata_list,
                use_manual_topics=True
            )
            
            # 手動トピックの場合、キーワードはトピック名を使用
            topic_words_dict = {}
            for topic_id in topic_docs.keys():
                topic_words_dict[topic_id] = {
                    'words': [MANUAL_TOPICS[topic_id]['name']],
                    'scores': [1.0],
                    'size': len(topic_docs[topic_id])
                }
            
            # 基準進行への距離を計算
            topic_distances = compute_topic_reference_distances(topic_progressions)
            
            # topic_infoは手動トピックの場合は空のDataFrameまたは辞書
            import pandas as pd
            topic_info = pd.DataFrame([{
                'Topic': topic_id,
                'Count': len(docs),
                'Name': MANUAL_TOPICS[topic_id]['name']
            } for topic_id, docs in topic_docs.items()])
            
            results[0] = {  # 手動トピックの場合は0をキーとして使用
                'topic_model': topic_model,
                'topic_docs': topic_docs,
                'topic_progressions': topic_progressions,
                'topic_distances': topic_distances,
                'topic_info': topic_info,
                'topic_words': topic_words_dict,
                'topics': topics
            }
            
            print(f"Completed: {len(topic_docs)} topics found")
            
        except Exception as e:
            print(f"Error analyzing with manual topics: {e}")
            import traceback
            traceback.print_exc()
        
        return results
    
    # BERTopicを使用
    for min_cluster_size in cluster_sizes:
        print(f"\n{'='*60}")
        print(f"Analyzing with min_cluster_size={min_cluster_size}")
        print(f"{'='*60}")
        
        try:
            topic_model, topic_docs, topic_progressions, topics = perform_topic_modeling(
                lyrics_list,
                metadata_list,
                min_cluster_size=min_cluster_size,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                use_manual_topics=False
            )
            
            # トピック情報を取得
            topic_info = topic_model.get_topic_info() if topic_model is not None else None
            
            # 各トピックのキーワードを取得
            topic_words_dict = {}
            for topic_id in topic_docs.keys():
                if topic_id == -1:
                    continue
                try:
                    if topic_model is not None:
                        topic_words_result = topic_model.get_topic(topic_id)
                        
                        words_list = []
                        scores_list = []
                        
                        if topic_words_result is not None and len(topic_words_result) > 0:
                            # BERTopicのバージョンによって、DataFrameまたはリストのタプルが返される
                            import pandas as pd
                            if isinstance(topic_words_result, pd.DataFrame):
                                # DataFrameの場合
                                words_list = topic_words_result['Word'].tolist()
                                scores_list = topic_words_result['Score'].tolist()
                            elif isinstance(topic_words_result, list):
                                # タプルのリスト [(word, score), ...] の場合
                                words_list = [item[0] for item in topic_words_result]
                                scores_list = [item[1] for item in topic_words_result]
                            else:
                                # その他の形式の場合（辞書など）
                                words_list = []
                                scores_list = []
                        
                        topic_words_dict[topic_id] = {
                            'words': words_list,
                            'scores': scores_list,
                            'size': len(topic_docs[topic_id])
                        }
                    else:
                        topic_words_dict[topic_id] = {
                            'words': [],
                            'scores': [],
                            'size': len(topic_docs[topic_id])
                        }
                except Exception as e:
                    print(f"Warning: Could not get words for topic {topic_id}: {e}")
                    topic_words_dict[topic_id] = {
                        'words': [],
                        'scores': [],
                        'size': len(topic_docs[topic_id])
                    }
            
            # 基準進行への距離を計算
            topic_distances = compute_topic_reference_distances(topic_progressions)
            
            results[min_cluster_size] = {
                'topic_model': topic_model,
                'topic_docs': topic_docs,
                'topic_progressions': topic_progressions,
                'topic_distances': topic_distances,
                'topic_info': topic_info,
                'topic_words': topic_words_dict,
                'topics': topics
            }
            
            print(f"Completed: {len(topic_docs)} topics found")
            
        except Exception as e:
            print(f"Error analyzing with min_cluster_size={min_cluster_size}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results

