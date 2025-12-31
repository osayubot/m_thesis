"""
共通関数・定数（t-SNE / UMAP可視化用 - トピックベース）
BERTopicを使用してトピック分割を行い、感情の代わりにトピック割合を表示
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# BERTopicのインポート
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from hdbscan import HDBSCAN
    from umap import UMAP
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    print("Warning: bertopic is not installed. Please install it with: pip install bertopic sentence-transformers umap-learn hdbscan")

from ..visualize_scattergraph_data.musical_distance import compute_distance_matrix, circular_distance
from ..visualize_scattergraph_data.common import (
    REFERENCE_PROGRESSIONS,
    REFERENCE_COLORS,
    load_analyzed_data,
    is_same_progression
)
from ..analyze_data.roman_numeral import section_to_roman_progression

# トピック用の色定義（自動生成されるトピックに色を割り当てる）
# トピック数が動的なため、色は動的に生成される
def generate_topic_colors(num_topics: int) -> Dict[int, str]:
    """
    トピック数に応じて色を生成
    
    Args:
        num_topics: トピック数
    
    Returns:
        {topic_id: color_hex, ...}
    """
    # 色相環から均等に色を選択
    import colorsys
    
    colors = {}
    for i in range(num_topics):
        hue = i / num_topics
        saturation = 0.7
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors[i] = hex_color
    
    return colors


def extract_chord_progressions_with_lyrics_for_topic(songs: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    コード進行と紐づく歌詞を抽出（トピック分析用）
    感情データのフィルタリングは行わず、歌詞があれば採用
    
    Returns:
        (progressions_data, songs_list)
        progressions_data: 各コード進行とその歌詞のリスト
        songs_list: 曲情報のリスト（元のsongsと同じ順序）
    """
    progressions_data = []
    songs_list = []
    
    for song in songs:
        # 曲情報を保存
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
        songs_list.append(song_info)
        
        analyzed = song.get('analyzed_chord_progressions_and_lyrics', [])
        
        for section in analyzed:
            chord_prog = section.get('normalized_chord_progression', [])
            if not chord_prog or chord_prog == ['N.C']:
                continue
            
            # ローマ数字に変換
            key = section.get('key')
            if not key:
                continue
            
            # normalized_chord_progressionを使用してローマ数字に変換
            section_for_roman = section.copy()
            section_for_roman['chord_progression'] = chord_prog
            roman_prog = section_to_roman_progression(section_for_roman, key)
            if not roman_prog:
                continue
            
            # フィルタリング条件1: コード進行は4つの連なりのもののみを採用
            if len(roman_prog) != 4:
                continue
            
            # 歌詞を取得（歌詞がないセクションはスキップ）
            lyric = section.get('lyric', '').strip()
            
            # 歌詞がない場合はスキップ
            if not lyric:
                continue
            
            # 同じコード進行をグループ化
            prog_key = tuple(roman_prog)
            found = False
            for prog_data in progressions_data:
                existing_prog = tuple(prog_data['roman_progression'])
                if existing_prog == prog_key:
                    # 既存のコード進行に歌詞を追加
                    prog_data['lyrics'].append({
                        'lyric': lyric,
                        'song_index': len(songs_list) - 1,
                        'typical_chord_distance': section.get('typical_chord_distance', {})
                    })
                    # typical_chord_distanceを更新（最小値を保持）
                    tcd = section.get('typical_chord_distance', {})
                    if tcd:
                        existing_tcd = prog_data.get('typical_chord_distance', {})
                        if not existing_tcd:
                            prog_data['typical_chord_distance'] = tcd.copy()
                        else:
                            for ref_name in ['odo', 'komuro', 'marusa']:
                                if ref_name in tcd and (ref_name not in existing_tcd or tcd[ref_name] < existing_tcd[ref_name]):
                                    existing_tcd[ref_name] = tcd[ref_name]
                    found = True
                    break
            
            if not found:
                # 新しいコード進行を追加
                lyrics_list = [{
                    'lyric': lyric,
                    'song_index': len(songs_list) - 1
                }]
                
                progressions_data.append({
                    'chord_progression': section.get('chord_progression', []),
                    'normalized_chord_progression': chord_prog,
                    'roman_progression': list(prog_key),
                    'lyrics': lyrics_list,
                    'key': key,
                })
    
    return progressions_data, songs_list


def add_reference_progressions(progressions_data: List[Dict], 
                                reference_progressions: Optional[Dict[str, List[str]]] = None) -> List[Dict]:
    """
    基準進行を空のデータポイントとして追加する
    
    この関数は、REFERENCE_PROGRESSIONSで定義された基準進行を、
    データに存在しない場合でも空のデータポイントとして追加します。
    追加された基準進行は、t-SNE/UMAPの座標計算に含まれ、
    散布図上で星（⭐）として表示されます。
    
    Args:
        progressions_data: 既存のコード進行データ
        reference_progressions: 基準進行の辞書（Noneの場合はREFERENCE_PROGRESSIONSを使用）
    
    Returns:
        基準進行を追加したコード進行データ（既に存在する場合はスキップ）
    """
    if reference_progressions is None:
        reference_progressions = REFERENCE_PROGRESSIONS
    
    # 各基準進行を処理
    for ref_name, ref_prog in reference_progressions.items():
        # 既存のデータに基準進行と一致するものがあるかチェック（循環性を考慮）
        found_existing = False
        for prog_data in progressions_data:
            roman_prog = prog_data.get('roman_progression', [])
            if roman_prog and is_same_progression(roman_prog, ref_prog):
                # 既存のデータに基準進行フラグを設定
                prog_data['isReferenceProgression'] = True
                prog_data['referenceProgressionName'] = ref_name
                # typical_chord_distanceも更新（自分自身なので距離0）
                if 'typical_chord_distance' not in prog_data:
                    prog_data['typical_chord_distance'] = {}
                prog_data['typical_chord_distance'][ref_name] = 0.0
                print(f"  Reference progression {ref_name} already exists in data, marking as reference...")
                found_existing = True
                break
        
        # 既に存在する場合はスキップ
        if found_existing:
            continue
        
        # 空のデータとして追加
        reference_prog_data = {
            'chord_progression': [],  # 空
            'normalized_chord_progression': [],  # 空
            'roman_progression': list(ref_prog),
            'lyrics': [],  # 空の歌詞リスト
            'typical_chord_distance': {ref_name: 0.0},  # 自分自身なので距離0
            'isReferenceProgression': True,
            'referenceProgressionName': ref_name
        }
        
        progressions_data.append(reference_prog_data)
        print(f"  Added reference progression {ref_name}: {' - '.join(ref_prog)}")
    
    return progressions_data


def perform_topic_modeling_on_lyrics(
    progressions_data: List[Dict],
    min_cluster_size: int = 30,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 5
) -> Tuple[Optional[object], Dict[int, List[int]], Dict[int, str], List[int]]:
    """
    各コード進行に属する歌詞に対してBERTopicでトピックモデリングを実行
    
    Args:
        progressions_data: コード進行データ（lyricsフィールドに歌詞が含まれる）
        min_cluster_size: HDBSCANの最小クラスタサイズ
        n_neighbors: UMAPの近傍数
        min_dist: UMAPの最小距離
        n_components: UMAPの次元数
    
    Returns:
        (topic_model, topic_to_lyric_indices, topic_names, topics)
        topic_model: BERTopicモデル
        topic_to_lyric_indices: {topic_id: [lyric_index_in_progression, ...]} 各トピックに属する歌詞のインデックス（コード進行内）
        topic_names: {topic_id: topic_name} 各トピックの名前
        topics: [topic_id, ...] 各歌詞に対応するトピックIDのリスト（全歌詞）
    """
    if not BERTOPIC_AVAILABLE:
        raise ImportError("bertopic is not installed. Please install it with: pip install bertopic sentence-transformers umap-learn hdbscan")
    
    # 全歌詞を収集
    all_lyrics = []
    lyric_to_progression_index = []  # 各歌詞がどのコード進行に属するか
    
    for prog_idx, prog_data in enumerate(progressions_data):
        lyrics = prog_data.get('lyrics', [])
        for lyric_data in lyrics:
            lyric_text = lyric_data.get('lyric', '').strip()
            if lyric_text:
                all_lyrics.append(lyric_text)
                lyric_to_progression_index.append(prog_idx)
    
    if len(all_lyrics) == 0:
        return None, {}, {}, []
    
    # データサイズに応じてmin_cluster_sizeとmin_samplesを調整
    # 200曲程度の場合は、より小さな値を使用
    num_lyrics = len(all_lyrics)
    
    # min_cluster_sizeをデータサイズに応じて調整
    # デフォルトの30が大きすぎる場合は、データサイズの1-2%程度を目安にする
    if min_cluster_size > num_lyrics * 0.02:
        # データサイズが小さい場合は、より小さな値を使用
        adjusted_min_cluster_size = max(5, int(num_lyrics * 0.01))
        print(f"Warning: min_cluster_size={min_cluster_size} is too large for {num_lyrics} lyrics. Adjusting to {adjusted_min_cluster_size}")
        min_cluster_size = adjusted_min_cluster_size
    
    # min_samplesも調整（min_cluster_sizeの1/3程度）
    min_samples = max(3, min_cluster_size // 3)
    
    print(f"Performing topic modeling on {len(all_lyrics)} lyrics...")
    print(f"Parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}, n_neighbors={n_neighbors}")
    
    # 日本語用のSentenceTransformerモデルを使用
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # UMAPモデル
    umap_model = UMAP(
        n_neighbors=min(n_neighbors, len(all_lyrics) - 1),
        n_components=n_components,
        min_dist=min_dist,
        metric='cosine',
        random_state=42
    )
    
    # HDBSCANモデル（K指定不要、自動でクラスタ数を決定）
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    
    # BERTopicモデル
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        language='japanese',
        verbose=True
    )
    
    # トピックモデリングを実行
    topics, probs = topic_model.fit_transform(all_lyrics)
    
    # トピック数を確認（-1はノイズ）
    unique_topics = set(topics)
    if -1 in unique_topics:
        unique_topics.remove(-1)
    num_topics = len(unique_topics)
    
    print(f"Found {num_topics} topics (excluding noise)")
    
    # 各トピックに属する歌詞のインデックスを集計
    topic_to_lyric_indices = defaultdict(list)
    
    for lyric_idx, topic_id in enumerate(topics):
        if topic_id != -1:  # ノイズは除外
            topic_to_lyric_indices[topic_id].append(lyric_idx)
    
    # 各トピックの名前を取得
    topic_names = {}
    topic_colors = generate_topic_colors(num_topics)
    
    for topic_id in unique_topics:
        try:
            # トピックのキーワードを取得
            topic_info = topic_model.get_topic(topic_id)
            if topic_info and len(topic_info) > 0:
                # 上位3語を結合してトピック名とする
                if isinstance(topic_info, list):
                    words = [item[0] for item in topic_info[:3]]
                else:
                    # DataFrameの場合
                    words = topic_info['Word'].tolist()[:3]
                topic_name = ' / '.join(words)
            else:
                topic_name = f"Topic {topic_id}"
        except:
            topic_name = f"Topic {topic_id}"
        
        topic_names[topic_id] = topic_name
    
    return topic_model, dict(topic_to_lyric_indices), topic_names, topics.tolist() if hasattr(topics, 'tolist') else list(topics)


def compute_distance_vectors(
    progressions_data: List[Dict],
    reference_progressions: Dict[str, List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    各コード進行から基準進行への距離ベクトルを計算
    
    Args:
        progressions_data: コード進行データ
        reference_progressions: 基準進行の辞書（Noneの場合はデフォルトを使用）
    
    Returns:
        (距離ベクトルの配列（n×m、n=コード進行数、m=基準進行数）, 基準進行名のリスト)
    """
    if reference_progressions is None:
        reference_progressions = REFERENCE_PROGRESSIONS
    
    # ローマ数字のコード進行を取得
    roman_progressions = [prog['roman_progression'] for prog in progressions_data]
    
    # 基準進行のリスト（順序を保持）
    ref_names = list(reference_progressions.keys())
    ref_progs = [reference_progressions[name] for name in ref_names]
    
    print(f"Computing distance vectors to {len(ref_names)} reference progressions...")
    print(f"Reference progressions: {ref_names}")
    
    # 各コード進行について、各基準進行への距離を計算
    distance_vectors = []
    for i, prog in enumerate(roman_progressions):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(roman_progressions)}...")
        
        vector = []
        for ref_prog in ref_progs:
            dist = circular_distance(prog, ref_prog)
            vector.append(dist)
        distance_vectors.append(vector)
    
    return np.array(distance_vectors), ref_names


def export_to_json_format_with_topics(
    progressions_data: List[Dict],
    coordinates: np.ndarray,
    songs_list: List[Dict],
    topic_model: object,
    topic_to_lyric_indices: Dict[int, List[int]],
    topic_names: Dict[int, str],
    all_topics: List[int],
    lyric_to_progression_index: List[int],
    all_lyrics_list: List[str],
    output_path: str,
    reference_progressions: Optional[Dict[str, List[str]]] = None
) -> Dict:
    """
    トピック割合の円グラフを含むJSON形式でデータを出力
    
    Args:
        progressions_data: コード進行データ
        coordinates: 座標配列
        songs_list: 曲情報のリスト
        topic_model: BERTopicモデル
        topic_to_lyric_indices: {topic_id: [lyric_index, ...]} 全歌詞中のインデックス
        topic_names: {topic_id: topic_name}
        all_topics: [topic_id, ...] 全歌詞のトピックID
        lyric_to_progression_index: [prog_index, ...] 各歌詞がどのコード進行に属するか
        output_path: 出力パス
        reference_progressions: 基準進行の辞書
    
    Returns:
        JSON形式のデータ
    """
    if reference_progressions is None:
        reference_progressions = REFERENCE_PROGRESSIONS
    
    # 座標の範囲を取得（0-100に正規化するため）
    if len(coordinates) == 0:
        return {"points": [], "topicSongs": {}}
    
    x_coords = coordinates[:, 0]
    y_coords = coordinates[:, 1]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    x_range = x_max - x_min if x_max > x_min else 1
    y_range = y_max - y_min if y_max > y_min else 1
    
    # 原点中心の場合、範囲を対称にする
    if abs(x_min + x_max) < 0.01 and abs(y_min + y_max) < 0.01:
        x_abs_max = max(abs(x_min), abs(x_max))
        y_abs_max = max(abs(y_min), abs(y_max))
        x_range = x_abs_max * 2 if x_abs_max > 0 else 1
        y_range = y_abs_max * 2 if y_abs_max > 0 else 1
        x_min = -x_abs_max
        y_min = -y_abs_max
    
    # トピックごとの曲リストを構築
    topic_songs = defaultdict(list)
    
    # トピックの色を生成
    unique_topics = set(all_topics)
    if -1 in unique_topics:
        unique_topics.remove(-1)
    topic_colors = generate_topic_colors(len(unique_topics))
    topic_id_to_color = {tid: topic_colors.get(i, '#808080') for i, tid in enumerate(sorted(unique_topics))}
    
    points = []
    
    # 各コード進行ごとにトピック割合を計算
    for prog_idx, (prog_data, coord) in enumerate(zip(progressions_data, coordinates)):
        lyrics = prog_data.get('lyrics', [])
        is_reference = prog_data.get('isReferenceProgression', False)
        
        # 基準進行の場合は空の歌詞でも処理する
        if not lyrics and not is_reference:
            continue
        
        # 座標を0-100に正規化
        x_norm = ((coord[0] - x_min) / x_range) * 100
        y_norm = ((coord[1] - y_min) / y_range) * 100
        
        # 基準進行の場合は空のpieデータを作成
        if is_reference:
            ref_name = prog_data.get('referenceProgressionName', '')
            roman_progression = prog_data.get('roman_progression', [])
            progression_str = ' - '.join(roman_progression) if roman_progression else 'N/A'
            
            point_data = {
                'x': float(round(x_norm, 2)),
                'y': float(round(y_norm, 2)),
                'r': float(round(12.0, 2)),
                'pie': [],
                'lyricCount': 0,
                'progression': progression_str,
                'roman_progression': roman_progression,
                'chord_progression': prog_data.get('chord_progression', []),
                'isReferenceProgression': True,
                'referenceProgressionName': ref_name,
                'progressionType': ref_name
            }
            if ref_name:
                point_data['strokeColor'] = REFERENCE_COLORS.get(ref_name, "#000000")
                point_data['referenceName'] = ref_name
            points.append(point_data)
            continue
        
        # このコード進行に属する歌詞のトピックを集計
        # lyric_to_progression_indexから、このコード進行に属する歌詞のインデックスを取得
        prog_lyric_indices = []
        for lyric_idx, prog_idx_in_list in enumerate(lyric_to_progression_index):
            if prog_idx_in_list == prog_idx:
                prog_lyric_indices.append(lyric_idx)
        
        if not prog_lyric_indices:
            continue
        
        # 各トピックに属する歌詞数を集計
        topic_counts = defaultdict(int)
        topic_lyrics_map = defaultdict(list)  # {topic_id: [{'lyric': lyric_text, 'song_index': song_index}, ...]}
        topic_song_indices = defaultdict(set)
        
        for lyric_idx in prog_lyric_indices:
            if lyric_idx >= len(all_topics) or lyric_idx >= len(all_lyrics_list):
                continue
            
            topic_id = all_topics[lyric_idx]
            if topic_id == -1:  # ノイズは除外
                continue
            
            # 全歌詞リストから直接取得
            lyric_text = all_lyrics_list[lyric_idx].strip()
            if not lyric_text:
                continue
            
            # このコード進行のlyricsリストからsong_indexを取得
            # lyricsリストの順序とlyric_to_progression_indexの順序は一致しているはず
            song_index = None
            relative_idx = 0
            for i, prog_idx_check in enumerate(lyric_to_progression_index):
                if prog_idx_check == prog_idx:
                    if i == lyric_idx:
                        if relative_idx < len(lyrics):
                            song_index = lyrics[relative_idx].get('song_index')
                        break
                    relative_idx += 1
            
            if lyric_text:
                topic_counts[topic_id] += 1
                topic_lyrics_map[topic_id].append({
                    'lyric': lyric_text,
                    'song_index': song_index
                })
                if song_index is not None:
                    topic_song_indices[topic_id].add(song_index)
        
        if not topic_counts:
            continue
        
        # ユニークな歌詞数を計算（重複を除去）
        unique_lyrics = set()
        for lyric_list in topic_lyrics_map.values():
            for lyric_item in lyric_list:
                unique_lyrics.add(lyric_item['lyric'])
        total_unique_lyrics = len(unique_lyrics)
        
        # 各トピックごとのユニークな歌詞数を計算
        topic_unique_counts = {}
        for topic_id in topic_counts.keys():
            topic_unique_set = set()
            for lyric_item in topic_lyrics_map.get(topic_id, []):
                topic_unique_set.add(lyric_item['lyric'])
            topic_unique_counts[topic_id] = len(topic_unique_set)
        
        # pieデータを作成（%で合計100になるように）
        pie_data = []
        for topic_id in sorted(topic_counts.keys()):
            unique_count = topic_unique_counts.get(topic_id, 0)
            if unique_count > 0 and total_unique_lyrics > 0:
                percentage = (unique_count / total_unique_lyrics) * 100.0
                topic_name = topic_names.get(topic_id, f"Topic {topic_id}")
                topic_color = topic_id_to_color.get(topic_id, '#808080')
                
                # 歌詞と曲情報を含むリスト（最大10個まで）
                lyrics_with_songs = topic_lyrics_map[topic_id][:10]
                lyrics_list = []
                for lyric_item in lyrics_with_songs:
                    lyric_entry = {
                        'lyric': lyric_item['lyric'],
                        'song_index': lyric_item.get('song_index')
                    }
                    if lyric_item.get('song_index') is not None and lyric_item['song_index'] < len(songs_list):
                        song_info = songs_list[lyric_item['song_index']]
                        lyric_entry['song'] = {
                            'title': song_info.get('title', ''),
                            'artist': song_info.get('artist', ''),
                            'composer': song_info.get('composer', ''),
                            'lyricist': song_info.get('lyricist', ''),
                            'release_date': song_info.get('release_date', ''),
                            'spotify_popularity': song_info.get('spotify_popularity', None)
                        }
                    lyrics_list.append(lyric_entry)
                
                pie_data.append({
                    'label': topic_name,
                    'c': topic_color,
                    'v': float(round(percentage, 2)),
                    'lyrics': lyrics_list
                })
                
                # このトピックに対応する曲を追加
                for song_index in topic_song_indices[topic_id]:
                    if song_index < len(songs_list):
                        song_info = songs_list[song_index]
                        is_duplicate = any(
                            s.get('title') == song_info.get('title') and 
                            s.get('artist') == song_info.get('artist')
                            for s in topic_songs[topic_id]
                        )
                        if not is_duplicate:
                            topic_songs[topic_id].append(song_info)
        
        # 合計が100になるように調整（丸め誤差対策）
        if pie_data:
            total_pie = sum(p['v'] for p in pie_data)
            if abs(total_pie - 100) > 0.01:
                diff = 100 - total_pie
                if pie_data:
                    new_value = pie_data[-1]['v'] + diff
                    pie_data[-1]['v'] = float(round(max(0.0, new_value), 2))
                    if new_value < 0:
                        negative_amount = abs(new_value)
                        pie_data[-1]['v'] = 0.0
                        positive_items = [p for p in pie_data[:-1] if p['v'] > 0]
                        if positive_items:
                            per_item_reduction = negative_amount / len(positive_items)
                            for item in positive_items:
                                item['v'] = float(round(max(0.0, item['v'] - per_item_reduction), 2))
        
        # 半径を計算
        lyric_count = total_unique_lyrics
        min_r = 4
        max_r = 20
        if lyric_count <= 1:
            r = min_r
        else:
            max_lyric_count = 100
            log_scale = np.log(lyric_count) / np.log(max_lyric_count)
            r = min_r + (max_r - min_r) * min(1.0, log_scale)
        
        # コード進行を取得
        roman_progression = prog_data.get('roman_progression', [])
        chord_progression = prog_data.get('chord_progression', [])
        
        if roman_progression:
            progression_str = ' - '.join(roman_progression)
        elif chord_progression:
            progression_str = ' - '.join(chord_progression)
        else:
            progression_str = 'N/A'
        
        # 基準進行との距離を判定
        stroke_color = None
        matched_reference_name = None
        progression_type = None
        
        tcd = prog_data.get('typical_chord_distance', {})
        if tcd:
            min_distance = float('inf')
            closest_ref = None
            for ref_name in ['odo', 'komuro', 'marusa']:
                if ref_name in tcd:
                    distance = tcd[ref_name]
                    if distance < min_distance:
                        min_distance = distance
                        closest_ref = ref_name
            if closest_ref:
                progression_type = closest_ref
                stroke_color = REFERENCE_COLORS.get(closest_ref, "#000000")
                matched_reference_name = closest_ref
        
        # 完全一致もチェック
        is_reference_progression = False
        reference_progression_name = None
        if reference_progressions and roman_progression:
            for ref_name, ref_prog in reference_progressions.items():
                if is_same_progression(roman_progression, ref_prog):
                    stroke_color = REFERENCE_COLORS.get(ref_name, "#000000")
                    matched_reference_name = ref_name
                    progression_type = ref_name
                    is_reference_progression = True
                    reference_progression_name = ref_name
                    break
        
        point_data = {
            'x': float(round(x_norm, 2)),
            'y': float(round(y_norm, 2)),
            'r': float(round(r, 2)),
            'pie': pie_data,
            'lyricCount': lyric_count,
            'progression': progression_str,
            'roman_progression': roman_progression,
            'chord_progression': chord_progression
        }
        
        if stroke_color:
            point_data['strokeColor'] = stroke_color
        if matched_reference_name:
            point_data['referenceName'] = matched_reference_name
        if progression_type:
            point_data['progressionType'] = progression_type
        if is_reference_progression:
            point_data['isReferenceProgression'] = True
            point_data['referenceProgressionName'] = reference_progression_name
        
        points.append(point_data)
    
    # トピックごとの曲リストを辞書形式に変換
    topic_songs_dict = {}
    for topic_id in sorted(unique_topics):
        topic_name = topic_names.get(topic_id, f"Topic {topic_id}")
        topic_songs_dict[topic_name] = topic_songs.get(topic_id, [])
    
    result = {
        'points': points,
        'topicSongs': topic_songs_dict
    }
    
    # JSONファイルに保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Exported JSON to {output_path}")
    return result

