"""
トピック分析結果を可視化用JSON形式に変換
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from ..visualize_scattergraph_data.common import REFERENCE_PROGRESSIONS, REFERENCE_COLORS
from ..visualize_scattergraph_data.musical_distance import circular_distance, compute_distance_matrix
from sklearn.manifold import MDS
try:
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


def cluster_progressions_by_distance(
    all_progressions: List[List[str]],
    min_cluster_size: int = 20,
    reference_progressions: Dict[str, List[str]] = None
) -> Tuple[Dict[int, List[int]], List[List[str]], Dict[int, List[int]]]:
    """
    コード進行を距離行列に基づいてクラスタリング
    
    Args:
        all_progressions: 全てのコード進行のリスト（ローマ数字）
        min_cluster_size: HDBSCANの最小クラスタサイズ
        reference_progressions: 基準進行の辞書（オプション）
    
    Returns:
        (cluster_to_indices, unique_progressions, cluster_to_progressions)
        cluster_to_indices: {cluster_id: [progression_index, ...]}
        unique_progressions: ユニークなコード進行のリスト
        cluster_to_progressions: {cluster_id: [progression_index_in_unique, ...]}
    """
    if not HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan is required for clustering. Install it with: pip install hdbscan")
    
    if reference_progressions is None:
        reference_progressions = REFERENCE_PROGRESSIONS
    
    # ユニークなコード進行を抽出（重複を除去）
    unique_progressions = []
    progression_to_index = {}
    original_to_unique = []
    
    for prog in all_progressions:
        prog_tuple = tuple(prog)
        if prog_tuple not in progression_to_index:
            progression_to_index[prog_tuple] = len(unique_progressions)
            unique_progressions.append(list(prog))
        original_to_unique.append(progression_to_index[prog_tuple])
    
    print(f"Total progressions: {len(all_progressions)}, Unique: {len(unique_progressions)}")
    
    if len(unique_progressions) < 2:
        return {}, unique_progressions, {}
    
    # 距離行列を計算
    print(f"Computing distance matrix for {len(unique_progressions)} unique progressions...")
    dist_matrix = compute_distance_matrix(unique_progressions, show_progress=True)
    
    # HDBSCANでクラスタリング
    print(f"Clustering progressions with HDBSCAN (min_cluster_size={min_cluster_size})...")
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=10,
        metric='precomputed',
        cluster_selection_method='eom'
    )
    cluster_labels = clusterer.fit_predict(dist_matrix)
    
    # クラスタごとにインデックスをグループ化
    cluster_to_progressions = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_labels):
        if cluster_id != -1:  # ノイズは除外
            cluster_to_progressions[cluster_id].append(idx)
    
    # 元のインデックス（重複含む）に戻す
    cluster_to_indices = defaultdict(list)
    for orig_idx, unique_idx in enumerate(original_to_unique):
        cluster_id = cluster_labels[unique_idx]
        if cluster_id != -1:
            cluster_to_indices[cluster_id].append(orig_idx)
    
    print(f"Found {len(cluster_to_progressions)} clusters (excluding noise)")
    
    return dict(cluster_to_indices), unique_progressions, dict(cluster_to_progressions)


def compute_topic_coordinates_from_progressions(
    topic_progressions: Dict[int, List[List[str]]],
    reference_progressions: Dict[str, List[str]] = None,
    use_clustering: bool = True,
    min_cluster_size: int = 20
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    コード進行の類似性に基づいてクラスタリングし、MDSで座標を計算
    
    Args:
        topic_progressions: {topic_id: [[roman_progression, ...], ...]} 各トピックに属するコード進行のリスト
        reference_progressions: 基準進行の辞書
        use_clustering: Trueの場合はコード進行でクラスタリング、Falseの場合はBERTopicのトピックを使用
        min_cluster_size: クラスタリングの最小クラスタサイズ
    
    Returns:
        (座標配列（n×2、n=クラスタ数+基準進行数）, 基準進行名のリスト, クラスタIDのリスト)
    """
    if reference_progressions is None:
        reference_progressions = REFERENCE_PROGRESSIONS
    
    ref_names = list(reference_progressions.keys())
    
    if use_clustering:
        # 全てのコード進行を収集
        all_progressions = []
        for topic_id, progressions in topic_progressions.items():
            if topic_id != -1:
                all_progressions.extend(progressions)
        
        if len(all_progressions) == 0:
            coordinates = np.zeros((len(ref_names), 2))
            return coordinates, ref_names, []
        
        # コード進行でクラスタリング
        cluster_to_indices, unique_progressions, cluster_to_unique = cluster_progressions_by_distance(
            all_progressions, min_cluster_size, reference_progressions
        )
        
        if len(cluster_to_unique) == 0:
            coordinates = np.zeros((len(ref_names), 2))
            return coordinates, ref_names, []
        
        cluster_ids = sorted(cluster_to_unique.keys())
        
        # 各クラスタの代表コード進行を選択（最も頻出するコード進行）
        from collections import Counter
        cluster_representative_progressions = []
        for cluster_id in cluster_ids:
            unique_indices = cluster_to_unique[cluster_id]
            cluster_progs = [unique_progressions[idx] for idx in unique_indices]
            prog_counter = Counter([tuple(prog) for prog in cluster_progs])
            most_common_prog_tuple = prog_counter.most_common(1)[0][0]
            cluster_representative_progressions.append(list(most_common_prog_tuple))
    else:
        # 元の方法：BERTopicのトピックを使用
        topic_ids = sorted([tid for tid in topic_progressions.keys() if tid != -1])
        cluster_representative_progressions = []
        cluster_ids = topic_ids
        
        for topic_id in topic_ids:
            progressions = topic_progressions[topic_id]
            if progressions:
                from collections import Counter
                prog_counter = Counter([tuple(prog) for prog in progressions])
                most_common_prog_tuple = prog_counter.most_common(1)[0][0]
                cluster_representative_progressions.append(list(most_common_prog_tuple))
            else:
                cluster_representative_progressions.append([])
    
    # 基準進行のコード進行を追加
    all_progressions_for_mds = list(reference_progressions.values()) + cluster_representative_progressions
    
    # 有効なコード進行のみをフィルタリング
    valid_progressions = []
    valid_indices = []
    for i, prog in enumerate(all_progressions_for_mds):
        if prog:  # 空でない場合
            valid_progressions.append(prog)
            valid_indices.append(i)
    
    if len(valid_progressions) < 2:
        # データが不十分な場合は、デフォルトの座標を返す
        n_total = len(ref_names) + len(cluster_ids)
        coordinates = np.zeros((n_total, 2))
        return coordinates, ref_names, cluster_ids
    
    # 距離行列を計算
    print(f"Computing distance matrix for {len(valid_progressions)} progressions (including {len(ref_names)} references)...")
    dist_matrix = compute_distance_matrix(valid_progressions, show_progress=True)
    
    # MDSで座標を計算
    print("Computing MDS coordinates from distance matrix...")
    mds = MDS(
        n_components=2,
        dissimilarity='precomputed',
        random_state=42,
        n_init=10,
        max_iter=1000
    )
    coordinates_mds = mds.fit_transform(dist_matrix)
    
    # 座標を元のサイズに戻す（無効なコード進行には0を設定）
    n_total = len(ref_names) + len(cluster_ids)
    coordinates = np.zeros((n_total, 2))
    
    coord_idx = 0
    for i in range(len(all_progressions_for_mds)):
        if i in valid_indices:
            coordinates[i] = coordinates_mds[coord_idx]
            coord_idx += 1
    
    return coordinates, ref_names, cluster_ids


def compute_topic_coordinates_from_distances(
    topic_distances: Dict[int, Dict[str, float]],
    reference_progressions: Dict[str, List[str]] = None
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    トピックと基準進行の距離から座標を計算（三辺測量法を使用）
    基準進行を固定点として、各トピックの座標を距離から直接計算
    
    Args:
        topic_distances: {topic_id: {ref_name: distance, ...}}
        reference_progressions: 基準進行の辞書
    
    Returns:
        座標配列（n×2、n=トピック数+基準進行数）
    """
    if reference_progressions is None:
        reference_progressions = REFERENCE_PROGRESSIONS
    
    ref_names = list(reference_progressions.keys())
    topic_ids = sorted([tid for tid in topic_distances.keys() if tid != -1])
    
    # 基準進行を固定点として配置（三角形の頂点に配置）
    n_total = len(ref_names) + len(topic_ids)
    coordinates = np.zeros((n_total, 2))
    
    # 基準進行間の距離を計算
    ref_distances = {}
    for i, ref_name_i in enumerate(ref_names):
        for j, ref_name_j in enumerate(ref_names):
            if i < j:
                ref_prog_i = reference_progressions[ref_name_i]
                ref_prog_j = reference_progressions[ref_name_j]
                dist = circular_distance(ref_prog_i, ref_prog_j)
                ref_distances[(ref_name_i, ref_name_j)] = dist
                ref_distances[(ref_name_j, ref_name_i)] = dist
    
    # 基準進行の座標を設定（距離関係を考慮して配置）
    # まず距離のスケールを決定
    if len(ref_names) >= 2:
        # 基準進行間の距離の最大値を取得
        max_ref_dist = max([ref_distances.get((ref_names[i], ref_names[j]), 0.0) 
                           for i in range(len(ref_names)) 
                           for j in range(i+1, len(ref_names))] or [10.0])
        # 最大距離を30にスケール（基準進行間の配置用）
        ref_scale = 30.0 / max_ref_dist if max_ref_dist > 0 else 1.0
    else:
        ref_scale = 1.0
    
    if len(ref_names) == 3:
        # 3つの基準進行を距離に基づいて三角形に配置
        d12 = ref_distances.get((ref_names[0], ref_names[1]), 10.0) * ref_scale
        d13 = ref_distances.get((ref_names[0], ref_names[2]), 10.0) * ref_scale
        d23 = ref_distances.get((ref_names[1], ref_names[2]), 10.0) * ref_scale
        
        # 最初の2つを水平に配置（中央付近）
        ref_coords = np.zeros((3, 2))
        center_x, center_y = 50.0, 50.0  # 中心座標
        base_size = max(d12, d13, d23) * 0.5  # 基準サイズ
        
        ref_coords[0] = np.array([center_x - base_size, center_y])
        ref_coords[1] = np.array([center_x + base_size, center_y])
        
        # 3つ目を三辺測量法で配置
        if d12 > 0:
            a = (d13**2 - d23**2 + d12**2) / (2 * d12)
            h_sq = d13**2 - a**2
            
            # hが小さすぎる場合（一直線に近い場合）や、aが負の場合
            # 正三角形に近い配置にする
            if h_sq < 0 or h_sq < (d12 * 0.2)**2 or a < 0:
                # 正三角形の高さを使用
                h = d12 * np.sqrt(3) / 2
                # 中心点から上方向に配置
                mid_point = (ref_coords[0] + ref_coords[1]) / 2
                ref_coords[2] = mid_point + np.array([0, h])
            else:
                h = np.sqrt(h_sq)
                mid_point = ref_coords[0] + np.array([a, 0])
                # 上方向に配置（hが正の場合）
                ref_coords[2] = mid_point + np.array([0, abs(h)])
        else:
            ref_coords[2] = ref_coords[0] + np.array([0, d13])
    elif len(ref_names) == 2:
        # 2つの基準進行を距離に基づいて配置
        d12 = ref_distances.get((ref_names[0], ref_names[1]), 10.0) * ref_scale
        ref_coords = np.array([
            [35.0, 50.0],
            [35.0 + d12, 50.0],
        ])
    else:
        # 1つの場合やそれ以外は等間隔に配置
        angle_step = 2 * np.pi / len(ref_names)
        ref_coords = np.array([
            [50.0 + 30.0 * np.cos(i * angle_step),
             50.0 + 30.0 * np.sin(i * angle_step)]
            for i in range(len(ref_names))
        ])
    
    # 基準進行の座標を設定
    for i, ref_name in enumerate(ref_names):
        coordinates[i] = ref_coords[i]
    
    # トピック距離のスケールも決定（基準進行間距離のスケールと一致させる）
    # トピックから基準進行への距離の範囲を確認
    topic_dist_values = []
    for topic_id in topic_ids:
        distances = topic_distances[topic_id]
        for ref_name in ref_names:
            dist = distances.get(ref_name, float('inf'))
            if dist != float('inf') and not np.isnan(dist) and dist >= 0:
                topic_dist_values.append(dist)
    
    if topic_dist_values:
        # トピック距離の代表値（中央値または平均）を使用してスケールを決定
        # 基準進行間距離と同じスケールを使用
        topic_scale = ref_scale
    else:
        topic_scale = ref_scale
    
    # 各トピックの座標を距離から計算（三辺測量法、スケールを考慮）
    for i, topic_id in enumerate(topic_ids):
        topic_idx = len(ref_names) + i
        distances = topic_distances[topic_id]
        
        # 各基準進行への距離を取得（スケール適用）
        dists_to_refs = []
        valid_ref_indices = []
        for j, ref_name in enumerate(ref_names):
            dist = distances.get(ref_name, float('inf'))
            if dist != float('inf') and not np.isnan(dist) and dist >= 0:
                dists_to_refs.append(dist * topic_scale)  # スケール適用
                valid_ref_indices.append(j)
        
        if len(valid_ref_indices) >= 2:
            # 少なくとも2つの基準進行への距離がある場合、三辺測量法で座標を計算
            ref1_idx = valid_ref_indices[0]
            ref2_idx = valid_ref_indices[1]
            ref1_pos = coordinates[ref1_idx]
            ref2_pos = coordinates[ref2_idx]
            d1 = dists_to_refs[0]
            d2 = dists_to_refs[1]
            
            # 2点間の距離
            ref_dist = np.linalg.norm(ref2_pos - ref1_pos)
            
            if ref_dist > 1e-6:  # ほぼ0でない場合
                # 2円の交点を計算
                a = (d1**2 - d2**2 + ref_dist**2) / (2 * ref_dist)
                h_sq = d1**2 - a**2
                if h_sq < 0:
                    h_sq = 0
                h = np.sqrt(h_sq)
                
                # 中間点からの方向ベクトル
                ref_vec = (ref2_pos - ref1_pos) / ref_dist
                mid_point = ref1_pos + a * ref_vec
                # 垂直ベクトル（時計回りに90度回転）
                perpendicular = np.array([-ref_vec[1], ref_vec[0]])
                
                # 2つの可能な位置
                pos1 = mid_point + h * perpendicular
                pos2 = mid_point - h * perpendicular
                
                # 3つ目の基準進行がある場合、それに近い方を選択
                if len(valid_ref_indices) >= 3:
                    ref3_idx = valid_ref_indices[2]
                    ref3_pos = coordinates[ref3_idx]
                    d3 = dists_to_refs[2]
                    expected_dist = d3
                    
                    dist1_to_ref3 = np.linalg.norm(pos1 - ref3_pos)
                    dist2_to_ref3 = np.linalg.norm(pos2 - ref3_pos)
                    
                    if abs(dist1_to_ref3 - expected_dist) < abs(dist2_to_ref3 - expected_dist):
                        coordinates[topic_idx] = pos1
                    else:
                        coordinates[topic_idx] = pos2
                else:
                    # デフォルトで上側（yが大きい方）を選択
                    if pos1[1] > pos2[1]:
                        coordinates[topic_idx] = pos1
                    else:
                        coordinates[topic_idx] = pos2
            else:
                # 基準進行が同じ位置にある場合、その位置に配置
                coordinates[topic_idx] = ref1_pos
        elif len(valid_ref_indices) == 1:
            # 1つの基準進行への距離のみの場合
            ref_idx = valid_ref_indices[0]
            ref_pos = coordinates[ref_idx]
            d = dists_to_refs[0]
            # 基準進行から距離dの位置に配置（右上方向）
            angle = np.pi / 4  # 45度の方向
            coordinates[topic_idx] = ref_pos + d * np.array([np.cos(angle), np.sin(angle)])
        else:
            # 有効な距離がない場合、中心に配置
            coordinates[topic_idx] = np.array([50.0, 50.0])
    
    return coordinates, ref_names, topic_ids


def generate_json_data(
    analysis_results: Dict[int, Dict],
    lyrics_list: List[str],
    metadata_list: List[Dict],
    output_path: str,
    reference_progressions: Dict[str, List[str]] = None
) -> None:
    """
    分析結果を可視化用JSON形式に変換して保存
    
    Args:
        analysis_results: analyze_topics_for_multiple_cluster_sizesの結果
        lyrics_list: 歌詞テキストのリスト
        metadata_list: 各歌詞のメタデータ
        output_path: 出力パス
        reference_progressions: 基準進行の辞書
    """
    if reference_progressions is None:
        reference_progressions = REFERENCE_PROGRESSIONS
    
    output_data = {
        'cluster_sizes': sorted(analysis_results.keys()),
        'reference_progressions': reference_progressions,
        'reference_colors': REFERENCE_COLORS,
        'analyses': {}
    }
    
    for min_cluster_size, results in analysis_results.items():
        topic_docs = results['topic_docs']
        topic_progressions = results['topic_progressions']
        topic_distances = results['topic_distances']
        topic_words = results.get('topic_words', {})
        topics = results['topics']
        
        # 座標を計算（コード進行の類似性でクラスタリングしてからMDS）
        try:
            # 手動トピックの場合（min_cluster_size=0）は、クラスタリングを行わず元のトピックIDを使用
            if min_cluster_size == 0:
                # 手動トピックの場合はクラスタリングをスキップ
                coordinates, ref_names, cluster_ids = compute_topic_coordinates_from_progressions(
                    topic_progressions,
                    reference_progressions,
                    use_clustering=False,  # 手動トピックの場合はクラスタリングしない
                    min_cluster_size=20  # 使用されないが、パラメータとして必要
                )
                # 手動トピックの場合、topic_docsに存在するトピックIDのみを使用
                # cluster_idsはtopic_progressionsのキーから取得されるが、
                # topic_docsに存在するもののみをフィルタリング
                topic_ids = sorted([tid for tid in cluster_ids if tid in topic_docs and tid != -1])
                # 座標のインデックスをマッピング（cluster_idsのインデックス -> topic_idsのインデックス）
                # cluster_idsからtopic_idsへのマッピングを作成
                cluster_id_to_coord_idx = {cid: idx for idx, cid in enumerate(cluster_ids)}
                topic_id_to_coord_idx = {tid: cluster_id_to_coord_idx[tid] for tid in topic_ids if tid in cluster_id_to_coord_idx}
            else:
                # BERTopicの場合はクラスタリングを使用
                cluster_size = min_cluster_size if min_cluster_size > 0 else 20
                coordinates, ref_names, cluster_ids = compute_topic_coordinates_from_progressions(
                    topic_progressions,
                    reference_progressions,
                    use_clustering=True,  # コード進行の類似性でクラスタリング
                    min_cluster_size=cluster_size
                )
                topic_ids = cluster_ids  # クラスタIDをトピックIDとして使用
        except Exception as e:
            print(f"Error computing coordinates for min_cluster_size={min_cluster_size}: {e}")
            continue
        
        # 座標を0-100に正規化（基準進行の相対的な配置は保持）
        if len(coordinates) > 0:
            n_refs = len(ref_names)
            n_topics = len(topic_ids)
            
            coordinates_normalized = np.zeros_like(coordinates)
            
            # 基準進行の座標はそのまま保持（相対的な配置を維持）
            for i in range(n_refs):
                coordinates_normalized[i] = coordinates[i]
            
            # トピックの座標のみを正規化（基準進行の範囲を考慮）
            if n_topics > 0:
                topic_coords = coordinates[n_refs:]
                ref_coords = coordinates[:n_refs]
                
                # 基準進行とトピックの全体の範囲を取得
                all_coords = coordinates
                all_x_min, all_x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
                all_y_min, all_y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
                
                all_x_range = all_x_max - all_x_min if all_x_max > all_x_min else 1
                all_y_range = all_y_max - all_y_min if all_y_max > all_y_min else 1
                
                # 基準進行の座標を0-100に正規化（相対的な配置を保持）
                for i in range(n_refs):
                    coordinates_normalized[i, 0] = ((coordinates[i, 0] - all_x_min) / all_x_range) * 100
                    coordinates_normalized[i, 1] = ((coordinates[i, 1] - all_y_min) / all_y_range) * 100
                
                # トピックの座標も同じ範囲で正規化
                for i in range(n_topics):
                    topic_idx = n_refs + i
                    coordinates_normalized[topic_idx, 0] = ((coordinates[topic_idx, 0] - all_x_min) / all_x_range) * 100
                    coordinates_normalized[topic_idx, 1] = ((coordinates[topic_idx, 1] - all_y_min) / all_y_range) * 100
        else:
            coordinates_normalized = coordinates
        
        # ポイントデータを構築
        points = []
        
        # 基準進行のポイント
        for i, ref_name in enumerate(ref_names):
            points.append({
                'x': float(coordinates_normalized[i, 0]),
                'y': float(coordinates_normalized[i, 1]),
                'r': 15.0,  # 基準進行は大きめ
                'type': 'reference',
                'reference_name': ref_name,
                'color': REFERENCE_COLORS.get(ref_name, '#000000'),
                'progression': reference_progressions[ref_name]
            })
        
        # トピックのポイント
        for i, topic_id in enumerate(topic_ids):
            topic_idx = len(ref_names) + i
            doc_indices = topic_docs[topic_id]
            size = len(doc_indices)
            
            # トピックの距離情報
            distances = topic_distances[topic_id]
            
            # トピックのキーワード
            words_info = topic_words.get(topic_id, {'words': [], 'scores': [], 'size': size})
            
            # 最も近い基準進行を特定
            min_dist = float('inf')
            closest_ref = None
            for ref_name, dist in distances.items():
                if dist < min_dist:
                    min_dist = dist
                    closest_ref = ref_name
            
            points.append({
                'x': float(coordinates_normalized[topic_idx, 0]),
                'y': float(coordinates_normalized[topic_idx, 1]),
                'r': max(5.0, min(20.0, np.sqrt(size) * 2)),  # サイズに応じた半径
                'type': 'topic',
                'topic_id': int(topic_id),
                'size': size,
                'words': words_info.get('words', [])[:10],  # 上位10語
                'word_scores': words_info.get('scores', [])[:10],
                'distances': {k: float(v) if not np.isinf(v) else None for k, v in distances.items()},
                'closest_reference': closest_ref,
                'min_distance': float(min_dist) if min_dist != float('inf') else None
            })
        
        # トピックごとの曲情報
        topic_songs = defaultdict(list)
        for doc_idx, topic_id in enumerate(topics):
            if topic_id == -1:  # ノイズは除外
                continue
            
            metadata = metadata_list[doc_idx]
            song_info = {
                'title': metadata.get('title', ''),
                'artist': metadata.get('artist', ''),
                'lyricist': metadata.get('lyricist', ''),
                'composer': metadata.get('composer', ''),
                'spotify_id': metadata.get('spotify_id', ''),
                'release_date': metadata.get('release_date', ''),
                'lyric_preview': lyrics_list[doc_idx][:100] if doc_idx < len(lyrics_list) else ''  # 最初の100文字
            }
            
            # 重複を避ける
            if song_info not in topic_songs[topic_id]:
                topic_songs[topic_id].append(song_info)
        
        output_data['analyses'][min_cluster_size] = {
            'min_cluster_size': min_cluster_size,
            'num_topics': len(topic_ids),
            'points': points,
            'topic_songs': {str(k): v for k, v in topic_songs.items()},
            'topic_info': {
                str(topic_id): {
                    'size': len(topic_docs[topic_id]),
                    'words': topic_words.get(topic_id, {}).get('words', [])[:10],
                    'distances': {k: float(v) if not np.isinf(v) else None 
                                 for k, v in topic_distances.get(topic_id, {}).items()}
                }
                for topic_id in topic_ids
            }
        }
    
    # JSONファイルに保存
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved JSON data to {output_path}")


if __name__ == "__main__":
    # テスト用
    pass

