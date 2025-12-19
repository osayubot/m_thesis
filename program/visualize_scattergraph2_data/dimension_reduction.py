"""
typical_chord_distanceから2次元座標を計算するモジュール
MDS/UMAP/t-SNEを使用
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.manifold import MDS, TSNE
from scipy.spatial.distance import pdist, squareform

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    umap = None


def compute_distance_matrix_from_typical_distances(
    data_list: List[Dict]
) -> np.ndarray:
    """
    typical_chord_distanceの3次元ベクトルから距離行列を計算
    
    Args:
        data_list: typical_chord_distanceを含むデータのリスト
    
    Returns:
        距離行列（n×n、n=データ数）
    """
    # 3次元ベクトルを抽出（odo, komuro, marusa）
    vectors = []
    for data in data_list:
        dist = data['typical_chord_distance']
        vectors.append([dist['odo'], dist['komuro'], dist['marusa']])
    
    vectors = np.array(vectors)
    
    # ユークリッド距離で距離行列を計算
    distances = pdist(vectors, metric='euclidean')
    distance_matrix = squareform(distances)
    
    return distance_matrix


def compute_mds_coordinates(
    data_list: List[Dict],
    n_components: int = 2,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    MDSで2次元座標を計算
    
    Args:
        data_list: typical_chord_distanceを含むデータのリスト
        n_components: 出力次元数（デフォルト: 2）
        random_state: 乱数シード
    
    Returns:
        2次元座標の配列（n×2、n=データ数）
    """
    distance_matrix = compute_distance_matrix_from_typical_distances(data_list)
    
    # MDSを適用
    mds = MDS(
        n_components=n_components,
        dissimilarity='precomputed',
        random_state=random_state,
        max_iter=300
    )
    coordinates = mds.fit_transform(distance_matrix)
    
    return coordinates


def compute_umap_coordinates(
    data_list: List[Dict],
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: Optional[int] = None
) -> Optional[np.ndarray]:
    """
    UMAPで2次元座標を計算
    
    Args:
        data_list: typical_chord_distanceを含むデータのリスト
        n_components: 出力次元数（デフォルト: 2）
        n_neighbors: 近傍数
        min_dist: 最小距離
        random_state: 乱数シード
    
    Returns:
        2次元座標の配列（n×2、n=データ数）、またはNone（UMAPが利用できない場合）
    """
    if not UMAP_AVAILABLE:
        return None
    
    # 3次元ベクトルを抽出
    vectors = []
    for data in data_list:
        dist = data['typical_chord_distance']
        vectors.append([dist['odo'], dist['komuro'], dist['marusa']])
    
    vectors = np.array(vectors)
    n_samples = len(vectors)
    
    # データ数が少ない場合の処理
    if n_samples < 3:
        # データが3未満の場合はPCA的な単純配置
        coordinates = np.array([[i * 50, 50] for i in range(n_samples)])
        return coordinates
    
    # n_neighborsをデータ数に応じて調整（データ数の半分以下、最小2）
    adjusted_n_neighbors = min(n_neighbors, max(2, n_samples - 1))
    
    # 常にrandom初期化を使用（spectral初期化は失敗する場合があり、警告が出るため）
    # random初期化でも十分な品質が得られる
    init_method = 'random'
    
    # UMAPを適用
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=adjusted_n_neighbors,
        min_dist=min_dist,
        init=init_method,
        random_state=random_state,
        verbose=False
    )
    
    try:
        coordinates = reducer.fit_transform(vectors)
    except Exception as e:
        # UMAPが失敗した場合はPCA的な単純配置にフォールバック
        print(f"  Warning: UMAP failed for {n_samples} samples, using fallback layout: {e}")
        # 主成分分析的な配置（単純な線形変換）
        mean_vec = vectors.mean(axis=0)
        centered = vectors - mean_vec
        # 最初の2次元を使う（または単純な配置）
        if vectors.shape[1] >= 2:
            coordinates = centered[:, :2] * 10 + 50  # スケールして中央に配置
        else:
            coordinates = np.array([[i * 50, 50] for i in range(n_samples)])
    
    return coordinates


def compute_tsne_coordinates(
    data_list: List[Dict],
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    t-SNEで2次元座標を計算
    
    Args:
        data_list: typical_chord_distanceを含むデータのリスト
        n_components: 出力次元数（デフォルト: 2）
        perplexity: パープレキシティ
        random_state: 乱数シード
    
    Returns:
        2次元座標の配列（n×2、n=データ数）
    """
    # 3次元ベクトルを抽出
    vectors = []
    for data in data_list:
        dist = data['typical_chord_distance']
        vectors.append([dist['odo'], dist['komuro'], dist['marusa']])
    
    vectors = np.array(vectors)
    n_samples = len(vectors)
    
    # データ数が少ない場合の処理
    if n_samples < 3:
        # データが3未満の場合はPCA的な単純配置
        coordinates = np.array([[i * 50, 50] for i in range(n_samples)])
        return coordinates
    
    # perplexityをデータ数に応じて調整
    # t-SNEでは perplexity < n_samples である必要がある
    # 推奨: perplexity <= n_samples - 1 かつ 5 <= perplexity <= 50
    adjusted_perplexity = min(perplexity, max(5.0, n_samples - 1))
    
    # init方法をデータ数に応じて調整
    # データ数が少ない場合は 'random' を使用（'pca'は少ないデータでは失敗する可能性がある）
    init_method = 'pca' if n_samples > 50 else 'random'
    
    # t-SNEを適用
    tsne = TSNE(
        n_components=n_components,
        perplexity=adjusted_perplexity,
        random_state=random_state,
        init=init_method
    )
    
    try:
        coordinates = tsne.fit_transform(vectors)
    except Exception as e:
        # t-SNEが失敗した場合はPCA的な単純配置にフォールバック
        print(f"  Warning: t-SNE failed for {n_samples} samples (perplexity={adjusted_perplexity}), using fallback layout: {e}")
        mean_vec = vectors.mean(axis=0)
        centered = vectors - mean_vec
        if vectors.shape[1] >= 2:
            coordinates = centered[:, :2] * 10 + 50  # スケールして中央に配置
        else:
            coordinates = np.array([[i * 50, 50] for i in range(n_samples)])
    
    return coordinates
