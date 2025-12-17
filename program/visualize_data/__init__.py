"""
データ可視化パッケージ
"""
from .mds_visualization import main as visualize_mds
from .tsne_visualization import main as visualize_tsne
from .musical_distance import (
    musical_levenshtein_distance,
    circular_distance,
    compute_distance_matrix
)

# UMAPが利用可能な場合のみインポート
try:
    from .umap_visualization import main as visualize_umap
    UMAP_AVAILABLE = True
except ImportError:
    visualize_umap = None
    UMAP_AVAILABLE = False

__all__ = [
    'visualize_mds',
    'visualize_umap',
    'visualize_tsne',
    'musical_levenshtein_distance',
    'circular_distance',
    'compute_distance_matrix',
    'UMAP_AVAILABLE',
]

