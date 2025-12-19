"""
Scatter-plot data generators (MDS / t-SNE / UMAP) for the visualization system.
"""

from .mds_visualization import main as visualize_mds  # noqa: F401
from .tsne_visualization import main as visualize_tsne  # noqa: F401

try:
    from .umap_visualization import main as visualize_umap  # noqa: F401
    UMAP_AVAILABLE = True
except Exception:
    visualize_umap = None
    UMAP_AVAILABLE = False

__all__ = [
    "visualize_mds",
    "visualize_tsne",
    "visualize_umap",
    "UMAP_AVAILABLE",
]


