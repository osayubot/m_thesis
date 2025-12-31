"""
UMAPによる座標計算とトピック割合の円グラフ可視化
BERTopicを使用してトピック分割を行い、感情の代わりにトピック割合を表示
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# UMAPをインポート
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn is not installed. Please install it with: pip install umap-learn")

from .common_bertopic import (
    REFERENCE_PROGRESSIONS,
    REFERENCE_COLORS,
    load_analyzed_data,
    extract_chord_progressions_with_lyrics_for_topic,
    compute_distance_vectors,
    perform_topic_modeling_on_lyrics,
    export_to_json_format_with_topics,
    add_reference_progressions
)
from ..visualize_scattergraph_data.musical_distance import compute_distance_matrix


def compute_umap_coordinates(
    progressions_data: List[Dict],
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1
) -> np.ndarray:
    """
    UMAPで座標を計算（全コード進行間の距離行列を使用）
    
    Args:
        progressions_data: コード進行データ
        n_components: 次元数（2または3）
        n_neighbors: UMAPの近傍数
        min_dist: UMAPの最小距離
    
    Returns:
        座標配列（n×n_components）
    """
    if not UMAP_AVAILABLE:
        raise ImportError("umap-learn is not installed. Please install it with: pip install umap-learn")
    
    # ローマ数字のコード進行を取得
    roman_progressions = [prog['roman_progression'] for prog in progressions_data]
    
    # 距離行列を計算
    print("Computing distance matrix...")
    dist_matrix = compute_distance_matrix(roman_progressions, show_progress=True)
    
    n = len(dist_matrix)
    
    # UMAPで座標を計算
    print("Computing UMAP coordinates...")
    reducer = umap.UMAP(
        n_components=n_components,
        metric='precomputed',
        n_neighbors=min(n_neighbors, n - 1),
        min_dist=min_dist,
        random_state=42
    )
    coordinates = reducer.fit_transform(dist_matrix)
    
    return coordinates


def main(
    data_dir: str = "data/analyzed",
    output_path: Optional[str] = None,
    json_output_path: Optional[str] = None,
    max_files: Optional[int] = None,
    export_json: bool = True,
    reference_progressions: Optional[Dict[str, List[str]]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    min_cluster_size: int = 15,
    bertopic_n_neighbors: int = 15,
    bertopic_min_dist: float = 0.1
):
    """
    メイン処理（トピックベース）
    
    Args:
        data_dir: 分析済みデータディレクトリ
        output_path: 画像出力パス（未使用、後方互換性のため）
        json_output_path: JSON出力パス（Noneの場合は自動生成）
        max_files: 最大ファイル数
        export_json: JSONを出力するか
        reference_progressions: 基準進行の辞書（Noneの場合はデフォルトを使用）
        n_neighbors: UMAPの近傍数（座標計算用）
        min_dist: UMAPの最小距離（座標計算用）
        min_cluster_size: HDBSCANの最小クラスタサイズ（BERTopic用）
        bertopic_n_neighbors: UMAPの近傍数（BERTopic内部用）
        bertopic_min_dist: UMAPの最小距離（BERTopic内部用）
    """
    if not UMAP_AVAILABLE:
        print("Error: umap-learn is not installed. Please install it with: pip install umap-learn")
        return
    
    # データディレクトリのパスを解決
    data_path = Path(data_dir)
    if not data_path.is_absolute():
        script_dir = Path(__file__).parent.parent.parent
        data_path = script_dir / data_dir
    
    if not data_path.exists():
        print(f"Error: Data directory not found: {data_path}")
        return
    
    # データを読み込む
    print(f"Loading data from {data_path}...")
    songs = load_analyzed_data(str(data_path), max_files)
    print(f"Loaded {len(songs)} songs")
    
    # コード進行と歌詞を抽出（トピック分析用、感情データのフィルタリングなし）
    print("Extracting chord progressions with lyrics...")
    progressions_data, songs_list = extract_chord_progressions_with_lyrics_for_topic(songs)
    print(f"Found {len(progressions_data)} unique chord progressions")
    
    if len(progressions_data) == 0:
        print("No chord progressions found!")
        return
    
    # 基準進行を空のデータとして追加
    print("\nAdding reference progressions as empty data points...")
    if reference_progressions is None:
        reference_progressions = REFERENCE_PROGRESSIONS
    progressions_data = add_reference_progressions(progressions_data, reference_progressions)
    print(f"Total progressions (including references): {len(progressions_data)}")
    
    # 歌詞が少ないコード進行を除外（基準進行は除く）
    progressions_data_with_lyrics = [
        p for p in progressions_data 
        if len(p.get('lyrics', [])) > 0 or p.get('isReferenceProgression', False)
    ]
    print(f"After filtering (with lyrics or reference): {len(progressions_data_with_lyrics)} progressions")
    
    if len(progressions_data_with_lyrics) < 2:
        print("Not enough progressions for UMAP!")
        return
    
    # BERTopicでトピック分析を実行
    print(f"\n{'='*60}")
    print("Performing topic modeling with BERTopic...")
    print(f"{'='*60}")
    print(f"Parameters: min_cluster_size={min_cluster_size}, n_neighbors={bertopic_n_neighbors}")
    
    # 全歌詞を収集してトピック分析
    all_lyrics = []
    lyric_to_progression_index = []
    
    for prog_idx, prog_data in enumerate(progressions_data_with_lyrics):
        lyrics = prog_data.get('lyrics', [])
        for lyric_data in lyrics:
            lyric_text = lyric_data.get('lyric', '').strip()
            if lyric_text:
                all_lyrics.append(lyric_text)
                lyric_to_progression_index.append(prog_idx)
    
    if len(all_lyrics) == 0:
        print("No lyrics found for topic modeling!")
        return
    
    # トピック分析を実行
    topic_model, topic_to_lyric_indices, topic_names, all_topics = perform_topic_modeling_on_lyrics(
        progressions_data_with_lyrics,
        min_cluster_size=min_cluster_size,
        n_neighbors=bertopic_n_neighbors,
        min_dist=bertopic_min_dist
    )
    
    if topic_model is None:
        print("Error: Topic modeling failed!")
        return
    
    # JSON出力
    if export_json:
        script_dir = Path(__file__).parent.parent.parent
        output_dir = script_dir / "vis_system" / "scattergraph_bertopic" / "data"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 直接距離行列ベースのUMAP（すべてのコード進行間の距離を使用）
        print(f"\n{'='*60}")
        print("Computing UMAP coordinates from direct distance matrix...")
        print(f"{'='*60}")
        
        # 直接距離行列を使ってUMAP座標を計算（基準進行も含む）
        coordinates = compute_umap_coordinates(
            progressions_data_with_lyrics,
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist
        )
        
        # JSONファイル名を決定
        if json_output_path is None:
            json_output_path = str(output_dir / "umap_all_topics.json")
        
        # 基準進行を特別表示するため、すべての基準進行を渡す
        if reference_progressions is None:
            reference_progressions = REFERENCE_PROGRESSIONS
        
        print(f"Exporting JSON to {json_output_path}...")
        export_to_json_format_with_topics(
            progressions_data_with_lyrics,
            coordinates,
            songs_list,
            topic_model,
            topic_to_lyric_indices,
            topic_names,
            all_topics,
            lyric_to_progression_index,
            all_lyrics,
            json_output_path,
            reference_progressions=reference_progressions
        )
    
    print("Done!")


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/analyzed"
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    main(data_dir=data_dir, output_path=output_path)

