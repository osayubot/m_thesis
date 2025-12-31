"""
UMAPによる座標計算と円グラフ可視化
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# UMAPをインポート（インストールされていない場合はエラーメッセージを表示）
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn is not installed. Please install it with: pip install umap-learn")

from .musical_distance import compute_distance_matrix, circular_distance
from ..analyze_data.roman_numeral import section_to_roman_progression

# common.pyから共通の関数・定数をインポート
from .common import (
    REFERENCE_PROGRESSIONS,
    REFERENCE_COLORS,
    EMOTION_COLORS,
    emotion_to_color,
    load_analyzed_data,
    extract_chord_progressions_with_lyrics,
    compute_distance_vectors,
    is_same_progression,
    export_to_json_format,
    add_reference_progressions
)

# UMAP用のファイル名マッピング
FILE_NAME_MAP = {
    'odo': 'umap_odo_pie_data.json',
    'komuro': 'umap_komuro_pie_data.json',
    'marusa': 'umap_marusa_pie_data.json',
}


def compute_reference_based_umap_coordinates(
    progressions_data: List[Dict],
    n_components: int = 2,
    reference_progressions: Dict[str, List[str]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'precomputed'
) -> Tuple[np.ndarray, List[str]]:
    """
    基準進行からの距離ベクトルを使ってUMAPで座標を計算
    
    Args:
        progressions_data: コード進行データ
        n_components: 次元数（2または3）
        reference_progressions: 基準進行の辞書（Noneの場合はデフォルトを使用）
        n_neighbors: UMAPの近傍数（デフォルト: 15）
        min_dist: UMAPの最小距離（デフォルト: 0.1）
        metric: 距離メトリック（'precomputed'で距離行列を直接使用）
    
    Returns:
        (座標配列（n×n_components）, 基準進行名のリスト)
    """
    if not UMAP_AVAILABLE:
        raise ImportError("umap-learn is not installed. Please install it with: pip install umap-learn")
    
    # 距離ベクトルを計算
    distance_vectors, ref_names = compute_distance_vectors(
        progressions_data, reference_progressions
    )
    
    # 距離ベクトル間のユークリッド距離で距離行列を作成
    print("Computing distance matrix from distance vectors...")
    n = len(distance_vectors)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(distance_vectors[i] - distance_vectors[j])
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
    
    # UMAPで座標を計算
    print("Computing UMAP coordinates...")
    reducer = umap.UMAP(
        n_components=n_components,
        metric='precomputed',
        n_neighbors=min(n_neighbors, n - 1),  # n_neighborsはデータ数-1以下
        min_dist=min_dist,
        random_state=42
    )
    coordinates = reducer.fit_transform(dist_matrix)
    
    return coordinates, ref_names


def compute_umap_coordinates(
    progressions_data: List[Dict],
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1
) -> np.ndarray:
    """
    UMAPで座標を計算（従来の方法：全コード進行間の距離行列を使用）
    
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
    dist_matrix = compute_distance_matrix(roman_progressions)
    
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
    show_lyrics: bool = True,
    export_json: bool = True,
    reference_progressions: Optional[Dict[str, List[str]]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1
):
    """
    メイン処理
    
    Args:
        data_dir: 分析済みデータディレクトリ
        output_path: 画像出力パス（Noneの場合は表示のみ）
        json_output_path: JSON出力パス（Noneの場合は自動生成）
        max_files: 最大ファイル数
        show_lyrics: 歌詞を表示するか（画像出力時）
        export_json: JSONを出力するか
        reference_progressions: 基準進行の辞書（Noneの場合はデフォルトを使用）
        n_neighbors: UMAPの近傍数
        min_dist: UMAPの最小距離
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
    
    # コード進行と歌詞を抽出
    print("Extracting chord progressions with lyrics...")
    progressions_data, songs_list = extract_chord_progressions_with_lyrics(songs)
    print(f"Found {len(progressions_data)} unique chord progressions")
    
    if len(progressions_data) == 0:
        print("No chord progressions found!")
        return
    
    # 基準進行を空のデータとして追加（UMAP計算に含めるため）
    # データに存在しない基準進行も、空のデータポイントとして追加され、
    # UMAPの座標計算に含まれます。散布図上では星（⭐）として表示されます。
    print("\nAdding reference progressions as empty data points...")
    if reference_progressions is None:
        reference_progressions = REFERENCE_PROGRESSIONS
    progressions_data = add_reference_progressions(progressions_data, reference_progressions)
    print(f"Total progressions (including references): {len(progressions_data)}")
    
    # 歌詞が少ないコード進行を除外
    # ただし、基準進行（isReferenceProgression=True）は必ず含める
    progressions_data_with_lyrics = [p for p in progressions_data if len(p.get('lyrics', [])) > 0 or p.get('isReferenceProgression', False)]
    print(f"After filtering (with lyrics or reference): {len(progressions_data_with_lyrics)} progressions")
    
    # 基準進行は必ず含める（制限は削除）
    # 注意: 以前はテスト用に100個に制限していましたが、全データを使用するように変更しました
    reference_progs = [p for p in progressions_data_with_lyrics if p.get('isReferenceProgression', False)]
    non_reference_progs = [p for p in progressions_data_with_lyrics if not p.get('isReferenceProgression', False)]
    
    # 制限を削除: 全データを使用
    # if len(non_reference_progs) > 100:
    #     print(f"Limiting to 100 progressions (from {len(non_reference_progs)})")
    #     import random
    #     random.seed(42)  # 再現性のためシードを固定
    #     sampled_progs = random.sample(non_reference_progs, 100)
    #     progressions_data_with_lyrics = reference_progs + sampled_progs
    #     print(f"Limited to {len(progressions_data_with_lyrics)} progressions (including {len(reference_progs)} references)")
    
    if len(progressions_data_with_lyrics) < 2:
        print("Not enough progressions for UMAP!")
        return
    
    # JSON出力
    if export_json:
        script_dir = Path(__file__).parent.parent.parent
        output_dir = script_dir / "vis_system" / "scattergraph" / "data"
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
        
        # JSONファイル名を決定（1つのファイルにまとめる）
        json_output_path = str(output_dir / "umap_all.json")
        
        # 基準進行を特別表示するため、すべての基準進行を渡す
        if reference_progressions is None:
            reference_progressions = REFERENCE_PROGRESSIONS
        
        print(f"Exporting JSON to {json_output_path}...")
        export_to_json_format(
            progressions_data_with_lyrics,
            coordinates,
            songs_list,
            json_output_path,
            reference_progressions=reference_progressions
        )
    
    print("Done!")


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/analyzed"
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    main(data_dir=data_dir, output_path=output_path)

