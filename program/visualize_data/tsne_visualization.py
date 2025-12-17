"""
t-SNEによる座標計算と円グラフ可視化
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from sklearn.manifold import TSNE

from .musical_distance import compute_distance_matrix, circular_distance
from ..analyze_data.roman_numeral import section_to_roman_progression

# mds_visualization.pyから共通の関数・定数をインポート
from .mds_visualization import (
    REFERENCE_PROGRESSIONS,
    EMOTION_COLORS,
    emotion_to_color,
    load_analyzed_data,
    extract_chord_progressions_with_lyrics,
    compute_distance_vectors,
    is_same_progression,
    export_to_json_format
)

# t-SNE用のファイル名マッピング
FILE_NAME_MAP = {
    'odo': 'tsne_odo_pie_data.json',
    'komuro': 'tsne_komuro_pie_data.json',
    'marusa': 'tsne_marusa_pie_data.json',
}


def compute_reference_based_tsne_coordinates(
    progressions_data: List[Dict],
    n_components: int = 2,
    reference_progressions: Dict[str, List[str]] = None,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    n_iter: int = 1000
) -> Tuple[np.ndarray, List[str]]:
    """
    基準進行からの距離ベクトルを使ってt-SNEで座標を計算
    
    Args:
        progressions_data: コード進行データ
        n_components: 次元数（2または3）
        reference_progressions: 基準進行の辞書（Noneの場合はデフォルトを使用）
        perplexity: t-SNEの困惑度（5〜50が一般的）
        learning_rate: 学習率
        n_iter: 反復回数
    
    Returns:
        (座標配列（n×n_components）, 基準進行名のリスト)
    """
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
    
    # t-SNEで座標を計算
    print("Computing t-SNE coordinates...")
    # perplexityはデータ数-1以下でなければならない
    actual_perplexity = min(perplexity, n - 1)
    if actual_perplexity < 5:
        actual_perplexity = max(2, n - 1)
    
    tsne = TSNE(
        n_components=n_components,
        metric='precomputed',
        perplexity=actual_perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=42,
        init='random'
    )
    coordinates = tsne.fit_transform(dist_matrix)
    
    return coordinates, ref_names


def compute_tsne_coordinates(
    progressions_data: List[Dict],
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    n_iter: int = 1000
) -> np.ndarray:
    """
    t-SNEで座標を計算（従来の方法：全コード進行間の距離行列を使用）
    
    Args:
        progressions_data: コード進行データ
        n_components: 次元数（2または3）
        perplexity: t-SNEの困惑度
        learning_rate: 学習率
        n_iter: 反復回数
    
    Returns:
        座標配列（n×n_components）
    """
    # ローマ数字のコード進行を取得
    roman_progressions = [prog['roman_progression'] for prog in progressions_data]
    
    # 距離行列を計算
    print("Computing distance matrix...")
    dist_matrix = compute_distance_matrix(roman_progressions)
    
    n = len(dist_matrix)
    
    # t-SNEで座標を計算
    print("Computing t-SNE coordinates...")
    actual_perplexity = min(perplexity, n - 1)
    if actual_perplexity < 5:
        actual_perplexity = max(2, n - 1)
    
    tsne = TSNE(
        n_components=n_components,
        metric='precomputed',
        perplexity=actual_perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=42,
        init='random'
    )
    coordinates = tsne.fit_transform(dist_matrix)
    
    return coordinates


def main(
    data_dir: str = "data/analyzed",
    output_path: Optional[str] = None,
    json_output_path: Optional[str] = None,
    max_files: Optional[int] = None,
    show_lyrics: bool = True,
    export_json: bool = True,
    use_reference_based: bool = True,
    reference_progressions: Optional[Dict[str, List[str]]] = None,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    n_iter: int = 1000
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
        use_reference_based: 基準進行ベースのt-SNEを使用するか
        reference_progressions: 基準進行の辞書（Noneの場合はデフォルトを使用）
        perplexity: t-SNEの困惑度
        learning_rate: 学習率
        n_iter: 反復回数
    """
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
    
    # 歌詞が少ないコード進行を除外
    progressions_data = [p for p in progressions_data if len(p['lyrics']) > 0]
    print(f"After filtering (with lyrics): {len(progressions_data)} progressions")
    
    if len(progressions_data) < 2:
        print("Not enough progressions for t-SNE!")
        return
    
    # JSON出力
    if export_json:
        script_dir = Path(__file__).parent.parent.parent
        output_dir = script_dir / "vis_system" / "data"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        if use_reference_based:
            if reference_progressions is None:
                reference_progressions = REFERENCE_PROGRESSIONS
            
            filename_map = FILE_NAME_MAP
            
            for ref_name, ref_prog in reference_progressions.items():
                print(f"\n{'='*60}")
                print(f"Processing reference progression (t-SNE): {ref_name}")
                print(f"Progression: {' - '.join(ref_prog)}")
                print(f"{'='*60}")
                
                # 基準進行がデータに含まれているかチェック
                ref_exists = False
                for prog_data in progressions_data:
                    if is_same_progression(prog_data.get('roman_progression', []), ref_prog):
                        ref_exists = True
                        print(f"  Found reference progression {ref_name} in data")
                        break
                
                # 基準進行がデータに含まれていない場合、ダミーデータを追加
                current_progressions_data = progressions_data.copy()
                if not ref_exists:
                    print(f"  Warning: Reference progression {ref_name} not found in data. Adding dummy entry...")
                    dummy_prog_data = {
                        'chord_progression': [],
                        'normalized_chord_progression': [],
                        'roman_progression': ref_prog.copy(),
                        'lyrics': [{
                            'lyric': f'[{ref_name}]',
                            'emotion': {'JOY': 0.5},
                            'color': '#808080',
                            'song_index': 0
                        }],
                        'key': 'C'
                    }
                    current_progressions_data.append(dummy_prog_data)
                    print(f"  Added dummy entry for {ref_name}")
                
                # 距離ベクトルを計算（3つの基準進行すべてを使用）
                distance_vectors, ref_names = compute_distance_vectors(
                    current_progressions_data,
                    reference_progressions=reference_progressions
                )
                
                # 主基準進行への距離を強調
                main_ref_index = ref_names.index(ref_name)
                weighted_vectors = distance_vectors.copy()
                weighted_vectors[:, main_ref_index] = weighted_vectors[:, main_ref_index] * 3.0
                
                # 重み付けされた距離ベクトル間のユークリッド距離で距離行列を作成
                print("Computing distance matrix from weighted distance vectors...")
                n = len(weighted_vectors)
                dist_matrix = np.zeros((n, n))
                
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = np.linalg.norm(weighted_vectors[i] - weighted_vectors[j])
                        dist_matrix[i][j] = dist
                        dist_matrix[j][i] = dist
                
                # t-SNEで座標を計算
                print("Computing t-SNE coordinates...")
                actual_perplexity = min(perplexity, n - 1)
                if actual_perplexity < 5:
                    actual_perplexity = max(2, n - 1)
                
                tsne = TSNE(
                    n_components=2,
                    metric='precomputed',
                    perplexity=actual_perplexity,
                    learning_rate=learning_rate,
                    n_iter=n_iter,
                    random_state=42,
                    init='random'
                )
                coordinates = tsne.fit_transform(dist_matrix)
                
                # JSONファイル名を決定
                filename = filename_map.get(ref_name, f"tsne_{ref_name}_pie_data.json")
                json_output_path = str(output_dir / filename)
                
                print(f"Exporting JSON to {json_output_path}...")
                export_to_json_format(
                    current_progressions_data,
                    coordinates,
                    songs_list,
                    json_output_path,
                    reference_progression=ref_prog,
                    reference_name=ref_name
                )
    
    print("Done!")


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/analyzed"
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    use_reference = len(sys.argv) > 3 and sys.argv[3].lower() == 'true'
    main(data_dir=data_dir, output_path=output_path, use_reference_based=use_reference)

