"""
scattergraph_bertopic: BERTopicによるトピック分析と基準進行との関係可視化（メイン処理）
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, List

from .topic_analysis import (
    extract_lyrics_for_topic_modeling,
    analyze_topics_for_multiple_cluster_sizes
)
from .json_generator import generate_json_data
from ..visualize_scattergraph_data.common import (
    load_analyzed_data,
    REFERENCE_PROGRESSIONS
)


def main(
    data_dir: str = "data/analyzed",
    output_path: Optional[str] = None,
    max_files: Optional[int] = None,
    cluster_sizes: Optional[List[int]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1
) -> None:
    """
    メイン処理
    
    Args:
        data_dir: 分析済みデータディレクトリ
        output_path: JSON出力パス（Noneの場合は自動生成）
        max_files: 最大ファイル数
        cluster_sizes: 試すmin_cluster_sizeのリスト（Noneの場合は[20, 30, 40]）
        n_neighbors: UMAPの近傍数
        min_dist: UMAPの最小距離
    """
    # データディレクトリのパスを解決
    data_path = Path(data_dir)
    if not data_path.is_absolute():
        script_dir = Path(__file__).parent.parent.parent
        data_path = script_dir / data_dir
    
    if not data_path.exists():
        print(f"Error: Data directory not found: {data_path}")
        return
    
    # デフォルトのクラスタサイズ
    if cluster_sizes is None:
        cluster_sizes = [20, 30, 40]
    
    # 出力パスを決定
    if output_path is None:
        script_dir = Path(__file__).parent.parent.parent
        output_dir = script_dir / "vis_system" / "scattergraph_bertopic" / "data"
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = str(output_dir / "topic_analysis.json")
    
    print("=" * 60)
    print("BERTopicによるトピック分析と基準進行との関係可視化")
    print("=" * 60)
    print(f"データディレクトリ: {data_path}")
    print(f"最大ファイル数: {max_files if max_files else '無制限'}")
    print(f"クラスタサイズ: {cluster_sizes}")
    print(f"出力パス: {output_path}")
    print("=" * 60)
    
    # データを読み込む
    print("\nデータを読み込んでいます...")
    songs = load_analyzed_data(str(data_path), max_files)
    print(f"読み込み完了: {len(songs)}曲")
    
    if len(songs) == 0:
        print("Error: No songs loaded!")
        return
    
    # 歌詞を抽出
    print("\n歌詞を抽出しています...")
    lyrics_list, metadata_list = extract_lyrics_for_topic_modeling(songs)
    print(f"抽出完了: {len(lyrics_list)}フレーズ")
    
    if len(lyrics_list) == 0:
        print("Error: No lyrics extracted!")
        return
    
    # トピック分析を実行
    print("\nトピック分析を実行しています...")
    print("（これには時間がかかる場合があります）")
    
    try:
        analysis_results = analyze_topics_for_multiple_cluster_sizes(
            lyrics_list,
            metadata_list,
            cluster_sizes=cluster_sizes,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            use_manual_topics=True  # 手動定義トピックを使用
        )
    except Exception as e:
        print(f"Error during topic analysis: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not analysis_results:
        print("Error: No analysis results generated!")
        return
    
    # JSONデータを生成
    print("\n可視化用JSONデータを生成しています...")
    try:
        generate_json_data(
            analysis_results,
            lyrics_list,
            metadata_list,
            output_path,
            reference_progressions=REFERENCE_PROGRESSIONS
        )
    except Exception as e:
        print(f"Error generating JSON data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("完了!")
    print("=" * 60)
    print(f"結果は {output_path} に保存されました")


if __name__ == "__main__":
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/analyzed"
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    main(data_dir=data_dir, output_path=output_path)

