#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UMAP/t-SNEによるコード進行の可視化データ生成（エントリーポイント - トピックベース）
円グラフで歌詞のトピック割合を表示

BERTopicを使用してトピック分割を行います：
- HDBSCANを使用（K指定不要、自動でトピック数を決定）
- 完全自動でトピック分割
- 感情の代わりにトピック割合を円グラフで表示

使用方法:
    python run_visualize_scattergraph_bertopic_data.py [data_dir] [max_files] [method] [min_cluster_size]

    method: 'umap', 'tsne', 'all' (デフォルト: 'all')
    min_cluster_size: HDBSCANの最小クラスタサイズ（デフォルト: 30）
    
    直接距離行列ベースで、すべてのコード進行間の距離を使って1つのJSONファイルを生成します。
    基準進行（王道・小室・丸サ）は色付きの枠線で特別表示されます。
"""

import sys
from pathlib import Path

try:
    from program.visualize_scattergraph_bertopic_data.umap_visualization_bertopic import main as umap_main
    UMAP_AVAILABLE = True
except Exception as e:
    UMAP_AVAILABLE = False
    umap_main = None
    print(f"Warning: UMAP is not available in this environment: {e}")

try:
    from program.visualize_scattergraph_bertopic_data.tsne_visualization_bertopic import main as tsne_main
    TSNE_AVAILABLE = True
except Exception as e:
    TSNE_AVAILABLE = False
    tsne_main = None
    print(f"Warning: t-SNE is not available in this environment: {e}")


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv

    script_dir = Path(__file__).parent
    default_data_dir = str(script_dir / "data" / "analyzed")

    data_dir = argv[1] if len(argv) > 1 else default_data_dir
    if len(argv) > 2:
        max_files_str = argv[2].lower()
        if max_files_str in ['none', 'null']:
            max_files = None
        else:
            max_files = int(max_files_str)
    else:
        max_files = None
    method = argv[3].lower() if len(argv) > 3 else "all"
    # デフォルト値を15に下げる（200曲程度の場合により適切）
    min_cluster_size = int(argv[4]) if len(argv) > 4 else 15

    print("=" * 60)
    print("コード進行（散布図）可視化データ生成 - トピックベース")
    print("=" * 60)
    print(f"データディレクトリ: {data_dir}")
    print(f"最大ファイル数: {max_files if max_files else '無制限'}")
    print(f"手法: {method}")
    print(f"最小クラスタサイズ: {min_cluster_size}")
    print(f"モード: 直接距離行列ベース（1つのJSONファイルにまとめる）")
    print(f"トピック分割: BERTopic + HDBSCAN（完全自動、K指定不要）")
    print("=" * 60)

    if method in ["umap", "all"]:
        if UMAP_AVAILABLE and umap_main is not None:
            print("\n" + "=" * 60)
            print("UMAP可視化を実行中（トピックベース）...")
            print("=" * 60)
            umap_main(
                data_dir=data_dir,
                max_files=max_files,
                export_json=True,
                min_cluster_size=min_cluster_size,
            )
        else:
            print("\nError: UMAP is not available. (umap-learn / bertopic / numba cache issue?)")

    if method in ["tsne", "all"]:
        if TSNE_AVAILABLE and tsne_main is not None:
            print("\n" + "=" * 60)
            print("t-SNE可視化を実行中（トピックベース）...")
            print("=" * 60)
            tsne_main(
                data_dir=data_dir,
                max_files=max_files,
                export_json=True,
                min_cluster_size=min_cluster_size,
            )
        else:
            print("\nError: t-SNE is not available.")

    print("\n" + "=" * 60)
    print("完了!")
    print("=" * 60)


if __name__ == "__main__":
    main()
