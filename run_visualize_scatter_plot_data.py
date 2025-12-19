#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MDS/UMAP/t-SNEによるコード進行の可視化データ生成（エントリーポイント）
円グラフで歌詞の感情を表示

使用方法:
    python run_visualize_scatter_plot_data.py [data_dir] [max_files] [method]

    method: 'mds', 'umap', 'tsne', 'all' (デフォルト: 'all')
    
    直接距離行列ベースで、すべてのコード進行間の距離を使って1つのJSONファイルを生成します。
    基準進行（王道・小室・丸サ）は色付きの枠線で特別表示されます。
"""

import sys
from pathlib import Path

from program.visualize_scatter_plot_data.mds_visualization import main as mds_main

try:
    from program.visualize_scatter_plot_data.umap_visualization import main as umap_main
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False
    umap_main = None
    print("Warning: UMAP is not available in this environment.")

from program.visualize_scatter_plot_data.tsne_visualization import main as tsne_main


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv

    script_dir = Path(__file__).parent
    default_data_dir = str(script_dir / "data" / "analyzed")

    data_dir = argv[1] if len(argv) > 1 else default_data_dir
    max_files = int(argv[2]) if len(argv) > 2 else None
    method = argv[3].lower() if len(argv) > 3 else "all"

    print("=" * 60)
    print("コード進行（散布図）可視化データ生成")
    print("=" * 60)
    print(f"データディレクトリ: {data_dir}")
    print(f"最大ファイル数: {max_files if max_files else '無制限'}")
    print(f"手法: {method}")
    print(f"モード: 直接距離行列ベース（1つのJSONファイルにまとめる）")
    print("=" * 60)

    if method in ["mds", "all"]:
        print("\n" + "=" * 60)
        print("MDS可視化を実行中...")
        print("=" * 60)
        mds_main(
            data_dir=data_dir,
            max_files=max_files,
            show_lyrics=True,
            export_json=True,
        )

    if method in ["umap", "all"]:
        if UMAP_AVAILABLE and umap_main is not None:
            print("\n" + "=" * 60)
            print("UMAP可視化を実行中...")
            print("=" * 60)
            umap_main(
                data_dir=data_dir,
                max_files=max_files,
                show_lyrics=True,
                export_json=True,
            )
        else:
            print("\nError: UMAP is not available. (umap-learn / numba cache issue?)")

    if method in ["tsne", "all"]:
        print("\n" + "=" * 60)
        print("t-SNE可視化を実行中...")
        print("=" * 60)
        tsne_main(
            data_dir=data_dir,
            max_files=max_files,
            show_lyrics=True,
            export_json=True,
        )

    print("\n" + "=" * 60)
    print("完了!")
    print("=" * 60)


if __name__ == "__main__":
    main()


