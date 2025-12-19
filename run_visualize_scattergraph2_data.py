#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
感情ごとのコード進行可視化データ生成（エントリーポイント）
typical_chord_distanceを基にMDS/UMAP/t-SNEで散布図を生成

使用方法:
    python run_visualize_scattergraph2_data.py [data_dir] [max_files] [method] [max_items]

    method: 'mds', 'umap', 'tsne', 'all' (デフォルト: 'all' - 全手法で生成)
    max_items: 感情ごとの最大アイテム数 (デフォルト: 10000)
    
    8感情ごとにJSONファイルを生成します（JOY_umap.json, TRUST_umap.json等）
    typical_chord_distance（odo, komuro, marusa）から距離行列を計算し、
    MDS/UMAP/t-SNEで2次元座標に変換します。
    
    注意: MDSはデータ量が多い場合メモリ不足でkillされる可能性があります。
          その場合は max_items を減らすか、'umap' または 'tsne' のみを指定してください。
"""

import sys
from pathlib import Path

from program.visualize_scattergraph2_data.main import main as scattergraph2_main


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv

    script_dir = Path(__file__).parent
    default_data_dir = str(script_dir / "data" / "analyzed")
    default_output_dir = str(script_dir / "vis_system" / "scattergraph2" / "data")

    data_dir = argv[1] if len(argv) > 1 else default_data_dir
    max_files = int(argv[2]) if len(argv) > 2 else None
    method = argv[3].lower() if len(argv) > 3 else "all"  # デフォルト: 全手法（MDS/UMAP/t-SNE）
    max_items = int(argv[4]) if len(argv) > 4 else 10000  # デフォルト: 10000アイテムに制限

    # 手法を決定（順番: umap -> t-sne -> MDS）
    if method == "all":
        methods = ['umap', 'tsne', 'mds']
    elif method in ['mds', 'umap', 'tsne']:
        methods = [method]
    else:
        print(f"警告: 不明な手法 '{method}'。'mds', 'umap', 'tsne', 'all' のいずれかを指定してください。")
        print("デフォルトとして 'all' を使用します。")
        methods = ['umap', 'tsne', 'mds']

    scattergraph2_main(
        data_dir=data_dir,
        output_dir=default_output_dir,
        max_files=max_files,
        methods=methods,
        max_items_per_emotion=max_items if max_items > 0 else None
    )


if __name__ == "__main__":
    main()


