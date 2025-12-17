#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MDS/UMAP/t-SNEによるコード進行の可視化（エントリーポイント）
円グラフで歌詞の感情を表示

使用方法:
    python run_visualize_data.py [data_dir] [max_files] [method]
    
    method: 'mds', 'umap', 'tsne', 'all' (デフォルト: 'all')
"""

import sys
from pathlib import Path
from program.visualize_data.mds_visualization import main as mds_main

# UMAPが利用可能かチェック
try:
    from program.visualize_data.umap_visualization import main as umap_main
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP is not available. Install with: pip install umap-learn")

# t-SNEは常に利用可能（scikit-learnに含まれる）
from program.visualize_data.tsne_visualization import main as tsne_main

if __name__ == "__main__":
    # スクリプトのディレクトリを取得
    script_dir = Path(__file__).parent
    
    # デフォルトのデータディレクトリ
    default_data_dir = str(script_dir / "data" / "analyzed")
    
    # コマンドライン引数から取得
    data_dir = sys.argv[1] if len(sys.argv) > 1 else default_data_dir
    max_files = int(sys.argv[2]) if len(sys.argv) > 2 else None
    method = sys.argv[3].lower() if len(sys.argv) > 3 else 'all'
    
    print("="*60)
    print("コード進行可視化データ生成")
    print("="*60)
    print(f"データディレクトリ: {data_dir}")
    print(f"最大ファイル数: {max_files if max_files else '無制限'}")
    print(f"手法: {method}")
    print("="*60)
    
    # MDS実行
    if method in ['mds', 'all']:
        print("\n" + "="*60)
        print("MDS可視化を実行中...")
        print("="*60)
        mds_main(
            data_dir=data_dir,
            max_files=max_files,
            show_lyrics=True,
            export_json=True,
            use_reference_based=True
        )
    
    # UMAP実行
    if method in ['umap', 'all']:
        if UMAP_AVAILABLE:
            print("\n" + "="*60)
            print("UMAP可視化を実行中...")
            print("="*60)
            umap_main(
                data_dir=data_dir,
                max_files=max_files,
                show_lyrics=True,
                export_json=True,
                use_reference_based=True
            )
        else:
            print("\nError: UMAP is not available. Install with: pip install umap-learn")
    
    # t-SNE実行
    if method in ['tsne', 'all']:
        print("\n" + "="*60)
        print("t-SNE可視化を実行中...")
        print("="*60)
        tsne_main(
            data_dir=data_dir,
            max_files=max_files,
            show_lyrics=True,
            export_json=True,
            use_reference_based=True
        )
    
    print("\n" + "="*60)
    print("完了!")
    print("="*60)

