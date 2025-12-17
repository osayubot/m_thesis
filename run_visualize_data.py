#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MDSによるコード進行の可視化（エントリーポイント）
円グラフで歌詞の感情を表示
"""

import sys
from pathlib import Path
from program.visualize_data.mds_visualization import main

if __name__ == "__main__":
    # スクリプトのディレクトリを取得
    script_dir = Path(__file__).parent
    
    # デフォルトのデータディレクトリ
    default_data_dir = str(script_dir / "data" / "analyzed")
    
    # コマンドライン引数から取得
    data_dir = sys.argv[1] if len(sys.argv) > 1 else default_data_dir
    json_output_path = sys.argv[2] if len(sys.argv) > 2 else None
    max_files = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    # デフォルトでTrue（基準進行ベースのMDS）、'false'を指定すると従来の方法
    use_reference_based = not (len(sys.argv) > 4 and sys.argv[4].lower() == 'false')
    
    # 実行
    main(
        data_dir=data_dir,
        json_output_path=json_output_path,
        max_files=max_files,
        show_lyrics=True,
        export_json=True,
        use_reference_based=use_reference_based
    )

