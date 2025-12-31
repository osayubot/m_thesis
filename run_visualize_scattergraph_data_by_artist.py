"""
アーティストごとの散布図を生成するランナースクリプト
"""
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from program.visualize_scattergraph_data.create_artist_scatterplots import main

if __name__ == "__main__":
    main(
        json_path=None,  # デフォルトパスを使用
        output_dir=None,  # デフォルトパスを使用
        min_cluster_size=20,
        min_phrases=20
    )

