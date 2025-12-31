#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
クラスタ単位の感情分布比較図の可視化（エントリーポイント）
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from program.visualize_cluster_emotion_data.cluster_emotion_analysis import main

if __name__ == "__main__":
    main()

