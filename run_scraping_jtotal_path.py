#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
J-Total Musicの検索結果ページからパス一覧を取得（エントリーポイント）
"""

import sys
from program.scraping_jtotal_path.main import main

if __name__ == "__main__":
    # コマンドライン引数から取得件数を取得（デフォルト: None = 制限なし）
    max_items = None
    if len(sys.argv) > 1:
        try:
            max_items = int(sys.argv[1])
            print(f"取得件数の上限: {max_items} 件")
        except ValueError:
            print(f"警告: 無効な引数 '{sys.argv[1]}'。制限なしで実行します。")
            max_items = None
    else:
        print("制限なしで全ページ分を取得します")
    
    main(max_items=max_items)

