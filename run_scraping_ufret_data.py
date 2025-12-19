#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ufret のデータを取得（エントリーポイント）
"""

import sys
from program.scraping_ufret_data.main import main

# 設定定数
START_ID = 59712  # 開始ID
END_ID = 188318   # 終了ID

if __name__ == "__main__":
    # コマンドライン引数から処理件数を取得（指定がない場合は全件）
    max_items = None
    if len(sys.argv) > 1:
        try:
            max_items = int(sys.argv[1])
        except ValueError:
            print(f"警告: 無効な引数 '{sys.argv[1]}'。全件で実行します。")
            max_items = None
    
    # メイン処理を実行
    # 各アイテムごとのログは自動的に data/log/{jtotal_path}.txt に保存されます
    main(max_items=max_items, start_id=START_ID, end_id=END_ID)

