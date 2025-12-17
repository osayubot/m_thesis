#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
データ読み込み関連の関数
"""

import os
import json


def load_data_path_files(data_path_dir: str = 'data/jtotal/json/path') -> list:
    """
    data/jtotal/json/pathディレクトリ内のJSONファイルを読み込んで、全エントリを返す
    
    Args:
        data_path_dir: data/jtotal/json/pathディレクトリのパス
    
    Returns:
        {title, artist, path}のリスト
    """
    all_items = []
    
    if not os.path.exists(data_path_dir):
        print(f"警告: {data_path_dir} ディレクトリが見つかりません")
        return all_items
    
    # JSONファイルを取得（数字で始まるファイル）
    json_files = [f for f in os.listdir(data_path_dir) if f.endswith('.json') and f[0].isdigit()]
    json_files.sort(key=lambda x: int(x.split('-')[0]) if '-' in x else 0)
    
    for json_file in json_files:
        file_path = os.path.join(data_path_dir, json_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                items = json.load(f)
                # 各アイテムにパスファイル名を付加（ログファイル名を決めるため）
                path_file_name = os.path.splitext(json_file)[0]  # "1-1000.json" -> "1-1000"
                for item in items:
                    item['path_file'] = path_file_name
                all_items.extend(items)
        except Exception as e:
            print(f"警告: {json_file} の読み込みに失敗しました: {e}")
    
    return all_items

