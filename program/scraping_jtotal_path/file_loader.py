#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
既存ファイルの読み込み関数
"""

import os
import json


def load_existing_files(output_dir: str) -> dict:
    """
    既存のJSONファイルを読み込んで、既に取得したパスを返す
    
    Args:
        output_dir: 出力ディレクトリ
    
    Returns:
        既に取得したパスの辞書
    """
    path_to_item = {}
    
    if not os.path.exists(output_dir):
        return path_to_item
    
    # 既存のJSONファイルを読み込む
    json_files = [f for f in os.listdir(output_dir) if f.endswith('.json') and f[0].isdigit()]
    json_files.sort(key=lambda x: int(x.split('-')[0]))
    
    for json_file in json_files:
        file_path = os.path.join(output_dir, json_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                items = json.load(f)
                for item in items:
                    path = item.get('path', '')
                    if path:
                        path_to_item[path] = item
        except Exception as e:
            print(f"警告: {json_file} の読み込みに失敗しました: {e}")
    
    return path_to_item

