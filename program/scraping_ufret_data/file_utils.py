#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ファイル保存関連のユーティリティ関数
"""

import os
import json


def get_subfolder_name(chord_id):
    """
    コードIDに基づいてサブフォルダ名を生成（1000曲ごとに分割）
    
    Args:
        chord_id: コードID
    
    Returns:
        str: サブフォルダ名（例: "1-1000", "1001-2000"）
    """
    # 1000曲ごとに分割
    folder_start = ((chord_id - 1) // 1000) * 1000 + 1
    folder_end = folder_start + 999
    return f"{folder_start}-{folder_end}"


def get_data_dir(base_dir=None, chord_id=None):
    """
    データディレクトリのパスを取得（サブフォルダを含む）
    
    Args:
        base_dir: ベースディレクトリ（Noneの場合は自動でdata/ufret/json/rawを探す）
        chord_id: コードID（サブフォルダを決定するために使用）
    
    Returns:
        str: データディレクトリのパス（サブフォルダを含む）
    """
    if base_dir is None:
        # 現在のファイルの場所からルートディレクトリを探す
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # program/scraping_ufret_data -> program -> ルート
        root_dir = os.path.dirname(os.path.dirname(current_dir))
        data_dir = os.path.join(root_dir, "data", "ufret", "json", "raw")
    else:
        data_dir = base_dir
    
    # サブフォルダを追加
    if chord_id is not None:
        subfolder = get_subfolder_name(chord_id)
        data_dir = os.path.join(data_dir, subfolder)
    
    return data_dir


def is_file_exists(chord_id, base_dir=None):
    """
    指定されたIDのJSONファイルが既に存在するかチェック
    
    Args:
        chord_id: コードID
        base_dir: ベースディレクトリ（Noneの場合は自動でdata/ufret/json/rawを探す）
    
    Returns:
        bool: ファイルが存在する場合True
    """
    data_dir = get_data_dir(base_dir, chord_id)
    filename = os.path.join(data_dir, f"{chord_id}.json")
    return os.path.exists(filename)


def save_to_json(data, chord_id, base_dir=None):
    """
    Save data to JSON file (saves to subfolder based on chord_id)
    
    Args:
        data: 保存するデータ（辞書）
        chord_id: コードID（ファイル名に使用、サブフォルダも決定）
        base_dir: ベースディレクトリ（Noneの場合は自動でdata/ufretを探す）
    
    Returns:
        bool: 保存成功時True
    """
    try:
        data_dir = get_data_dir(base_dir, chord_id)
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        filename = os.path.join(data_dir, f"{chord_id}.json")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"データを{filename}に保存しました。")
        return True
    except Exception as e:
        print(f"JSONファイルの保存中にエラーが発生しました: {str(e)}")
        return False

