#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
メイン処理
"""

import os
import json
import glob
import sys

from .data_loader import load_data_path_files
from .spotify_utils import add_spotify_info
from .extract_html_data import extract_and_save_html_data


def main(max_items=None):
    """メイン処理"""
    # data/jtotal/json/pathディレクトリから全エントリを読み込む
    print("data/jtotal/json/pathディレクトリからデータを読み込み中...")
    all_items = load_data_path_files('data/jtotal/json/path')
    print(f"合計 {len(all_items)} 件のエントリを読み込みました")
    
    if not all_items:
        print("処理するデータがありません。")
        exit(1)
    
    # 処理済みのパスを確認（既にJSONファイルが存在するか）
    # 移調前のJSONが保存されていれば処理済みとみなす
    processed_paths = set()
    if os.path.exists('data/jtotal/json/raw'):
        for json_file in glob.glob('data/jtotal/json/raw/**/*.json', recursive=True):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'jtotal_path' in data:
                        processed_paths.add(data['jtotal_path'])
            except:
                pass
    
    print(f"既に処理済み（移調前のJSONが存在）: {len(processed_paths)} 件")
    
    # 未処理のアイテムのみをフィルタリング
    unprocessed_items = [item for item in all_items if item.get('path', '') not in processed_paths]
    
    # 処理件数の制限を適用
    if max_items is not None:
        items_to_process = unprocessed_items[:max_items]
        print(f"処理対象: {len(items_to_process)} 件（上限: {max_items} 件）")
    else:
        items_to_process = unprocessed_items
        print(f"処理対象: 全 {len(items_to_process)} 件（未処理のみ）")
    
    print("-" * 50)
    
    # 各エントリを処理
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for i, item in enumerate(items_to_process, 1):
        title = item.get('title', '')
        artist = item.get('artist', '')
        jtotal_path = item.get('path', '')
        
        if not jtotal_path:
            print(f"[{i}/{len(items_to_process)}] パスが空のためスキップ")
            skip_count += 1
            continue
        
        # 既に処理済みかチェック
        if jtotal_path in processed_paths:
            print(f"[{i}/{len(items_to_process)}] 既に処理済み: {title} - {artist} ({jtotal_path})")
            skip_count += 1
            continue
        print(f"[{i}/{len(items_to_process)}] 処理中: {title} - {artist}")
        print(f"  パス: {jtotal_path}")
        
        # まず、基本的なJSONデータを作成
        json_data = {
            'title': title,
            'artist': artist,
            'jtotal_path': jtotal_path
        }
        
        # Spotify情報を先に検索
        print("  Spotify情報を検索中...")
        print(f"    曲名: {title}")
        print(f"    アーティスト（日本語）: {artist}")
        if jtotal_path:
            from program.jtotal_data_scraping.spotify_utils import extract_artist_en_from_path
            artist_en = extract_artist_en_from_path(jtotal_path)
            if artist_en:
                print(f"    アーティスト（英語）: {artist_en}")
        json_data = add_spotify_info(json_data)
        
        # Spotify情報があるかチェック（spotify_idが存在するか）
        has_spotify_info = 'spotify_id' in json_data and json_data.get('spotify_id')
        
        if has_spotify_info:
            print("  Spotify情報が見つかりました。j-total-chordからデータを取得します...")
        else:
            print("  Spotify情報が見つかりませんでしたが、データを取得します...")
        
        # Spotify情報の有無に関わらず、j-total-chordの処理を実行
        url = f'https://music.j-total.net/data/{jtotal_path}.html'
        
        # 既存のSpotify情報を抽出
        existing_spotify_info = {}
        spotify_keys = ['album', 'spotify_id', 'release_date', 'duration_ms', 
                       'spotify_artist_id', 'spotify_artist_en', 'spotify_popularity']
        for key in spotify_keys:
            if key in json_data:
                existing_spotify_info[key] = json_data[key]
        
        try:
            # HTMLからデータを抽出して、移調前のJSONを保存
            print("  HTMLデータを抽出して移調前のJSONを保存...")
            extract_result = extract_and_save_html_data(url, jtotal_path, existing_spotify_info)
            
            if not extract_result.get('success'):
                print(f"  ✗ HTMLデータの抽出に失敗しました")
                error_count += 1
                continue
            
            # 移調前のJSON保存が成功
            if extract_result.get('original_play_key'):
                print(f"  ✓ 移調前のJSONを保存しました（キー情報あり: {extract_result.get('original_play_key')}）")
            else:
                print(f"  ✓ 移調前のJSONを保存しました（キー情報なし）")
            success_count += 1
                
        except Exception as e:
            print(f"  ✗ エラーが発生しました: {e}")
            error_count += 1
    
    print("\n" + "=" * 50)
    print(f"処理完了:")
    print(f"  成功: {success_count} 件")
    print(f"  スキップ: {skip_count} 件")
    print(f"  エラー: {error_count} 件")
    print(f"  処理対象: {len(items_to_process)} 件")
    print(f"  全エントリ: {len(all_items)} 件")


if __name__ == "__main__":
    # コマンドライン引数から処理件数を取得（デフォルト: 100件）
    max_items = 100
    if len(sys.argv) > 1:
        try:
            max_items = int(sys.argv[1])
        except ValueError:
            print(f"警告: 無効な引数 '{sys.argv[1]}'。デフォルトの100件で実行します。")
            max_items = 100
    
    main(max_items=max_items)

