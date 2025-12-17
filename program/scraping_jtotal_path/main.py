#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
メイン処理
"""

import os
import json

from .scraper import scrape_paths_from_search_page
from .file_loader import load_existing_files

# 検索結果ページのURL
search_url = 'https://music.j-total.net/db/search.cgi?mode=search&page=1&sort=id_new&word=%89%CC&method=and'


def main(max_items=None):
    """メイン処理
    
    Args:
        max_items: 新規に取得する最大件数（Noneの場合は制限なし）
    """
    
    print(f"検索結果ページを取得中: {search_url}")
    
    output_dir = 'data/jtotal/json/path'
    os.makedirs(output_dir, exist_ok=True)
    
    # 既存のファイルを読み込む
    print("既存のファイルを読み込み中...")
    path_to_item = load_existing_files(output_dir)
    existing_count = len(path_to_item)
    print(f"既存のパス: {existing_count} 件")
    
    if max_items:
        print(f"新規取得の上限: {max_items} 件")
    
    # 既存のファイルから最後の番号を取得して、続きのページから開始
    json_files = [f for f in os.listdir(output_dir) if f.endswith('.json') and f[0].isdigit()]
    if json_files:
        json_files.sort(key=lambda x: int(x.split('-')[0]))
        last_file = json_files[-1]
        # ファイル名から最後の番号を取得（例: "25001-25835.json" -> 25835）
        last_num = int(last_file.split('-')[1].replace('.json', ''))
        # 25835件 ÷ 25件/ページ ≈ 1034ページから開始（余裕を持たせて少し前から）
        start_page = max(1, (last_num // 25) - 10)  # 10ページ前から開始（重複チェックのため）
        print(f"続きから取得を開始します（ページ {start_page} から、既存: {last_num} 件まで取得済み）")
    else:
        start_page = 1
        last_num = 0
    
    # すべてのページからパスを取得（複数ページがある場合）
    all_items = []  # {title, path, artist}のリスト
    page = start_page
    empty_page_count = 0  # 連続してパスが見つからないページのカウント
    
    while True:
        # ページ番号をURLに反映
        if page == 1:
            current_url = search_url
        else:
            # ページ番号を変更
            current_url = search_url.replace('page=1', f'page={page}')
        
        print(f"ページ {page} を処理中...")
        
        try:
            items = scrape_paths_from_search_page(current_url)
            
            if not items:
                # パスが見つからない場合、連続カウントを増やす
                empty_page_count += 1
                print(f"ページ {page}: パスが見つかりませんでした（連続{empty_page_count}回目）")
                # 連続して5回パスが見つからない場合は終了
                if empty_page_count >= 5:
                    print("連続して5回パスが見つからないため、処理を終了します")
                    break
                page += 1
                continue
            else:
                # パスが見つかった場合はカウントをリセット
                empty_page_count = 0
            
            # 重複を排除しながら追加（情報が空でない場合は更新）
            new_count = 0
            for item in items:
                path = item['path']
                if path not in path_to_item:
                    path_to_item[path] = item
                    new_count += 1
                else:
                    # 既存の情報を更新（空でない場合のみ）
                    existing = path_to_item[path]
                    if item.get('title') and not existing.get('title'):
                        existing['title'] = item['title']
                    if item.get('artist') and not existing.get('artist'):
                        existing['artist'] = item['artist']
            
            if new_count > 0:
                print(f"  新規: {new_count} 件、既存: {len(items) - new_count} 件")
            
            print(f"ページ {page}: {len(items)} 件のパスを取得")
            
            # 新規取得件数の上限をチェック
            current_new_count = len(path_to_item) - existing_count
            if max_items and current_new_count >= max_items:
                print(f"新規取得件数が上限 ({max_items} 件) に達したため、処理を終了します")
                break
            
            # 次のページがあるかチェック（簡易的な方法）
            # 実際のページネーション構造に応じて調整が必要な場合があります
            page += 1
            
            # 安全のため、最大ページ数を制限（全曲取得のため上限を上げる）
            # 28943件 ÷ 25件/ページ ≈ 1158ページ
            if page > 1200:  # 余裕を持たせて1200ページまで
                print("最大ページ数に到達したため、処理を終了します")
                break
                
        except Exception as e:
            print(f"ページ {page} の処理でエラー: {e}")
            break
    
    # パスでソート
    all_items = sorted(path_to_item.values(), key=lambda x: x['path'])
    
    total_count = len(all_items)
    new_count = total_count - existing_count
    print(f"\n合計 {total_count} 件のパスを取得しました（新規: {new_count} 件）")
    
    # JSONファイルに保存（1000曲ずつに分割）
    
    chunk_size = 1000
    total_files = 0
    
    for i in range(0, len(all_items), chunk_size):
        chunk = all_items[i:i + chunk_size]
        start_num = i + 1
        end_num = min(i + chunk_size, len(all_items))
        
        output_file = os.path.join(output_dir, f'{start_num}-{end_num}.json')
        
        # JSONデータを作成（[{title:"", artist:"", path:""}]形式）
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)
        
        print(f"パス一覧を保存しました: {output_file} ({len(chunk)} 件)")
        total_files += 1
    
    print(f"\n合計 {total_files} 個のファイルに分割して保存しました")
    
    # 最初の10件を表示
    print("\n取得したパスの例（最初の10件）:")
    for i, item in enumerate(all_items[:10], 1):
        title = item.get('title', '') if item.get('title') else '(タイトルなし)'
        artist = item.get('artist', '') if item.get('artist') else '(アーティストなし)'
        print(f"  {i}. {title} - {artist} - {item['path']}")


if __name__ == "__main__":
    main()

