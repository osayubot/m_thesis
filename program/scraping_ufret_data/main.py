#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
U-FRETデータ取得のメイン処理
"""

import time
import sys
from .scraper import MusicScraper
from .file_utils import save_to_json, is_file_exists

def main(max_items=500, start_id=1, end_id=188318):
    """
    メイン処理
    
    Args:
        max_items: 処理する最大件数（デフォルト: 500）
        start_id: 開始ID（デフォルト: 1）
        end_id: 終了ID（デフォルト: 151620）
    """
    # IDの範囲を生成
    # all_chord_ids = list(range(start_id, end_id + 1))  # 全件処理用（コメントアウト）
    all_chord_ids = list(range(start_id, min(start_id + 151620, end_id + 1)))  # とりあえず50000件までに制限
    
    # 既に処理済みのファイルをスキップ
    processed_ids = set()
    for chord_id in all_chord_ids:
        if is_file_exists(chord_id):
            processed_ids.add(chord_id)
    
    print(f"合計 {len(all_chord_ids)} 件のIDがあります")
    print(f"既に処理済み: {len(processed_ids)} 件")
    
    # 未処理のIDのみをフィルタリング
    unprocessed_ids = [chord_id for chord_id in all_chord_ids if chord_id not in processed_ids]
    
    # 処理件数の制限を適用
    # if max_items is not None:
    ids_to_process = unprocessed_ids[:max_items]
    print(f"処理対象: {len(ids_to_process)} 件（上限: {max_items} 件）")
    # else:
    #     ids_to_process = unprocessed_ids
    #     print(f"処理対象: 全 {len(ids_to_process)} 件（未処理のみ）")
            
    if not ids_to_process:
        print("処理するデータがありません。")
        return
    
    print("-" * 50)
    
    scraper = MusicScraper()
    scraper.start_driver()
    
    # 統計情報
    success_count = 0
    skip_count = 0
    error_count = 0
    start_time = time.time()
    times = []  # 各曲の処理時間を記録
    
    try:
        for i, chord_id in enumerate(ids_to_process, 1):
            url = f"https://www.ufret.jp/song.php?data={chord_id}"
            
            try:
                song_start_time = time.time()
                print(f"\n[{i}/{len(ids_to_process)}] ID {chord_id} のデータを取得中...")
                
                result = scraper.get_music_data(url, chord_id)
                
                if result:
                    if save_to_json(result, chord_id):
                        song_time = time.time() - song_start_time
                        times.append(song_time)
                        avg_time = sum(times) / len(times) if times else 0
                        remaining = len(ids_to_process) - i
                        estimated_remaining_time = avg_time * remaining
                        
                        print(f"  ✓ データを {chord_id}.json に保存しました。")
                        print(f"  処理時間: {song_time:.2f}秒 | 平均: {avg_time:.2f}秒/曲 | 残り時間見積もり: {estimated_remaining_time/60:.1f}分")
                        success_count += 1
                    else:
                        error_count += 1
                else:
                    print("  ✗ データの取得に失敗しました。")
                    error_count += 1
                    
            except Exception as e:
                print(f"  ✗ ID {chord_id} の処理中にエラーが発生しました: {str(e)}")
                error_count += 1
                continue
            
    finally:
        scraper.close_driver()
        
        # 最終統計
        total_time = time.time() - start_time
        print("\n" + "=" * 50)
        print(f"処理完了:")
        print(f"  成功: {success_count} 件")
        print(f"  スキップ: {skip_count} 件")
        print(f"  エラー: {error_count} 件")
        print(f"  処理対象: {len(ids_to_process)} 件")
        print(f"  全ID: {len(all_chord_ids)} 件")
        print(f"  総処理時間: {total_time/60:.1f}分 ({total_time:.1f}秒)")
        if times:
            print(f"  平均処理時間: {sum(times)/len(times):.2f}秒/曲")
            # print(f"  全{len(all_chord_ids)}件の見積もり: {sum(times)/len(times) * len(all_chord_ids) / 3600:.1f}時間")  # 全件処理用（コメントアウト）

if __name__ == "__main__":
    # コマンドライン引数から処理件数を取得
    max_items = 1000  # デフォルト1000曲
    start_id = 1
    end_id = 188318
    
    if len(sys.argv) > 1:
        try:
            max_items = int(sys.argv[1])
        except ValueError:
            print(f"警告: 無効な引数 '{sys.argv[1]}'。デフォルトの500件で実行します。")
            max_items = 500
    
    # if len(sys.argv) > 2:  # 全件処理用（コメントアウト）
    #     try:
    #         start_id = int(sys.argv[2])
    #     except ValueError:
    #         print(f"警告: 無効な開始ID '{sys.argv[2]}'。デフォルトの1から開始します。")
    #         start_id = 1
    
    # if len(sys.argv) > 3:  # 全件処理用（コメントアウト）
    #     try:
    #         end_id = int(sys.argv[3])
    #     except ValueError:
    #         print(f"警告: 無効な終了ID '{sys.argv[3]}'。デフォルトの188318まで処理します。")
    #         end_id = 188318
    
    main(max_items=max_items, start_id=start_id, end_id=end_id)