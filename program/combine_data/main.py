#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
J-TotalとU-FRETのデータを結合する関数
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional


def load_jtotal_json_files(base_dir: str = 'data/jtotal/json/raw') -> Dict[str, Dict]:
    """
    J-TotalのJSONファイルを読み込んで、spotify_idをキーとした辞書を返す
    
    Args:
        base_dir: J-TotalのJSONファイルがあるディレクトリ
    
    Returns:
        {spotify_id: json_data}の辞書
    """
    jtotal_data = {}
    
    if not os.path.exists(base_dir):
        print(f"警告: {base_dir} ディレクトリが見つかりません")
        return jtotal_data
    
    # 再帰的にJSONファイルを検索
    json_files = glob.glob(os.path.join(base_dir, '**/*.json'), recursive=True)
    
    print(f"J-TotalのJSONファイルを読み込み中... ({len(json_files)} 件)")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                spotify_id = data.get('spotify_id')
                
                if spotify_id:
                    # spotify_idが既に存在する場合は、最初に見つかったものを優先
                    if spotify_id not in jtotal_data:
                        jtotal_data[spotify_id] = data
                else:
                    # spotify_idがない場合はスキップ
                    pass
        except Exception as e:
            print(f"警告: {json_file} の読み込みに失敗しました: {e}")
    
    print(f"J-Total: {len(jtotal_data)} 件のspotify_idが見つかりました")
    return jtotal_data


def load_ufret_json_files(base_dir: str = 'data/ufret/json/raw') -> Dict[str, Dict]:
    """
    U-FRETのJSONファイルを読み込んで、spotify_idをキーとした辞書を返す
    
    Args:
        base_dir: U-FRETのJSONファイルがあるディレクトリ
    
    Returns:
        {spotify_id: json_data}の辞書
    """
    ufret_data = {}
    
    if not os.path.exists(base_dir):
        print(f"警告: {base_dir} ディレクトリが見つかりません")
        return ufret_data
    
    # 再帰的にJSONファイルを検索（サブフォルダ内も含む）
    json_files = glob.glob(os.path.join(base_dir, '**/*.json'), recursive=True)
    
    print(f"U-FRETのJSONファイルを読み込み中... ({len(json_files)} 件)")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                spotify_id = data.get('spotify_id')
                
                if spotify_id:
                    # spotify_idが既に存在する場合は、最初に見つかったものを優先
                    if spotify_id not in ufret_data:
                        ufret_data[spotify_id] = data
                else:
                    # spotify_idがない場合はスキップ
                    pass
        except Exception as e:
            print(f"警告: {json_file} の読み込みに失敗しました: {e}")
    
    print(f"U-FRET: {len(ufret_data)} 件のspotify_idが見つかりました")
    return ufret_data


def combine_jtotal_and_ufret(
    jtotal_dir: str = 'data/jtotal/json/raw',
    ufret_dir: str = 'data/ufret/json/raw',
    output_dir: str = 'data/combined',
    save_individual: bool = True
) -> List[Dict]:
    """
    J-TotalとU-FRETのデータをspotify_idで結合する
    
    Args:
        jtotal_dir: J-TotalのJSONファイルがあるディレクトリ
        ufret_dir: U-FRETのJSONファイルがあるディレクトリ
        output_dir: 結合結果を保存するディレクトリ
        save_individual: 個別のファイルとして保存するか（Trueの場合、spotify_idごとに保存）
    
    Returns:
        結合されたデータのリスト
    """
    # データを読み込む
    jtotal_data = load_jtotal_json_files(jtotal_dir)
    ufret_data = load_ufret_json_files(ufret_dir)
    
    # spotify_idが一致するものを結合
    combined_data = []
    matched_spotify_ids = set()
    
    print("\nspotify_idでマッチング中...")
    
    for spotify_id in jtotal_data.keys():
        if spotify_id in ufret_data:
            jtotal_item = jtotal_data[spotify_id]
            ufret_item = ufret_data[spotify_id]
            
            # 結合データを作成（指定された順序で、重複はufret優先）
            combined_item = {
                'jtotal_path': jtotal_item.get('jtotal_path', ''),
                'ufret_id': ufret_item.get('ufret_id'),
                'title': ufret_item.get('title') or jtotal_item.get('title'),
                'artist': ufret_item.get('artist') or jtotal_item.get('artist'),
                'lyricist': ufret_item.get('lyricist') or jtotal_item.get('lyricist'),
                'composer': ufret_item.get('composer') or jtotal_item.get('composer'),
                'album': ufret_item.get('album') or jtotal_item.get('album'),
                'spotify_id': spotify_id,
                'release_date': ufret_item.get('release_date') or jtotal_item.get('release_date'),
                'duration_ms': ufret_item.get('duration_ms') or jtotal_item.get('duration_ms'),
                'spotify_artist_id': ufret_item.get('spotify_artist_id') or jtotal_item.get('spotify_artist_id'),
                'spotify_artist_en': ufret_item.get('spotify_artist_en') or jtotal_item.get('spotify_artist_en'),
                'spotify_popularity': ufret_item.get('spotify_popularity') or jtotal_item.get('spotify_popularity'),
            }
            
            # jtotal_original_keyとjtotal_original_play_key（あれば）
            if jtotal_item.get('original_key'):
                combined_item['jtotal_original_key'] = jtotal_item.get('original_key')
            if jtotal_item.get('original_play_key'):
                combined_item['jtotal_original_play_key'] = jtotal_item.get('original_play_key')

            # ufret_original_keyとufret_capo（あれば、0の値も含む）
            if 'ufret_original_key' in ufret_item:
                combined_item['ufret_original_key'] = ufret_item.get('ufret_original_key')
            if 'ufret_capo' in ufret_item:
                combined_item['ufret_capo'] = ufret_item.get('ufret_capo')  
            
            # コード進行と歌詞
            combined_item['jtotal_chord_progressions_and_lyrics'] = jtotal_item.get('jtotal_chord_progressions_and_lyrics')
            combined_item['ufret_chord_progressions_and_lyrics'] = ufret_item.get('ufret_chord_progressions_and_lyrics')
            
            combined_data.append(combined_item)
            matched_spotify_ids.add(spotify_id)
    
    print(f"マッチしたspotify_id: {len(matched_spotify_ids)} 件")
    
    # 出力ディレクトリを作成
    if save_individual or combined_data:
        os.makedirs(output_dir, exist_ok=True)
    
    # 個別ファイルとして保存
    if save_individual:
        print(f"\n結合データを個別ファイルとして保存中... ({output_dir})")
        for item in combined_data:
            spotify_id = item['spotify_id']
            output_file = os.path.join(output_dir, f"{spotify_id}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(item, f, ensure_ascii=False, indent=2)
        print(f"  {len(combined_data)} 件のファイルを保存しました")
    
    # 全件を1つのファイルに保存
    if combined_data:
        output_file = os.path.join(output_dir, 'all_combined.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)
        print(f"  全件を {output_file} に保存しました")
    
    return combined_data


def main():
    """メイン処理"""
    import sys
    
    # コマンドライン引数からディレクトリを指定可能
    jtotal_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/jtotal/json/raw'
    ufret_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/ufret/json/raw'
    output_dir = sys.argv[3] if len(sys.argv) > 3 else 'data/combined'
    
    print("=" * 50)
    print("J-TotalとU-FRETのデータを結合します")
    print("=" * 50)
    print(f"J-Totalディレクトリ: {jtotal_dir}")
    print(f"U-FRETディレクトリ: {ufret_dir}")
    print(f"出力ディレクトリ: {output_dir}")
    print("=" * 50)
    
    combined_data = combine_jtotal_and_ufret(
        jtotal_dir=jtotal_dir,
        ufret_dir=ufret_dir,
        output_dir=output_dir,
        save_individual=True
    )
    
    print("\n" + "=" * 50)
    print(f"処理完了: {len(combined_data)} 件のデータを結合しました")
    print("=" * 50)


if __name__ == "__main__":
    main()

