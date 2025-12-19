"""
感情ごとのコード進行可視化データ生成メイン処理
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

from .emotion_data_extractor import (
    load_analyzed_data,
    extract_emotion_based_data,
    get_emotion_statistics
)
from .json_generator import main as generate_json_main


def main(
    data_dir: str,
    output_dir: str,
    max_files: Optional[int] = None,
    methods: list[str] = ['mds', 'umap', 'tsne'],
    max_items_per_emotion: Optional[int] = 10000
) -> None:
    """
    メイン処理
    
    Args:
        data_dir: 分析済みデータのディレクトリ
        output_dir: 出力ディレクトリ
        max_files: 最大ファイル数（Noneの場合は全て）
        methods: 使用する次元削減手法のリスト
        max_items_per_emotion: 感情ごとの最大アイテム数（メモリ節約のため、Noneの場合は制限なし）
    """
    print("=" * 60)
    print("感情ごとのコード進行可視化データ生成")
    print("=" * 60)
    print(f"データディレクトリ: {data_dir}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"最大ファイル数: {max_files if max_files else '無制限'}")
    print(f"手法: {', '.join(methods)}")
    print(f"感情ごとの最大アイテム数: {max_items_per_emotion if max_items_per_emotion else '無制限'}")
    print("=" * 60)
    
    # データを読み込む
    print("\nデータを読み込んでいます...")
    songs = load_analyzed_data(data_dir, max_files)
    print(f"読み込み完了: {len(songs)}曲")
    
    # 感情ごとにデータを抽出
    print("\n感情ごとにデータを抽出しています...")
    emotion_data = extract_emotion_based_data(songs)
    
    # 統計情報を表示
    stats = get_emotion_statistics(emotion_data)
    print("\n各感情のデータ数:")
    for emotion, count in stats.items():
        print(f"  {emotion}: {count}")
    
    # JSONファイルを生成
    print("\nJSONファイルを生成しています...")
    generate_json_main(emotion_data, output_dir, methods, max_items_per_emotion)
    
    print("\n" + "=" * 60)
    print("完了!")
    print("=" * 60)
