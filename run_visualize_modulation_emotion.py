#!/usr/bin/env python3
"""
転調前後の感情ベクトル可視化のエントリーポイント
"""
import sys
import argparse
from pathlib import Path

# プロジェクトルートをパスに追加
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from program.visualize_modulation_emotion import (
    load_modulation_data,
    visualize_modulation_emotion_vectors,
)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize emotion vector changes before and after modulation'
    )
    parser.add_argument(
        'data_dir',
        type=str,
        help='Path to analyzed data directory (e.g., data/analyzed)'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='vis_system/modulation_emotion/modulation_emotion_vectors.png',
        help='Output image path (default: vis_system/modulation_emotion/modulation_emotion_vectors.png)'
    )
    parser.add_argument(
        '--projection',
        '-p',
        type=str,
        choices=['pca', 'joy_sadness'],
        default='pca',
        help='Projection method: pca or joy_sadness (default: pca)'
    )
    parser.add_argument(
        '--max-arrows',
        type=int,
        default=None,
        help='Maximum number of arrows to display (default: all)'
    )
    
    args = parser.parse_args()
    
    # 出力ディレクトリを作成
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("転調前後の感情ベクトル可視化")
    print(f"データディレクトリ: {args.data_dir}")
    print(f"出力パス: {args.output}")
    print(f"投影方法: {args.projection}")
    print()
    
    # データを読み込む
    print("転調データを読み込み中...")
    events = load_modulation_data(args.data_dir)
    
    if not events:
        print("転調イベントが見つかりませんでした。")
        return
    
    # 可視化
    print("可視化中...")
    visualize_modulation_emotion_vectors(
        events,
        str(output_path),
        projection_method=args.projection,
        max_arrows=args.max_arrows,
    )
    
    print("完了しました！")


if __name__ == '__main__':
    main()

