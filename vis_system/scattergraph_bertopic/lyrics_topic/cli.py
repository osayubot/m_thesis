"""
CLIエントリポイント
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import random

from .config import Config
from .data_loader import preprocess_data
from .embedding import EmbeddingCalculator
from .topic_model import TopicModelPipeline
from .mapping import TopicMapper
from .output import (
    create_phrase_level_output,
    create_topic_info_output,
    create_song_level_output,
    create_evaluation_output
)
from .utils import setup_logging, ensure_dir

logger = setup_logging()


def set_random_seed(seed: int):
    """再現性のためのシード設定"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def train(
    input_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    max_files: int | None = None
):
    """
    メインの学習処理
    
    Args:
        input_dir: 入力データディレクトリ
        output_dir: 出力ディレクトリ
        config_path: 設定ファイルパス
        max_files: 最大ファイル数（Noneなら無制限）
    """
    # 設定を読み込む
    logger.info(f"Loading config from {config_path}")
    config = Config.from_yaml(config_path)
    
    # シードを設定
    set_random_seed(config.random_seed)
    
    # ディレクトリを準備
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    cache_dir = Path(config.cache_dir)
    ensure_dir(cache_dir)
    
    # データの前処理
    logger.info("=" * 60)
    logger.info("Step 1: Data preprocessing")
    logger.info("=" * 60)
    phrases, song_metadata = preprocess_data(input_dir, config, max_files)
    
    if len(phrases) == 0:
        logger.error("No phrases extracted. Exiting.")
        return
    
    # テキストリストを取得
    texts = [p['text'] for p in phrases]
    
    # Embedding計算器を初期化
    logger.info("=" * 60)
    logger.info("Step 2: Embedding calculation")
    logger.info("=" * 60)
    embedding_calculator = EmbeddingCalculator(config.embedding, cache_dir)
    embeddings = embedding_calculator.compute_embeddings(texts, use_cache=True)
    
    # トピックモデルパイプラインを初期化
    logger.info("=" * 60)
    logger.info("Step 3: Topic modeling")
    logger.info("=" * 60)
    topic_model_pipeline = TopicModelPipeline(config, embedding_calculator)
    
    # BERTopicモデルを学習
    free_topics, free_topic_probs = topic_model_pipeline.fit(texts, embeddings)
    
    # トピック数を削減
    if config.topic_model.reduce_topics:
        logger.info("=" * 60)
        logger.info("Step 4: Topic reduction")
        logger.info("=" * 60)
        free_topics, free_topic_probs = topic_model_pipeline.reduce_topics(texts)
    
    # 上位k個の自由トピック確率を取得
    free_topic_probs_topk = topic_model_pipeline.get_top_k_probs(free_topic_probs)
    
    # トピックマッパーを初期化
    logger.info("=" * 60)
    logger.info("Step 5: Topic mapping")
    logger.info("=" * 60)
    topic_mapper = TopicMapper(config, embedding_calculator, topic_model_pipeline)
    
    # 自由トピックIDのリスト（-1を除く）
    unique_free_topics = sorted(set(t for t in free_topics if t != -1))
    
    # マッピング行列を計算
    mapping_matrix = topic_mapper.compute_mapping_matrix(unique_free_topics)
    
    # フレーズの手動トピック確率を計算
    manual_probs = topic_mapper.map_phrase_probs(free_topic_probs_topk, free_topics)
    
    # 出力を生成
    logger.info("=" * 60)
    logger.info("Step 6: Output generation")
    logger.info("=" * 60)
    
    # フレーズレベル出力
    phrase_output_path = output_dir / "phrase_level.parquet"
    create_phrase_level_output(
        phrases,
        free_topics,
        free_topic_probs_topk,
        manual_probs,
        phrase_output_path
    )
    
    # トピック情報出力
    topic_info_output_path = output_dir / "topic_info.csv"
    create_topic_info_output(
        topic_model_pipeline,
        topic_mapper,
        topic_info_output_path
    )
    
    # 曲レベル出力
    song_output_path = output_dir / "song_level.parquet"
    create_song_level_output(
        phrases,
        free_topics,
        manual_probs,
        song_metadata,
        song_output_path
    )
    
    # 評価出力
    eval_output_path = output_dir / "evaluation.json"
    create_evaluation_output(
        manual_probs,
        free_topics,
        eval_output_path
    )
    
    logger.info("=" * 60)
    logger.info("Completed!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"  - Phrase level: {phrase_output_path}")
    logger.info(f"  - Topic info: {topic_info_output_path}")
    logger.info(f"  - Song level: {song_output_path}")
    logger.info(f"  - Evaluation: {eval_output_path}")


def main():
    """CLIメイン関数"""
    parser = argparse.ArgumentParser(
        description="日本語歌詞データに特化したBERTopicパイプライン"
    )
    
    parser.add_argument(
        'command',
        choices=['train'],
        help='実行するコマンド'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='入力データディレクトリ'
    )
    
    parser.add_argument(
        '--out',
        type=str,
        default=None,
        help='出力ディレクトリ（デフォルト: vis_system/scattergraph_bertopic/data）'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='設定ファイルパス（デフォルト: lyrics_topic/config.yaml）'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='最大ファイル数（デフォルト: 無制限）'
    )
    
    args = parser.parse_args()
    
    # デフォルトの設定ファイルパス
    if args.config is None:
        # このファイルのディレクトリから相対パスで取得
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(args.config)
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # デフォルトの出力ディレクトリ
    if args.out is None:
        # プロジェクトルートから相対パスで取得
        project_root = Path(__file__).parent.parent.parent.parent
        output_dir = project_root / "vis_system" / "scattergraph_bertopic" / "data"
    else:
        output_dir = Path(args.out)
    
    if args.command == 'train':
        try:
            train(
                input_dir=args.input,
                output_dir=output_dir,
                config_path=config_path,
                max_files=args.max_files
            )
        except KeyboardInterrupt:
            logger.info("\n処理が中断されました。")
            sys.exit(1)
        except Exception as e:
            logger.error(f"エラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()

