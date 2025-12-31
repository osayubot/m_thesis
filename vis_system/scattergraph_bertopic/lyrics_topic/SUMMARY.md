# 実装完了サマリー

## 実装したもの

### 1. パッケージ構造

- `program/lyrics_topic/` パッケージを作成
- モジュール化された設計（config, data_loader, embedding, topic_model, mapping, output, utils, cli）

### 2. 設定管理

- `config.py`: 型安全な設定クラス（dataclass 使用）
- `config.yaml`: 設定ファイル例（すべてのパラメータを YAML で管理）
- 手動定義トピック（8 カテゴリ）を`config.py`に定義

### 3. データ処理

- `data_loader.py`:
  - JSON 読み込み（1 曲 1JSON、1 ファイル複数曲の両方に対応）
  - 重複フレーズの間引き
  - 短いフレーズの結合
  - 空文字・1 文字トークンの除去

### 4. Embedding 計算

- `embedding.py`:
  - SentenceTransformer "intfloat/multilingual-e5-base" を使用
  - E5 prefix ("query: {text}") の自動適用
  - Parquet 形式でのキャッシュ（50 万件対応）
  - GPU 自動検出

### 5. トピックモデル

- `topic_model.py`:
  - BERTopic パイプライン（UMAP + HDBSCAN + c-TF-IDF）
  - `reduce_topics`で目標トピック数（20）に削減
  - 各フレーズの上位 k 個の自由トピック確率を計算

### 6. トピックマッピング

- `mapping.py`:
  - 自由トピックの代表テキスト取得（top words + representative docs）
  - 自由トピック → 手動トピックの soft 確率マッピング
  - コサイン類似度 →softmax で確率化

### 7. 出力生成

- `output.py`:
  - `phrase_level.parquet`: フレーズ単位のトピック確率
  - `topic_info.csv`: トピック情報
  - `song_level.parquet`: 曲単位のトピック確率（フレーズの平均）
  - `evaluation.json`: 評価指標（-1 率、エントロピー分布等）

### 8. CLI

- `cli.py`:
  - `python -m vis_system.scattergraph3.lyrics_topic.cli train --input ... --out ... --config ...`
  - コマンドライン引数解析
  - エラーハンドリング

### 9. ドキュメント

- `README.md`: 使用方法、データ形式、設定方法
- `IMPLEMENTATION_NOTES.md`: 実装の詳細ポイント
- `SUMMARY.md`: このファイル

## 使用方法

### 基本的な実行

```bash
python -m vis_system.scattergraph3.lyrics_topic.cli train \
  --input data/analyzed \
  --out output/ \
  --config vis_system/scattergraph3/lyrics_topic/config.yaml
```

### 必要な依存関係

```bash
pip install bertopic sentence-transformers umap-learn hdbscan scikit-learn pandas pyarrow pyyaml torch
```

## 出力ファイル

1. **phrase_level.parquet**: 各フレーズのトピック確率
2. **topic_info.csv**: トピック情報（キーワード、代表ドキュメント、マッピング情報）
3. **song_level.parquet**: 各曲のトピック確率（フレーズの平均）
4. **evaluation.json**: 評価指標

## 重要な実装ポイント

1. **Embedding キャッシュ**: 50 万件のデータに対応するため、Parquet 形式でキャッシュ
2. **バッチ処理**: Embedding 計算はバッチ処理で実行
3. **重複処理**: 曲内で同一 lyric が 3 回以上出る場合は最大 2 回に間引く
4. **短いフレーズ結合**: 文字数<12 のフレーズは前後フレーズと結合
5. **Soft 確率マッピング**: 自由トピック → 手動トピックの確率マッピング
6. **再現性**: random_seed でシード固定
7. **BERTopic バージョン差分対応**: get_topic()の戻り値の形式差に対応

## 設定ファイルの主要パラメータ

- `embedding.model_name`: "intfloat/multilingual-e5-base"
- `hdbscan.min_cluster_size`: 120（データ規模に応じて調整）
- `topic_model.target_free_topics`: 20（reduce_topics 後の目標トピック数）
- `data_processing.min_phrase_length`: 12（これより短いフレーズは結合）
- `data_processing.max_duplicate_count`: 2（曲内で同一 lyric がこれより多い場合は間引く）

## 次のステップ

1. 実際のデータで実行して動作確認
2. パラメータの調整（特に`hdbscan.min_cluster_size`）
3. 評価指標の確認（-1 率、エントロピー分布等）
4. 必要に応じて可視化機能の追加
