# 日本語歌詞データに特化した BERTopic パイプライン

## 概要

日本語歌詞データ（約 50 万フレーズ）に対して BERTopic を使用したトピック分析を行い、自由トピックから手動定義 8 カテゴリへのマッピングを実装したパイプラインです。

## 機能

1. **データ前処理**

   - JSON ファイルの読み込み（1 曲 1JSON、1 ファイル複数曲の両方に対応）
   - 重複フレーズの間引き（曲内で同一 lyric が 3 回以上出る場合は最大 2 回に）
   - 短いフレーズの結合（文字数<12 の場合は前後フレーズと結合）
   - 空文字・1 文字トークンの除去

2. **Embedding 計算とキャッシュ**

   - SentenceTransformer "intfloat/multilingual-e5-base" を使用
   - E5 prefix ("query: {text}") の自動適用
   - Parquet 形式での embedding キャッシュ（50 万件対応）

3. **BERTopic モデル**

   - 自由トピックの生成（初期 40〜80 程度）
   - `reduce_topics`で最終的に 20 トピックに収束
   - 各フレーズの上位 k 個の自由トピック確率を計算

4. **トピックマッピング**

   - 自由トピック → 手動 8 カテゴリへの soft 確率マッピング
   - 自由トピックの代表テキストと手動トピックの説明文を embedding 化
   - コサイン類似度 →softmax で確率分布に変換

5. **出力**
   - `phrase_level.parquet`: フレーズ単位のトピック確率
   - `topic_info.csv`: トピック情報（キーワード、代表ドキュメント等）
   - `song_level.parquet`: 曲単位のトピック確率（フレーズの平均）
   - `evaluation.json`: 評価指標（-1 率、エントロピー分布等）

## インストール

```bash
pip install bertopic sentence-transformers umap-learn hdbscan scikit-learn pandas pyarrow pyyaml torch
```

## 使用方法

### 基本的な使い方

```bash
python -m vis_system.scattergraph3.lyrics_topic.cli train \
  --input data/analyzed \
  --out vis_system/scattergraph3/data \
  --config vis_system/scattergraph3/lyrics_topic/config.yaml
```

または、デフォルトの出力ディレクトリを使用する場合：

```bash
python -m vis_system.scattergraph3.lyrics_topic.cli train \
  --input data/analyzed \
  --config vis_system/scattergraph3/lyrics_topic/config.yaml
```

### オプション

- `--input`: 入力データディレクトリ（JSON ファイルが含まれる）
- `--out`: 出力ディレクトリ（デフォルト: `vis_system/scattergraph3/data`）
- `--config`: 設定ファイルパス（デフォルト: `vis_system/scattergraph3/lyrics_topic/config.yaml`）
- `--max-files`: 最大ファイル数（デフォルト: 無制限）

### 設定ファイル

`config.yaml`で以下のパラメータを調整可能：

- **embedding**: モデル名、バッチサイズ、デバイス等
- **umap**: n_neighbors, n_components, min_dist 等
- **hdbscan**: min_cluster_size, min_samples 等
- **vectorizer**: ngram_range, min_df, max_df, stop_words 等
- **data_processing**: 重複除去、短いフレーズ結合の設定
- **topic_model**: reduce_topics の目標トピック数、top_k 等
- **mapping**: softmax の温度パラメータ等

## データ形式

### 入力 JSON 形式

各 JSON ファイルは以下の形式を想定：

```json
{
  "ufret_id": "12345",
  "title": "曲名",
  "artist": "アーティスト名",
  "analyzed_chord_progressions_and_lyrics": [
    {
      "lyric": "歌詞フレーズ",
      "chord_progression": ["C", "Am", "F", "G"],
      "normalized_chord_progression": ["C", "Am", "F", "G"],
      "key": "C",
      "emotion": {...}
    },
    ...
  ]
}
```

### 出力形式

#### phrase_level.parquet

- `song_id`: 曲 ID
- `phrase_id`: フレーズ ID
- `text`: 歌詞テキスト
- `free_topic_id`: 自由トピック ID
- `free_topic_probs`: 上位 k 個の自由トピック確率（JSON 文字列）
- `manual_prob_{id}_{name}`: 各手動トピックの確率
- `metadata`: メタデータ（JSON 文字列）

#### song_level.parquet

- `song_id`: 曲 ID
- `title`, `artist`: 曲情報
- `n_phrases`: フレーズ数
- `dominant_manual_topic_id`: 主要な手動トピック ID
- `manual_prob_{id}_{name}`: 各手動トピックの平均確率
- `entropy`: 確率分布のエントロピー

## 手動定義トピック（8 カテゴリ）

0. **恋愛**: 恋愛、好き、愛、恋、想い、あなた、君、告白、デート、付き合う、恋人、彼氏、彼女、片思い、両想い
1. **別れ／未練**: 別れ、別れる、さよなら、未練、後悔、涙、泣く、つらい、悲しい、終わり、去る、離れる、破局
2. **夢・未来・応援**: 夢、未来、希望、応援、頑張る、努力、目標、夢中、叶う、チャレンジ、前進、成長、エール
3. **日常／等身大**: 日常、普通、平凡、生活、朝、昼、夕方、何気ない、ありふれた、リアル、現実、日常会話
4. **夜・都会**: 夜、都会、街、夜景、ネオン、都市、ビル、夜更かし、夜道、繁華街、都会の生活、夜の街
5. **季節（夏・冬）**: 夏、冬、季節、暑い、寒い、雪、雨、太陽、花火、祭り、クリスマス、夏休み、冬休み
6. **内省／孤独**: 孤独、一人、内省、考える、静か、寂しい、思い出す、過去、記憶、自分、心、内面、独り
7. **前進／決意**: 前進、決意、決める、変わる、新しい、スタート、始まり、挑戦、覚悟、決断、進む、歩く

## 実装のポイント

### 1. Embedding キャッシュ

50 万件のデータを扱うため、embedding は Parquet 形式でキャッシュします。テキストのハッシュを計算してキャッシュキーとし、同じデータの再計算を避けます。

### 2. バッチ処理

Embedding 計算はバッチ処理で実行し、メモリ効率を向上させます。デフォルトバッチサイズは 32 ですが、GPU メモリに応じて調整可能です。

### 3. 重複フレーズの処理

曲内で同一 lyric が 3 回以上出現する場合（サビ等）、最初の 2 回のみを保持します。これにより、サビの過剰な影響を抑制します。

### 4. 短いフレーズの結合

文字数が 12 未満のフレーズは、前後 3 フレーズの範囲で結合を試みます。これにより、-1（アウトライヤー）の割合を減らします。

### 5. Soft 確率マッピング

自由トピック → 手動トピックのマッピングは、代表テキストと説明文の embedding 類似度を softmax で確率化します。各フレーズの自由トピック確率を重みとして、手動トピック確率を計算します。

### 6. 再現性

`random_seed`を設定ファイルで管理し、numpy、random、torch のシードを固定します。

## BERTopic バージョン差分への対応

BERTopic のバージョンによって、`get_topic()`の戻り値が異なる場合があります：

- 古いバージョン: `[(word, score), ...]` のリスト
- 新しいバージョン: `pd.DataFrame` または `[(word, score), ...]`

本実装では両方の形式に対応しています。

## トラブルシューティング

### メモリ不足

- `embedding.batch_size`を小さくする（例: 16）
- `hdbscan.min_cluster_size`を大きくする（トピック数を減らす）

### -1（ノイズ）が多すぎる

- `data_processing.min_phrase_length`を小さくする
- `data_processing.combine_window`を大きくする
- `hdbscan.min_cluster_size`を小さくする

### トピック数が多すぎる/少なすぎる

- `hdbscan.min_cluster_size`を調整
- `topic_model.target_free_topics`を調整

## ライセンス

（プロジェクトのライセンスに従う）
