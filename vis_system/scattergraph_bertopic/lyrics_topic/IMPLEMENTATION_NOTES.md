# 実装ノート

## フォルダ構成

```
vis_system/scattergraph3/lyrics_topic/
├── __init__.py          # パッケージ初期化
├── config.py            # 設定管理（Configクラス、MANUAL_TOPICS）
├── config.yaml          # 設定ファイル例
├── data_loader.py       # データ読み込みと前処理
├── embedding.py         # Embedding計算とキャッシュ
├── topic_model.py       # BERTopicモデルパイプライン
├── mapping.py           # 自由トピック→手動トピックマッピング
├── output.py            # 出力生成（phrase/song/topic/evaluation）
├── utils.py             # ユーティリティ関数
├── cli.py               # CLIエントリポイント
├── README.md            # 使用方法
└── IMPLEMENTATION_NOTES.md  # このファイル
```

## 重要な実装ポイント

### 1. データ前処理の流れ

```
JSON読み込み
  ↓
フレーズ抽出（analyzed_chord_progressions_and_lyricsから）
  ↓
重複除去（曲内で同一lyricがmax_duplicate_count回以上なら間引く）
  ↓
短いフレーズ結合（文字数<min_phrase_lengthなら前後combine_windowフレーズと結合）
  ↓
空文字・1文字除去
  ↓
前処理完了
```

### 2. Embedding キャッシュの仕組み

- テキストのハッシュ（件数と最初の 100 フレーズの総文字数）でキャッシュキーを生成
- Parquet 形式で保存（numpy 配列をリストのリストに変換）
- キャッシュヒット時は再計算をスキップ

### 3. BERTopic パイプライン

1. **初期学習**: `fit_transform(docs, embeddings=emb)` で学習
2. **トピック削減**: `reduce_topics(docs, topics, nr_topics=target)` で目標トピック数に削減
3. **確率取得**: `calculate_probabilities=True` の場合、各フレーズのトピック確率を取得
4. **上位 k 個抽出**: 各フレーズの上位 k 個のトピック確率のみを保存

### 4. Soft 確率マッピングの詳細

```
自由トピックの代表テキスト取得
  ↓
  - top_words（上位10語）
  - representative_docs（上位3件）
  - これらを結合して代表テキストとする
  ↓
代表テキストをembedding化
  ↓
手動トピックの説明文をembedding化
  ↓
コサイン類似度計算（自由トピック × 手動トピック）
  ↓
Softmaxで確率化（温度パラメータで調整可能）
  ↓
マッピング行列完成（n_free_topics × n_manual_topics）
  ↓
各フレーズの自由トピック確率で重み付け
  ↓
手動トピック確率を計算
```

### 5. 出力データの構造

#### phrase_level.parquet

- 各フレーズごとの行
- 自由トピック確率は上位 k 個のみ（JSON 文字列）
- 手動トピック確率は 8 カテゴリ分すべて（各カラム）

#### song_level.parquet

- 各曲ごとの行
- フレーズの手動トピック確率を平均化
- 主要トピック（argmax）を計算

#### topic_info.csv

- 各自由トピックごとの行
- キーワード、代表ドキュメント、マッピング情報

### 6. メモリ効率化

- Embedding は Parquet でキャッシュ（メモリマップ可能）
- バッチ処理で embedding 計算
- 確率配列は必要な部分のみ保存（上位 k 個）

### 7. 再現性の確保

- `random_seed`を設定ファイルで管理
- numpy、random、torch のシードを固定
- UMAP、HDBSCAN の random_state も固定

### 8. エラーハンドリング

- 空データや欠損キーがあっても落ちにくい設計
- 各ステップでログ出力
- 例外発生時は詳細なトレースバックを出力

## BERTopic バージョン差分への対応

### get_topic()の戻り値

```python
# 両方の形式に対応
topic_words = topic_model.get_topic(topic_id)
if isinstance(topic_words, pd.DataFrame):
    words = topic_words['Word'].tolist()
    scores = topic_words['Score'].tolist()
elif isinstance(topic_words, list):
    words = [w[0] for w in topic_words]
    scores = [w[1] for w in topic_words]
```

### reduce_topics()の引数

```python
# 新しいバージョンではdocsとtopicsを明示的に渡す
reduced_topics, reduced_probs = topic_model.reduce_topics(
    docs, topics, nr_topics=target_nr_topics
)
```

## パフォーマンス最適化のヒント

1. **GPU 使用**: `embedding.device: "cuda"` を設定（自動検出も可能）
2. **バッチサイズ**: GPU メモリに応じて調整（32→64→128）
3. **マルチプロセッシング**: JSON 読み込みは並列化可能（未実装）
4. **キャッシュ活用**: 同じデータの再計算を避ける

## 拡張可能性

- 他の embedding モデルへの対応（設定ファイルで変更可能）
- 他のトピックモデルへの対応（BERTopic 以外）
- 評価指標の追加（coherence、diversity 等）
- 可視化機能の追加（トピック分布、マッピング可視化等）
