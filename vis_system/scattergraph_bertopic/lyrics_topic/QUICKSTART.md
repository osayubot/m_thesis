# クイックスタートガイド

## 起動方法

### 1. 基本的な実行

プロジェクトのルートディレクトリ（`/Users/kai/Desktop/大学院/修士論文`）から実行：

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

### 2. オプション付きの実行例

#### 最大ファイル数を制限する場合

```bash
python -m vis_system.scattergraph3.lyrics_topic.cli train \
  --input data/analyzed \
  --out output/ \
  --config vis_system/scattergraph3/lyrics_topic/config.yaml \
  --max-files 100
```

#### カスタム設定ファイルを使用する場合

```bash
python -m vis_system.scattergraph3.lyrics_topic.cli train \
  --input data/analyzed \
  --out output/ \
  --config my_custom_config.yaml
```

### 3. 実行前の確認事項

1. **依存関係のインストール**

   ```bash
   pip install bertopic sentence-transformers umap-learn hdbscan scikit-learn pandas pyarrow pyyaml torch
   ```

2. **入力データの確認**

   - `data/analyzed/` ディレクトリに JSON ファイルが存在することを確認
   - JSON ファイルは以下の形式である必要があります：
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
           "key": "C"
         }
       ]
     }
     ```

3. **出力ディレクトリの確認**
   - デフォルトでは `vis_system/scattergraph3/data/` に出力されます
   - ディレクトリが存在しない場合は自動的に作成されます

### 4. 実行中のログ

実行中は以下のようなログが表示されます：

```
============================================================
Step 1: Data preprocessing
============================================================
Loading 1000 JSON files from data/analyzed
Extracted 50000 phrases from 1000 songs
Removed 500 duplicate phrases
Combined short phrases: 50000 -> 49500
Preprocessing completed: 49500 phrases from 1000 songs

============================================================
Step 2: Embedding calculation
============================================================
Computing embeddings for 49500 texts...
Computed embeddings: shape (49500, 768)

============================================================
Step 3: Topic modeling
============================================================
Fitting BERTopic model on 49500 texts...
Initial topics: 45 (noise ratio: 15.2%)

============================================================
Step 4: Topic reduction
============================================================
Reducing topics to 20...
Reduced topics: 20 (noise ratio: 12.5%)

============================================================
Step 5: Topic mapping
============================================================
Computing free topic → manual topic mapping matrix...
Computed mapping matrix: shape (20, 8)

============================================================
Step 6: Output generation
============================================================
Creating phrase-level output...
Saved phrase-level output: output/phrase_level.parquet (49500 phrases)
Creating topic info output...
Saved topic info output: output/topic_info.csv (20 topics)
Creating song-level output...
Saved song-level output: output/song_level.parquet (1000 songs)
Creating evaluation output...
Saved evaluation output: output/evaluation.json
```

### 5. 出力ファイル

実行が完了すると、`vis_system/scattergraph3/data/` ディレクトリに以下のファイルが生成されます：

- **phrase_level.parquet**: 各フレーズのトピック確率
- **topic_info.csv**: トピック情報（キーワード、代表ドキュメント、マッピング情報）
- **song_level.parquet**: 各曲のトピック確率（フレーズの平均）
- **evaluation.json**: 評価指標（-1 率、エントロピー分布等）

### 6. トラブルシューティング

#### メモリ不足エラー

- `config.yaml`の`embedding.batch_size`を小さくする（例: 16）
- `config.yaml`の`hdbscan.min_cluster_size`を大きくする（トピック数を減らす）

#### -1（ノイズ）が多すぎる

- `config.yaml`の`data_processing.min_phrase_length`を小さくする
- `config.yaml`の`data_processing.combine_window`を大きくする
- `config.yaml`の`hdbscan.min_cluster_size`を小さくする

#### 実行時間が長すぎる

- `--max-files`オプションでファイル数を制限する
- `config.yaml`の`embedding.batch_size`を大きくする（メモリに余裕がある場合）

### 7. 設定ファイルのカスタマイズ

`vis_system/scattergraph3/lyrics_topic/config.yaml`を編集して、パラメータを調整できます。

主要なパラメータ：

- `hdbscan.min_cluster_size`: トピック数の調整（大きいほどトピック数が減る）
- `topic_model.target_free_topics`: 削減後の目標トピック数
- `data_processing.min_phrase_length`: 短いフレーズ結合の閾値
- `embedding.batch_size`: Embedding 計算のバッチサイズ
