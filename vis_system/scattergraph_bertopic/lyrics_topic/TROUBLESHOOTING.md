# トラブルシューティングガイド

## 問題: トピック数が少なすぎる（2 個しかない）

### 原因

- `hdbscan.min_cluster_size`が大きすぎる
- データ規模に対して設定が適切でない

### 解決策

#### 1. min_cluster_size を調整

データ規模に応じて`min_cluster_size`を調整：

```yaml
# データ規模に応じた目安
# フレーズ数 < 1000: min_cluster_size = 10-20
# フレーズ数 1000-5000: min_cluster_size = 20-50
# フレーズ数 5000-20000: min_cluster_size = 50-100
# フレーズ数 > 20000: min_cluster_size = 100-200
```

**現在のデータ（約 1841 フレーズ）の場合：**

```yaml
hdbscan:
  min_cluster_size: 30 # 120から30に減らす
  min_samples: 3 # 5から3に減らす
```

#### 2. UMAP パラメータを調整

より細かいクラスタを検出するため：

```yaml
umap:
  n_neighbors: 15 # 20から15に減らす（局所構造を重視）
  min_dist: 0.05 # 0.1から0.05に減らす（クラスタを分離）
```

#### 3. データ前処理を見直す

```yaml
data_processing:
  min_phrase_length: 10 # 12から10に減らす（より多くのフレーズを保持）
```

### 実行方法

```bash
python -m vis_system.scattergraph3.lyrics_topic.cli train \
  --input data/analyzed \
  --out output/ \
  --config vis_system/scattergraph3/lyrics_topic/config_small.yaml
```

## 問題: コード進行が偏っている

### 原因

- データセット自体に偏りがある可能性
- トピックが少なすぎて、コード進行の多様性が失われている

### 解決策

#### 1. トピック数を増やす

上記の設定調整でトピック数を増やすことで、コード進行の多様性が反映されやすくなります。

#### 2. データの確認

- 入力データに偏りがないか確認
- 特定のアーティストや時代に偏っていないか確認

#### 3. コード進行のフィルタリング条件を見直す

- 4 コード進行のみに限定している場合、他の長さの進行も含める
- 最小コード数の条件を緩和する

## 問題: -1（ノイズ）が多すぎる

### 原因

- `min_cluster_size`が大きすぎる
- フレーズが短すぎる
- データの多様性が低い

### 解決策

```yaml
hdbscan:
  min_cluster_size: 20 # さらに小さくする
  min_samples: 2 # さらに小さくする

data_processing:
  min_phrase_length: 8 # さらに小さくする
  combine_window: 5 # 3から5に増やす（より多くのフレーズを結合）
```

## 問題: 実行時間が長すぎる

### 解決策

```yaml
embedding:
  batch_size: 64 # 32から64に増やす（メモリに余裕がある場合）

# または、データを制限
# --max-files 100 オプションを使用
```

## 問題: メモリ不足

### 解決策

```yaml
embedding:
  batch_size: 16 # 32から16に減らす

umap:
  low_memory: true # メモリ効率を優先
```

## 推奨設定の例

### 小規模データ（< 2000 フレーズ）

```yaml
hdbscan:
  min_cluster_size: 20-30
  min_samples: 2-3

umap:
  n_neighbors: 10-15
  min_dist: 0.05

topic_model:
  target_free_topics: 10-15
```

### 中規模データ（2000-10000 フレーズ）

```yaml
hdbscan:
  min_cluster_size: 40-60
  min_samples: 3-5

umap:
  n_neighbors: 15-20
  min_dist: 0.1

topic_model:
  target_free_topics: 15-20
```

### 大規模データ（> 10000 フレーズ）

```yaml
hdbscan:
  min_cluster_size: 80-120
  min_samples: 5-10

umap:
  n_neighbors: 20-30
  min_dist: 0.1

topic_model:
  target_free_topics: 20-30
```

## パラメータ調整のコツ

1. **段階的に調整**: 一度に大幅に変更せず、少しずつ調整
2. **データ規模を確認**: まずデータの規模を確認してから設定を決める
3. **試行錯誤**: 複数の設定で試して、最適な設定を見つける
4. **評価指標を確認**: `evaluation.json`の-1 率やエントロピーを確認
