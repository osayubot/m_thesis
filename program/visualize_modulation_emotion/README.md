# 転調前後の感情ベクトル可視化

転調前後のセクションの感情データを取得し、2D 空間に投影して矢印で可視化するモジュール。

## 機能

- 分析済みデータから転調イベントを自動検出
- 転調タイプの自動分類（半音上、全音上、関係調など）
- 転調前後の感情ベクトルを 2D 空間に投影（PCA または JOY-SADNESS 軸）
- 矢印で転調前 → 転調後の感情変化を可視化

## 使用方法

### 基本的な使用

```python
from program.visualize_modulation_emotion import (
    load_modulation_data,
    visualize_modulation_emotion_vectors,
)

# データを読み込む
events = load_modulation_data('data/analyzed')

# 可視化
visualize_modulation_emotion_vectors(
    events,
    'output/modulation_emotion_vectors.png',
    projection_method='pca',  # 'pca' または 'joy_sadness'
)
```

### コマンドラインから実行

```bash
python run_visualize_modulation_emotion.py data/analyzed vis_system/modulation_emotion/modulation_emotion_vectors.png
```

## 転調タイプの分類

- **semitone_up**: 半音上（例: C → C#）
- **tone_up**: 全音上（例: C → D）
- **semitone_down**: 半音下（例: C → B）
- **tone_down**: 全音下（例: C → Bb）
- **relative_major_minor**: 関係調（長調 ↔ 短調、例: C → Am）
- **fifth_related**: 5 度関係（例: C → G）
- **other**: その他

## 投影方法

### PCA 投影

感情ベクトルの分散を最大化する 2 次元空間に投影。主成分分析を使用。

### JOY-SADNESS 軸投影

JOY と SADNESS の 2 軸で直接投影。解釈が容易。

## 出力

- 転調前の点: 円形、小さめ
- 転調後の点: 四角形、大きめ
- 矢印: 転調前 → 転調後
- 色: 転調タイプごとに色分け

## 依存関係

- numpy
- matplotlib
- scikit-learn (PCA 投影を使用する場合)
