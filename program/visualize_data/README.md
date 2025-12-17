# 音楽的レーベンシュタイン距離

コード進行の類似度計算におけるレーベンシュタイン距離の実装について説明します。

## 概要

この実装では、コード進行の類似度を計算するために、機能的な近さと循環性を考慮したレーベンシュタイン距離を使用しています。

## 重みづけの詳細

### 1. 置換コスト（機能的な近さに基づく）

コードの機能的な近さに基づいて置換コストを設定しています：

- **同じコード**: `0.0`
- **同機能（T-T, PD-PD, D-D）**: `0.2`
- **近傍機能（T-PD, PD-D）**: `0.5`
- **遠い関係（T-D）**: `0.8`
- **機能不明**: `1.0`

#### 機能分類

- **T (Tonic)**: `i`, `iii`, `vi`, `I`, `III`, `VI`
- **PD (Predominant)**: `ii`, `iv`, `II`, `IV`
- **D (Dominant)**: `v`, `vii`, `V`, `VII`, `V7`, `v7`

### 2. 削除・挿入コスト

- **繰り返しコード（同じコードが連続）**: `0.5`
- **通常の削除・挿入**: `0.7`

**注意**: 現在の実装では 4 つのコード進行のみを対象としているため、削除・挿入コストは通常発生しません（置換のみ）。

### 3. 文脈を考慮した調整

**現在は無効化されています**（`use_context=False`）。

以前の実装では、自然な進行（T→PD→D→T など）に対してコストを調整していましたが、現在は機能的な近さのみで評価しています。

### 4. 循環性の考慮

コード進行は循環的な性質を持っているため、`circular_distance()`関数で全回転を試して最小距離を返します。

**例**:

- 進行 1: `i → V → vi → ii`
- 進行 2: `V → vi → ii → i`（1 つ回転）

通常の距離では開始位置が違うだけで大きな距離になりますが、循環距離では距離`0.0`（同じ進行）とみなされます。

## 実装の詳細

### 主要な関数

#### `musical_levenshtein_distance(seq1, seq2, use_context=False)`

基本的なレーベンシュタイン距離を計算します。

- `seq1`, `seq2`: コード進行（ローマ数字のリスト）
- `use_context`: 文脈を考慮するか（現在は`False`が推奨）

#### `circular_distance(seq1, seq2)`

循環性を考慮した距離を計算します。内部で`musical_levenshtein_distance`を全回転に対して呼び出し、最小距離を返します。

#### `compute_distance_matrix(progressions)`

コード進行のリストから距離行列を計算します。

## 使用例

### 基本的な距離計算

```python
from musical_distance import musical_levenshtein_distance

seq1 = ["i", "vi", "ii", "V"]
seq2 = ["i", "iii", "ii", "V"]  # vi → iii（同機能 T-T）

dist = musical_levenshtein_distance(seq1, seq2, use_context=False)
# 結果: 0.2（同機能なのでコスト0.2）
```

### 循環距離の計算

```python
from musical_distance import circular_distance

seq1 = ["i", "V", "vi", "ii"]
seq2 = ["V", "vi", "ii", "i"]  # 1つ回転

dist = circular_distance(seq1, seq2)
# 結果: 0.0（同じ進行とみなされる）
```

### 距離行列の計算

```python
from musical_distance import compute_distance_matrix

progressions = [
    ["i", "V", "vi", "ii"],
    ["i", "vi", "ii", "V"],
    ["i", "iii", "ii", "V"],
]

dist_matrix = compute_distance_matrix(progressions)
# 結果: 3×3の距離行列
```

## コスト計算の仕組み

レーベンシュタイン距離は動的計画法（DP）で計算されます。各セルで以下の 3 つの選択肢から最小コストを選びます：

```python
dp[i][j] = min(
    dp[i-1][j] + del_cost,      # 削除コストを足す
    dp[i][j-1] + ins_cost,      # 挿入コストを足す
    dp[i-1][j-1] + sub_cost     # 置換コストを足す
)
```

コストは「足し算」で累積されますが、各ステップでは「最小コストのパス」を選びます。

### 計算例

`["i", "vi", "ii", "V"]` vs `["i", "iii", "ii", "V"]`の場合：

- `dp[0][0] = 0.0`
- `dp[1][1] = 0.0` (i→i, コスト 0.0)
- `dp[2][2] = 0.0 + 0.2 = 0.2` (vi→iii, コスト 0.2)
- `dp[3][3] = 0.2 + 0.0 = 0.2` (ii→ii, コスト 0.0)
- `dp[4][4] = 0.2 + 0.0 = 0.2` (V→V, コスト 0.0)

最終距離 = `0.2`

## 実際の使用状況

可視化システム（`mds_visualization.py`）では、`circular_distance()`が使用されています：

```python
from .musical_distance import circular_distance

dist = circular_distance(prog, ref_prog)
```

これにより、コード進行の循環性が考慮され、より正確な類似度計算が可能になります。

## 注意事項

1. **4 つのコード進行に固定**: 現在の実装では 4 つのコード進行のみを対象としているため、削除・挿入コストは通常発生しません。

2. **文脈調整は無効化**: 以前は自然な進行に対してコストを調整していましたが、現在は機能的な近さのみで評価しています。

3. **循環性の考慮**: 回転で同じ進行は同じとみなされます。これはコード進行の「パターン」としての類似性を測る目的には適切です。

## 関連ファイル

- `musical_distance.py`: 実装ファイル
- `mds_visualization.py`: 可視化システムでの使用例
