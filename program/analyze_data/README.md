# analyze_data モジュール

音楽データの分析・処理を行うモジュールです。コード進行からキー（調）を推定し、転調を検出し、セクションごとにキーを割り当てる機能を提供します。

## 概要

このモジュールは以下の主要な機能を提供します：

1. **キー推定**: コード進行からキー（調）を機械学習で推定
2. **転調検出**: 楽曲内のキー変更（転調）を検出
3. **キー割り当て**: セクションごとにキーを割り当て
4. **評価**: キー推定モデルの性能を評価
5. **感情分析**: 歌詞から感情を分析（BERT モデル使用）

## ファイル構成

```
analyze_data/
├── __init__.py
├── main.py                    # メイン処理
├── data_extraction.py         # データ抽出関数
├── chord_normalization.py     # コード・キー正規化
├── roman_numeral.py           # ローマ数字変換
├── transposition.py           # 移調処理
├── modulation.py              # 転調検出
├── key_assignment.py          # キー割り当て
├── evaluation.py              # 評価関数
├── emotion_analysis.py        # 感情分析
└── file_utils.py              # ファイル操作
```

## 主要機能

### 1. データ抽出 (`data_extraction.py`)

コード進行データを抽出・正規化します。

#### 主要関数

- **`extract_jtotal_chords(song: dict) -> List[str]`**

  - J-Total のコード進行を抽出・正規化
  - セクションを横断してフラットなリストを返す

- **`extract_ufret_chords(song: dict) -> List[str]`**

  - U-FRET のコード進行を抽出・正規化
  - セクションを横断してフラットなリストを返す

- **`extract_jtotal_chords_with_section_spans(song: dict)`**

  - J-Total のコード進行を抽出し、セクションごとのスパン情報も返す
  - 戻り値: `(chords_norm, section_spans, section_norm_counts)`

- **`extract_ufret_chords_with_section_spans(song: dict)`**

  - U-FRET のコード進行を抽出し、セクションごとのスパン情報も返す

- **`root_hist_12(chords: List[str]) -> np.ndarray`**

  - コード進行から 12 音のルート音ヒストグラムを計算（12 次元ベクトル）

- **`load_dataset(json_dir: str, recursive: bool = False)`**
  - JSON ファイルからデータセットを読み込み
  - 戻り値: `(texts_all, texts_last, X_root, y)`
    - `texts_all`: 全コード進行のテキスト表現
    - `texts_last`: 最後の部分のコード進行テキスト
    - `X_root`: ルート音ヒストグラム（12 次元）
    - `y`: キーラベル

### 2. コード・キー正規化 (`chord_normalization.py`)

コード名とキー名を正規化します（統一フォーマットに変換）。

#### 主要関数

- **`normalize_chord(ch: str) -> Optional[str]`**

  - コード名を正規化（例: `"C#"` → `"Db"`, `"Am"` → `"Am"`）
  - シャープをフラットに統一

- **`normalize_root(root: str) -> Optional[str]`**

  - ルート音を正規化（12 音階のいずれかに統一）

- **`normalize_key_label(k: str) -> Optional[str]`**

  - キー名を正規化（例: `"C#"` → `"Db"`, `"Am"` → `"Am"`）

- **`split_key_label(k: str) -> Tuple[Optional[str], bool]`**
  - キー名をルート音とマイナー/メジャーフラグに分割
  - 例: `"Am"` → `("A", True)`, `"C"` → `("C", False)`

#### 定数

- **`ROOTS_12`**: 12 音階のルート音リスト（フラット表記）

  ```python
  ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
  ```

- **`ROOT_TO_IDX`**: ルート音からインデックスへのマッピング

### 3. ローマ数字変換 (`roman_numeral.py`)

コード進行をローマ数字（度数）に変換します。

#### 主要関数

- **`degree_in_key(root: str, key_root: str, is_minor_key: bool)`**

  - コードのルート音をキーに対する度数（ローマ数字）に変換
  - 戻り値: `(roman_base, quality)`
    - `roman_base`: `"I"`, `"IV"`, `"bVII"` など
    - `quality`: `"m"`, `"M"`, `"dim"` など

- **`chord_to_roman(chord_raw: str, key_label: str)`**

  - コード名をローマ数字に変換
  - 例: `"G"` in key `"C"` → `"V"`

- **`section_to_roman_progression(section: dict, fallback_key: str = None)`**
  - セクションのコード進行をローマ数字のリストに変換

### 4. 移調処理 (`transposition.py`)

コード進行間の移調関係を推定・処理します。

#### 主要関数

- **`estimate_transposition_shift(jt_chords: List[str], uf_chords: List[str], N: int = 12) -> Optional[int]`**

  - J-Total と U-FRET のコード進行間の移調シフトを推定
  - 最初の N 個のコードの半音差から最も頻繁な差を返す

- **`transpose_key(key: str, semitone: int) -> Optional[str]`**

  - キーを指定した半音数だけ移調
  - 例: `transpose_key("C", 7)` → `"G"`

- **`semitone_diff(r1: str, r2: str) -> Optional[int]`**

  - 2 つのルート音間の半音差を計算（-11〜+11）

- **`lyric_initial_match(jt_sec: dict, uf_sec: dict) -> bool`**
  - J-Total と U-FRET のセクションの歌詞の最初の文字が一致するかチェック
  - データ統合の品質指標として使用

### 5. 転調検出 (`modulation.py`)

楽曲内のキー変更（転調）を検出します。

#### アルゴリズム

1. **スライディングウィンドウ**: コード進行をウィンドウ（デフォルト 16 コード）で分割
2. **特徴量抽出**: 各ウィンドウから TF-IDF ベクトルとルート音ヒストグラムを抽出
3. **キー確率計算**: 学習済み分類器で各ウィンドウのキー確率を計算
4. **Viterbi アルゴリズム**: HMM で時系列的に平滑化（キー変更の頻度を抑制）

#### 主要関数

- **`sliding_window_probs(chords_norm, vec_all, clf, W=16, H=4)`**

  - スライディングウィンドウで各ウィンドウのキー確率を計算
  - 戻り値: `(probs, spans)`
    - `probs`: (T, K) 各ウィンドウのキー確率行列
    - `spans`: 各ウィンドウのコード位置 `[(start, end), ...]`

- **`viterbi_hmm(log_emission, switch_penalty=4.0)`**

  - Viterbi アルゴリズムで最適なキー列を推定
  - `switch_penalty`: キー変更のペナルティ（大きいほど変更を抑制）

- **`detect_modulations(state_path, spans, classes, min_run_windows=3)`**

  - キー列から転調ポイントを検出
  - `min_run_windows`: 転調とみなす最小連続ウィンドウ数

- **`modulation_analysis_for_song(song_json, vec_all, clf, ...)`**

  - 楽曲全体の転調分析を実行
  - 戻り値: 転調情報を含む辞書

- **`make_window_features(norm_chords, vec_all)`**
  - ウィンドウの特徴量ベクトルを作成
  - TF-IDF ベクトル + ルート音ヒストグラム（12 次元）

### 6. キー割り当て (`key_assignment.py`)

セクションごとにキーを割り当てます。

#### 主要関数

- **`assign_keys_to_ufret_sections(song_json, vec_all, clf, W=16, H=4, switch_penalty=4.0)`**

  - U-FRET セクションにキーを割り当て
  - スライディングウィンドウ + Viterbi でキー列を推定し、セクションごとに投票

- **`assign_keys_to_jtotal_sections(song_json, vec_all, clf, ...)`**

  - J-Total セクションにキーを割り当て

- **`assign_keys_with_probabilities(song_json, vec_all, clf, ...)`**

  - 確率平均方式でキーを割り当て
  - セクション内のウィンドウ確率を重み付き平均

- **`assign_keys_to_ufret_with_transposition(song_json, vec_all, clf, ...)`**
  - 移調補正を考慮して U-FRET セクションにキーを割り当て
  - J-Total のキー情報を移調して U-FRET に適用

### 7. 評価 (`evaluation.py`)

キー推定モデルの性能を評価します。

#### 主要関数

- **`evaluate_with_cv(texts_all, texts_last, X_root, y, n_splits=3, random_state=42)`**

  - クロスバリデーションでモデルを評価
  - 評価指標:
    - **Top-1 Accuracy**: 完全一致の精度
    - **Top-3 Accuracy**: 上位 3 候補に正解が含まれる割合
    - **Close-1 Accuracy**: 音楽的に近いキーを正解とみなした精度
    - **Close-Top3 Accuracy**: 上位 3 候補に音楽的に近いキーが含まれる割合

- **`is_musically_close(true_k: str, pred_k: str) -> bool`**

  - 予測キーが正解キーと音楽的に近いかを判定
  - 判定条件:
    - 完全一致
    - 平行調（例: `C` ↔ `Am`）
    - 同主調（例: `C` ↔ `Cm`）
    - 五度圏の隣接（例: `C` ↔ `G` / `F`）

- **`top_k_accuracy(y_true, y_pred_proba, clf_classes, k=3)`**

  - Top-k 精度を計算
  - 予測確率の上位 k 個に正解が含まれれば正解とみなす

- **`fifth_neighbors(root: str)`**
  - 五度圏の隣接キー（属調・下属調）を返す
  - 例: `C` → `{G, F}`

### 8. 感情分析 (`emotion_analysis.py`)

歌詞から感情を分析します（BERT モデル使用）。

#### 主要関数

- **`analyze_emotion(lyric: str) -> Optional[Dict[str, float]]`**

  - 歌詞から 8 種類の感情スコアを計算
  - 使用モデル: `koshin2001/Japanese-to-emotions`
  - 感情ラベル: `JOY`, `SADNESS`, `ANTICIPATION`, `SURPRISE`, `ANGER`, `FEAR`, `DISGUST`, `TRUST`
  - 戻り値: 各感情のスコア（0.0〜1.0）の辞書

- **`add_emotion_to_sections(sections: list) -> list`**
  - セクションリストの各セクションに感情分析を追加

### 9. ファイル操作 (`file_utils.py`)

分析結果の保存・読み込みを行います。

#### 主要関数

- **`save_song_with_keys(song_json, output_dir: str, create_subdirs: bool = True)`**

  - キー情報を含む楽曲データを保存
  - `analyzed_chord_progressions_and_lyrics` フィールドを保存

- **`process_and_save_songs_with_keys(input_dir, output_dir, vec_all, clf, ...)`**

  - 複数の楽曲を一括処理して保存
  - 戻り値: 処理統計 `{"processed": int, "skipped": int, "errors": int}`

- **`print_modulation_log(song, res, context=6, use_ufret=True)`**
  - 転調分析結果をログ出力

### 10. メイン処理 (`main.py`)

全体の処理フローを実行します。

#### 処理の流れ

1. **データセット読み込み**: `data/combined/` から JSON ファイルを読み込み
2. **クロスバリデーション評価**: キー推定モデルの性能を評価
3. **モデル学習**: 全データでモデルを学習
4. **転調分析例**: 転調がある楽曲の例を表示
5. **キー割り当て例**: セクションごとのキー割り当て例を表示
6. **一括処理**: 全楽曲を処理して `data/analyzed/` に保存

## 使用方法

### 基本的な使い方

```bash
python -m program.analyze_data.main
```

または

```python
from program.analyze_data import main
main.main()
```

### 個別機能の使用例

#### キー推定

```python
from program.analyze_data.data_extraction import load_dataset
from program.analyze_data.evaluation import evaluate_with_cv

texts_all, texts_last, X_root, y = load_dataset("data/combined", recursive=True)
evaluate_with_cv(texts_all, texts_last, X_root, y, n_splits=3)
```

#### 転調検出

```python
from program.analyze_data.modulation import modulation_analysis_for_song
from program.analyze_data.data_extraction import extract_ufret_chords

# モデルを学習済みとして仮定
res = modulation_analysis_for_song(
    song_json, vec_all, clf,
    W=16, H=4,
    switch_penalty=4.0,
    min_run_windows=3,
    use_ufret=True
)

print(f"転調数: {len(res['modulations'])}")
for mod in res['modulations']:
    print(f"{mod['from_key']} → {mod['to_key']} at chord {mod['at_chord_index']}")
```

#### キー割り当て

```python
from program.analyze_data.key_assignment import assign_keys_to_ufret_sections

analyzed_sections = assign_keys_to_ufret_sections(
    song_json, vec_all, clf,
    W=16, H=4, switch_penalty=4.0
)

for sec in analyzed_sections:
    print(f"Key: {sec['key']}, Confidence: {sec['key_confidence']}")
```

#### 感情分析

```python
from program.analyze_data.emotion_analysis import analyze_emotion

lyric = "今日もいい天気だね"
emotions = analyze_emotion(lyric)
print(emotions)  # {'JOY': 0.8, 'SADNESS': 0.1, ...}
```

## アルゴリズムの詳細

### キー推定アルゴリズム

1. **特徴量抽出**:

   - TF-IDF ベクトル: コード進行をテキストとして扱い、TF-IDF でベクトル化
   - ルート音ヒストグラム: 12 音のルート音の分布（12 次元）

2. **分類器**: LogisticRegression（`solver="saga"`, `C=4.0`）

3. **スライディングウィンドウ**:

   - ウィンドウサイズ: 16 コード（デフォルト）
   - ステップサイズ: 4 コード（デフォルト）

4. **Viterbi アルゴリズム**:
   - 同じキーを維持: コスト 0
   - キーを変更: コスト `switch_penalty`（デフォルト 4.0）
   - 頻繁なキー変更を抑制

### 評価指標

- **Top-1 Accuracy**: 完全一致の精度
- **Top-3 Accuracy**: 上位 3 候補に正解が含まれる割合
- **Close-1 Accuracy**: 音楽的に近いキーを正解とみなした精度
  - 平行調、同主調、五度圏の隣接を考慮
- **Close-Top3 Accuracy**: 上位 3 候補に音楽的に近いキーが含まれる割合

## 依存パッケージ

- `numpy`: 数値計算
- `scipy`: スパース行列操作
- `scikit-learn`: 機械学習（TfidfVectorizer, LogisticRegression, StratifiedKFold）
- `torch`: PyTorch（感情分析用）
- `transformers`: Hugging Face Transformers（感情分析用）

## データ形式

### 入力データ（`data/combined/`）

```json
{
  "jtotal_path": "001a/002_arashi/051",
  "title": "楽曲タイトル",
  "artist": "アーティスト名",
  "jtotal_original_play_key": "G",
  "jtotal_chord_progressions_and_lyrics": [
    {
      "chord_progression": ["G", "Em", "C", "D"],
      "lyric": "歌詞"
    }
  ],
  "ufret_chord_progressions_and_lyrics": [
    {
      "chord_progression": ["G", "Em", "C", "D"],
      "lyric": "歌詞"
    }
  ]
}
```

### 出力データ（`data/analyzed/`）

```json
{
  "title": "楽曲タイトル",
  "artist": "アーティスト名",
  "ufret_play_key": "G",
  "analyzed_chord_progressions_and_lyrics": [
    {
      "chord_progression": ["G", "Em", "C", "D"],
      "lyric": "歌詞",
      "key": "G",
      "key_confidence": 0.95,
      "emotion": {
        "JOY": 0.8,
        "SADNESS": 0.1,
        ...
      }
    }
  ]
}
```

## 注意事項

- キー推定には十分なコード数（最低 12 コード以上）が必要
- 転調検出にはより多くのコード（40 コード以上推奨）が必要
- 感情分析には PyTorch と Transformers ライブラリが必要
- モデルの学習には時間がかかる場合がある

## 関連モジュール

- `scraping_jtotal_data`: J-Total データ取得
- `scraping_ufret_data`: U-FRET データ取得
- `combine_data`: データ結合
- `visualize_data`: データ可視化
