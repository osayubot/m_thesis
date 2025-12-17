# J-POP コード進行と歌詞感情の可視化分析システム

## 概要

本研究は、J-POP における**コード進行パターン**と**歌詞の感情表現**の関係性を分析・可視化するシステムです。多次元尺度構成法（MDS: Multidimensional Scaling）を用いてコード進行の類似性を 2 次元平面上にマッピングし、各点に歌詞から抽出した感情分布を円グラフとして表示します。

### 研究目的

1. J-POP におけるコード進行パターンと歌詞の感情表現の関係性を定量的に分析する
2. 「王道進行」「小室進行」「丸サ進行」などの定番コード進行を基準とした楽曲の分布を可視化する
3. 作曲家・作詞家・アーティストごとの特徴を分析できるインタラクティブな可視化ツールを開発する

---

## 理論的背景

### 1. コード進行の音楽理論

#### 1.1 度数表記（ローマ数字表記）

コード進行を分析する際、キーに依存しない**度数表記**（ローマ数字表記）を使用します。これにより、異なるキーの楽曲間でもコード進行パターンを比較できます。

| 度数 | メジャーキー（C） | 機能               |
| ---- | ----------------- | ------------------ |
| I    | C                 | トニック（安定）   |
| ii   | Dm                | サブドミナント     |
| iii  | Em                | トニック代理       |
| IV   | F                 | サブドミナント     |
| V    | G                 | ドミナント（緊張） |
| vi   | Am                | トニック代理       |
| vii° | Bdim              | ドミナント代理     |

#### 1.2 代表的なコード進行パターン

本システムでは以下の基準進行を使用しています：

| 名称         | 進行                   | 特徴                                        |
| ------------ | ---------------------- | ------------------------------------------- |
| **王道進行** | IV → V → iii → vi      | J-POP で最も頻出。感動的・切ない印象        |
| **小室進行** | vi → IV → V → I        | 90 年代に小室哲哉が多用。力強く前進する印象 |
| **丸サ進行** | IVM7 → III7 → vi7 → I7 | シティポップ系。おしゃれで都会的な印象      |

#### 1.3 コード進行間の距離計算

コード進行間の類似度は、**音楽的レーベンシュタイン距離**を用いて計算します。

```
音楽的距離 = 編集距離 + Σ(音程差コスト)
```

- **編集距離**: 挿入・削除・置換の最小操作回数
- **音程差コスト**: コード間の音程差（半音数）に基づくペナルティ
- **循環性**: コード進行の循環的な性質を考慮（例: I-V-vi-IV と vi-IV-I-V は同一視）

### 2. 感情分析の理論

#### 2.1 プルチックの感情の輪（Plutchik's Wheel of Emotions）

本システムでは、心理学者 Robert Plutchik が提唱した**8 つの基本感情**モデルを採用しています。

```
        ANTICIPATION（期待）
            ↑
    TRUST ←─┼─→ JOY（喜び）
  （信頼）   │
            │
    FEAR ←─┼─→ SURPRISE（驚き）
  （恐れ）   │
            │
   ANGER ←─┼─→ SADNESS（悲しみ）
  （怒り）   │
            ↓
        DISGUST（嫌悪）
```

| 感情                 | 色（本システム）    | 対極の感情   |
| -------------------- | ------------------- | ------------ |
| JOY（喜び）          | #FFFF73（黄色）     | SADNESS      |
| SADNESS（悲しみ）    | #5150F8（青）       | JOY          |
| ANTICIPATION（期待） | #F3AB63（オレンジ） | SURPRISE     |
| SURPRISE（驚き）     | #74BBF9（水色）     | ANTICIPATION |
| ANGER（怒り）        | #E93323（赤）       | FEAR         |
| FEAR（恐れ）         | #429429（緑）       | ANGER        |
| DISGUST（嫌悪）      | #EB60F8（紫）       | TRUST        |
| TRUST（信頼）        | #88FC6E（黄緑）     | DISGUST      |

#### 2.2 BERT による感情分類

歌詞の感情分析には、日本語感情分析用にファインチューニングされた**BERT モデル**を使用しています。

- **モデル**: `koshin2001/Japanese-to-emotions`
- **入力**: 歌詞テキスト（最大 512 トークン）
- **出力**: 8 感情のスコア（0.0〜1.0、シグモイド活性化）
- **特徴**: マルチラベル分類（複数の感情が同時に高くなりうる）

### 3. 多次元尺度構成法（MDS）

#### 3.1 MDS の原理

MDS（Multidimensional Scaling）は、高次元の距離データを低次元空間（通常 2 次元）に射影する手法です。

**目的関数（ストレス関数）**:

```
Stress = √(Σ(d_ij - δ_ij)² / Σd_ij²)
```

- `d_ij`: 低次元空間での点 i,j 間の距離
- `δ_ij`: 元の距離行列での点 i,j 間の距離

#### 3.2 基準進行ベース MDS

本システムでは、従来の MDS を拡張した**基準進行ベース MDS**を提案しています。

**手順**:

1. 各コード進行から基準進行（王道・小室・丸サ）への距離ベクトルを計算
2. 距離ベクトル間のユークリッド距離で距離行列を構築
3. MDS で 2 次元座標に射影

**利点**:

- 解釈可能性の向上（基準進行との関係が明確）
- 安定した配置（基準進行が固定点として機能）
- 比較分析の容易化（異なるデータセット間の比較が可能）

### 4. データフィルタリング条件

可視化の品質を確保するため、以下のフィルタリング条件を適用しています：

1. **コード進行長**: 4 コードの進行のみを採用（最も一般的な小節構造）
2. **感情閾値**: 最大感情スコアが 0.5 以上の歌詞のみを採用（ノイズ除去）

---

## システム構成

```
.
├── program/
│   ├── scraping_jtotal_path/   # J-Totalパス取得モジュール
│   ├── scraping_jtotal_data/   # J-Totalデータ取得モジュール
│   ├── scraping_ufret_data/    # U-FRETデータ取得モジュール
│   ├── combine_data/           # データ結合モジュール
│   ├── analyze_data/           # データ分析モジュール
│   │   ├── emotion_analysis.py # 感情分析（BERT）
│   │   ├── roman_numeral.py    # ローマ数字変換
│   │   ├── key_assignment.py   # キー推定
│   │   └── transposition.py    # 移調処理
│   └── visualize_data/         # データ可視化モジュール
│       ├── mds_visualization.py # MDS可視化
│       └── musical_distance.py  # 音楽的距離計算
├── data/
│   ├── jtotal/                 # J-Total Musicのデータ
│   ├── ufret/                  # U-FRETのデータ
│   ├── combined/               # 結合データ
│   └── analyzed/               # 分析済みデータ
├── vis_system/                 # 可視化Webアプリケーション
│   ├── index.html              # メインページ
│   └── data/                   # 可視化用JSONデータ
│       ├── mds_odo_pie_data.json     # 王道進行ベース
│       ├── mds_komuro_pie_data.json  # 小室進行ベース
│       └── mds_marusa_pie_data.json  # 丸サ進行ベース
├── run_*.py                    # 各処理のエントリーポイント
├── requirements.txt            # 依存パッケージ
└── .env                        # 環境変数（API認証情報）
```

---

## セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定

`.env`ファイルを作成し、Spotify API 認証情報を設定：

```bash
# .envファイル
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

Spotify API の認証情報は[Spotify Developer Dashboard](https://developer.spotify.com/dashboard)で取得できます。

### 3. Playwright のセットアップ

```bash
playwright install chromium
```

---

## 使用方法

### データ収集から可視化までの全体フロー

```bash
# 1. J-Totalパス取得
python run_scraping_jtotal_path.py [件数]

# 2. J-Totalデータ取得
python run_scraping_jtotal_data.py [件数]

# 3. U-FRETデータ取得
python run_scraping_ufret_data.py [件数]

# 4. データ結合
python run_combine_data.py

# 5. データ分析（感情分析・コード進行分析）
python run_analyze_data.py

# 6. MDS可視化データ生成
python run_visualize_data.py
```

### 可視化システムの起動

```bash
cd vis_system
python3 -m http.server 8000
```

ブラウザで `http://localhost:8000` を開く

---

## 可視化システムの機能

### インタラクティブ機能

| 機能               | 操作     | 説明                                   |
| ------------------ | -------- | -------------------------------------- |
| データセット切替   | セレクタ | 王道進行/小室進行/丸サ進行ベースを選択 |
| フィルタリング     | セレクタ | アーティスト/作曲家/作詞家で絞り込み   |
| リリース年フィルタ | 入力欄   | 年範囲で絞り込み                       |
| 人気度フィルタ     | 入力欄   | Spotify 人気度で絞り込み               |
| ツールチップ       | ホバー   | コード進行・感情分布を表示             |
| 歌詞表示           | クリック | 選択した感情の歌詞一覧を表示           |

### 円グラフの解釈

- **位置**: MDS 座標（類似したコード進行が近くに配置）
- **サイズ**: 該当する歌詞フレーズの数
- **色分け**: 感情の分布（プルチックの 8 感情）
- **黒枠**: 基準進行（王道・小室・丸サ）と一致する点

---

## データ形式

### 分析済みデータ（`data/analyzed/*.json`）

```json
{
  "title": "楽曲タイトル",
  "artist": "アーティスト名",
  "lyricist": "作詞者",
  "composer": "作曲者",
  "release_date": "2020-01-01",
  "spotify_popularity": 75,
  "original_play_key": "G",
  "analyzed_chord_progressions_and_lyrics": [
    {
      "chord_progression": ["G", "Em", "C", "D"],
      "normalized_chord_progression": ["G", "Em", "C", "D"],
      "key": "G",
      "lyric": "歌詞のフレーズ",
      "emotion": {
        "JOY": 0.123,
        "SADNESS": 0.456,
        "ANTICIPATION": 0.089,
        ...
      }
    }
  ]
}
```

### 可視化データ（`vis_system/data/*.json`）

```json
{
  "points": [
    {
      "x": 47.61,
      "y": 69.62,
      "r": 8.82,
      "pie": [
        {
          "label": "JOY",
          "c": "#FFFF73",
          "v": 75.0,
          "lyrics": [
            {
              "lyric": "歌詞フレーズ",
              "song": {
                "title": "曲名",
                "artist": "アーティスト",
                "release_date": "2020-01-01",
                "spotify_popularity": 75
              }
            }
          ]
        }
      ],
      "progression": "IV - V - iii - vi",
      "roman_progression": ["IV", "V", "iii", "vi"],
      "strokeColor": "#000000"
    }
  ]
}
```

---

## 技術スタック

| カテゴリ       | 技術                              |
| -------------- | --------------------------------- |
| データ収集     | Python, Playwright, BeautifulSoup |
| 感情分析       | PyTorch, Transformers (BERT)      |
| 可視化計算     | NumPy, scikit-learn (MDS)         |
| フロントエンド | HTML, CSS, JavaScript, Chart.js   |
| API            | Spotify Web API (spotipy)         |

---

## 参考文献

1. Plutchik, R. (1980). _Emotion: A Psychoevolutionary Synthesis_. Harper & Row.
2. Kruskal, J. B. (1964). Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis. _Psychometrika_, 29(1), 1-27.
3. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. _NAACL-HLT_.

---

## 注意事項

- J-Total Music、U-FRET の利用規約を遵守してください
- スクレイピングは適切な間隔を空けて実行してください
- 収集したデータは研究目的でのみ使用してください
- Spotify API 認証情報は`.env`ファイルで管理し、公開リポジトリにコミットしないでください

---

## ライセンス

本研究は修士論文のために開発されたものです。
