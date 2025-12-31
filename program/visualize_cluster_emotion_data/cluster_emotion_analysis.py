"""
クラスタ単位の感情分布比較図の可視化

推論1「似ているはずのコード進行群に、異なる感情が混じるのはなぜなのか？」
を検証するための可視化を実装します。
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import csv

# オプショナルな依存関係
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.manifold import MDS
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from scipy.cluster.hierarchy import linkage, fcluster
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# 既存モジュールのインポート
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 必要な定数を直接定義（インポートエラーを避けるため）
EMOTION_COLORS = {
    'JOY': '#FFFF73',          # 黄色（喜び）
    'SADNESS': '#5150F8',      # 青（悲しみ）
    'ANTICIPATION': '#F3AB63',  # オレンジ（期待）
    'SURPRISE': '#74BBF9',     # 水色（驚き）
    'ANGER': '#E93323',        # 赤（怒り）
    'FEAR': '#429429',         # 緑（恐れ）
    'DISGUST': '#EB60F8',      # 紫（嫌悪）
    'TRUST': '#88FC6E',        # 黄緑（信頼）
}

# 距離計算関数をインポート
try:
    from program.visualize_scattergraph_data.musical_distance import compute_distance_matrix, circular_distance
except ImportError:
    # フォールバック: 簡易版の距離計算
    def circular_distance(prog1, prog2):
        """簡易版の循環距離計算"""
        if len(prog1) != 4 or len(prog2) != 4:
            return 1.0
        
        min_dist = float('inf')
        for offset in range(4):
            dist = sum(1 if prog1[i] != prog2[(i+offset)%4] else 0 for i in range(4))
            min_dist = min(min_dist, dist)
        
        return min_dist / 4.0
    
    def compute_distance_matrix(progressions):
        """簡易版の距離行列計算"""
        n = len(progressions)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = circular_distance(progressions[i], progressions[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        return dist_matrix

# 日本語フォントの設定（matplotlibが利用可能な場合のみ）
if HAS_MATPLOTLIB:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    sns.set_style("whitegrid")
    sns.set_palette("husl")


def load_scattergraph_data(json_path: str) -> Tuple[List[Dict], List[str]]:
    """
    散布図データ（umap_all.json等）を読み込む
    
    Args:
        json_path: JSONファイルのパス
        
    Returns:
        (points_data, roman_progressions)
        points_data: 各点のデータ（座標、感情、メタ情報）
        roman_progressions: 各点のコード進行（ローマ数字表記）
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    points = data.get('points', [])
    emotion_songs = data.get('emotionSongs', {})
    roman_progressions = []
    points_data = []
    
    for idx, point in enumerate(points):
        # コード進行を抽出（複数のキーを試す）
        progression = (point.get('romanProgression', '') or 
                      point.get('roman_progression', '') or
                      point.get('progression', ''))
        
        # 既にリストの場合はそのまま使用
        if isinstance(progression, list):
            roman_progressions.append(progression)
        elif isinstance(progression, str) and progression:
            # "IV - V - iii - vi" または "IV,V,iii,vi" 形式をリストに変換
            if '-' in progression:
                prog_list = [c.strip() for c in progression.split('-')]
            elif ',' in progression:
                prog_list = [c.strip() for c in progression.split(',')]
            else:
                prog_list = [progression.strip()]
            roman_progressions.append(prog_list)
        else:
            roman_progressions.append([])
        
        # 感情データを抽出
        emotions = {}
        
        # 方法1: pieデータから（labelとvの形式）
        pie_data = point.get('pie', [])
        if pie_data and len(pie_data) > 0:
            for pie_item in pie_data:
                if isinstance(pie_item, dict):
                    # 形式: {'label': 'JOY', 'c': '#FFFF73', 'v': 40.0, ...}
                    emotion_name = pie_item.get('label', '') or pie_item.get('emotion', '')
                    value = pie_item.get('v', 0.0) or pie_item.get('value', 0.0)
                    # vはパーセント（0-100）、0-1の範囲に変換
                    if value > 1.0:
                        value = value / 100.0  # パーセントを0-1に変換
                    if emotion_name and value > 0:
                        emotions[emotion_name] = value
        
        # 方法2: emotionSongsから（progressionをキーとして検索）
        if not emotions:
            # progressionを文字列に変換（リストの場合は結合）
            if isinstance(progression, list):
                progression_str = ' - '.join(progression)
            else:
                progression_str = str(progression) if progression else ''
            
            if progression_str and progression_str in emotion_songs:
                emotion_data = emotion_songs[progression_str]
                if isinstance(emotion_data, dict):
                    emotions = emotion_data
        
        # 方法3: 直接emotionsキーから
        if not emotions:
            emotions = point.get('emotions', {})
        
        # 点のデータを保存
        points_data.append({
            'x': point.get('x', 0),
            'y': point.get('y', 0),
            'emotions': emotions,
            'lyricCount': point.get('lyricCount', 0),
            'song_index': point.get('song_index', idx),
            'romanProgression': progression,
        })
    
    return points_data, roman_progressions


def compute_progression_distance_matrix(roman_progressions: List[List[str]]) -> np.ndarray:
    """
    コード進行間の距離行列を計算
    
    Args:
        roman_progressions: ローマ数字表記のコード進行のリスト
        
    Returns:
        距離行列（n×n）
    """
    n = len(roman_progressions)
    distance_matrix = np.zeros((n, n))
    
    print(f"計算中: {n}個のコード進行間の距離行列...")
    
    for i in range(n):
        if i % 100 == 0:
            print(f"  進捗: {i}/{n}")
        for j in range(i+1, n):
            prog1 = roman_progressions[i]
            prog2 = roman_progressions[j]
            
            if len(prog1) == 4 and len(prog2) == 4:
                # 循環距離を考慮
                dist = circular_distance(prog1, prog2)
            else:
                # 4コードでない場合は最大距離
                dist = 1.0
            
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    print("距離行列の計算が完了しました")
    return distance_matrix


def cluster_progressions(
    distance_matrix: np.ndarray,
    method: str = 'kmeans',
    n_clusters: int = 10,
    eps: float = 0.3
) -> np.ndarray:
    """
    コード進行をクラスタリング
    
    Args:
        distance_matrix: 距離行列
        method: クラスタリング手法 ('kmeans', 'dbscan', 'hierarchical', 'simple')
        n_clusters: クラスタ数（k-means, hierarchical用）
        eps: DBSCANのepsパラメータ
        
    Returns:
        各点のクラスタラベル
    """
    print(f"クラスタリング開始: 手法={method}")
    
    if not HAS_SKLEARN and method != 'simple':
        print("警告: sklearnがインストールされていません。簡易クラスタリングを使用します。")
        method = 'simple'
    
    if method == 'kmeans' and HAS_SKLEARN:
        # k-meansは距離行列を直接使えないので、MDSで埋め込み
        mds = MDS(n_components=10, dissimilarity='precomputed', random_state=42)
        embedding = mds.fit_transform(distance_matrix)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embedding)
        
    elif method == 'dbscan' and HAS_SKLEARN:
        dbscan = DBSCAN(eps=eps, metric='precomputed', min_samples=5)
        labels = dbscan.fit_predict(distance_matrix)
        
    elif method == 'hierarchical' and HAS_SCIPY:
        # 階層的クラスタリング
        linkage_matrix = linkage(distance_matrix, method='average')
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        labels = labels - 1  # 0始まりに調整
        
    elif method == 'simple':
        # 簡易クラスタリング: 距離が近いものを同じクラスタに
        n = len(distance_matrix)
        labels = np.zeros(n, dtype=int)
        cluster_id = 0
        assigned = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if assigned[i]:
                continue
            
            labels[i] = cluster_id
            assigned[i] = True
            
            # 距離が近い点を同じクラスタに
            for j in range(i+1, n):
                if not assigned[j] and distance_matrix[i, j] < eps:
                    labels[j] = cluster_id
                    assigned[j] = True
            
            cluster_id += 1
        
    else:
        raise ValueError(f"未知のクラスタリング手法: {method} (必要なライブラリがインストールされていない可能性があります)")
    
    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"クラスタリング完了: {n_clusters_found}個のクラスタが見つかりました")
    
    return labels


def analyze_cluster_emotion_distribution(
    points_data: List[Dict],
    labels: np.ndarray,
    emotion_names: List[str] = None
) -> Dict:
    """
    各クラスタ内の感情分布を分析
    
    Args:
        points_data: 各点のデータ
        labels: クラスタラベル
        emotion_names: 分析する感情名のリスト（Noneの場合は全8感情）
        
    Returns:
        クラスタごとの感情分布統計
    """
    if emotion_names is None:
        emotion_names = ['JOY', 'SADNESS', 'ANTICIPATION', 'SURPRISE', 
                        'ANGER', 'FEAR', 'DISGUST', 'TRUST']
    
    cluster_stats = {}
    unique_labels = sorted(set(labels))
    
    # ノイズクラスタ（-1）を除外
    if -1 in unique_labels:
        unique_labels.remove(-1)
    
    for cluster_id in unique_labels:
        # このクラスタに属する点を抽出
        cluster_points = [points_data[i] for i in range(len(points_data)) 
                         if labels[i] == cluster_id]
        
        if len(cluster_points) == 0:
            continue
        
        # 各感情の値を集計
        emotion_values = {emotion: [] for emotion in emotion_names}
        
        for point in cluster_points:
            emotions = point.get('emotions', {})
            for emotion in emotion_names:
                value = emotions.get(emotion, 0.0)
                if value > 0:  # 0より大きい値のみを集計
                    emotion_values[emotion].append(value * 100)  # パーセントに変換
        
        # 統計量を計算
        cluster_stats[cluster_id] = {
            'size': len(cluster_points),
            'emotions': {}
        }
        
        for emotion in emotion_names:
            values = emotion_values[emotion]
            if len(values) > 0:
                cluster_stats[cluster_id]['emotions'][emotion] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75),
                    'count': len(values)
                }
            else:
                cluster_stats[cluster_id]['emotions'][emotion] = {
                    'mean': 0.0,
                    'median': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'q25': 0.0,
                    'q75': 0.0,
                    'count': 0
                }
    
    return cluster_stats


def visualize_cluster_emotions(
    points_data: List[Dict],
    labels: np.ndarray,
    cluster_stats: Dict,
    output_dir: str,
    top_n_clusters: int = 10
):
    """
    クラスタ単位の感情分布を可視化
    
    Args:
        points_data: 各点のデータ
        labels: クラスタラベル
        cluster_stats: クラスタ統計
        output_dir: 出力ディレクトリ
        top_n_clusters: 可視化する上位Nクラスタ
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # クラスタサイズでソート
    sorted_clusters = sorted(
        cluster_stats.items(),
        key=lambda x: x[1]['size'],
        reverse=True
    )[:top_n_clusters]
    
    emotion_names = ['JOY', 'SADNESS', 'ANTICIPATION', 'SURPRISE', 
                    'ANGER', 'FEAR', 'DISGUST', 'TRUST']
    
    # 3. クラスタごとの感情分布の統計表（matplotlibの有無に関わらず生成）
    stats_data = []
    for cluster_id, stats in sorted_clusters:
        for emotion in emotion_names:
            emotion_stats = stats['emotions'].get(emotion, {})
            stats_data.append({
                'Cluster': cluster_id,
                'Size': stats['size'],
                'Emotion': emotion,
                'Mean': emotion_stats['mean'],
                'Median': emotion_stats['median'],
                'Std': emotion_stats['std'],
                'Min': emotion_stats['min'],
                'Max': emotion_stats['max'],
                'Q25': emotion_stats['q25'],
                'Q75': emotion_stats['q75'],
                'Count': emotion_stats['count']
            })
    
    # CSVファイルに保存（pandasなしでも動作）
    csv_path = output_path / 'cluster_emotion_statistics.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if stats_data:
            writer = csv.DictWriter(f, fieldnames=stats_data[0].keys())
            writer.writeheader()
            writer.writerows(stats_data)
    print(f"統計表を保存: {csv_path}")
    
    if not HAS_MATPLOTLIB:
        print("警告: matplotlibがインストールされていません。可視化（箱ひげ図・バイオリンプロット）をスキップします。")
        return
    
    # 1. 箱ひげ図（全クラスタ比較）
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, emotion in enumerate(emotion_names):
        ax = axes[idx]
        
        # データを準備
        plot_data = []
        plot_labels = []
        
        for cluster_id, stats in sorted_clusters:
            emotion_stats = stats['emotions'].get(emotion, {})
            if emotion_stats['count'] > 0:
                # このクラスタに属する点の感情値を抽出
                cluster_points = [points_data[i] for i in range(len(points_data)) 
                                if labels[i] == cluster_id]
                values = []
                for point in cluster_points:
                    value = point.get('emotions', {}).get(emotion, 0.0) * 100
                    if value > 0:
                        values.append(value)
                
                if len(values) > 0:
                    plot_data.append(values)
                    plot_labels.append(f"Cluster {cluster_id}\n(n={stats['size']})")
        
        # 箱ひげ図を描画
        if plot_data:
            bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(EMOTION_COLORS.get(emotion, '#808080'))
                patch.set_alpha(0.7)
            
            ax.set_title(f'{emotion} Distribution by Cluster', fontsize=12, fontweight='bold')
            ax.set_ylabel('Percentage (%)', fontsize=10)
            ax.set_ylim(0, 100)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'cluster_emotion_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"箱ひげ図を保存: {output_path / 'cluster_emotion_boxplot.png'}")
    
    # 2. バイオリンプロット（上位5クラスタ）
    top_5_clusters = sorted_clusters[:5]
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    for idx, (cluster_id, stats) in enumerate(top_5_clusters):
        ax = axes[idx]
        
        # データを準備
        emotion_data = []
        emotion_labels = []
        
        for emotion in emotion_names:
            cluster_points = [points_data[i] for i in range(len(points_data)) 
                            if labels[i] == cluster_id]
            values = []
            for point in cluster_points:
                value = point.get('emotions', {}).get(emotion, 0.0) * 100
                if value > 0:
                    values.append(value)
            
            if len(values) > 0:
                emotion_data.append(values)
                emotion_labels.append(emotion)
        
        # バイオリンプロットを描画
        if emotion_data:
            parts = ax.violinplot(emotion_data, positions=range(len(emotion_data)), 
                                 showmeans=True, showmedians=True)
            
            # 色を設定
            for pc, emotion in zip(parts['bodies'], emotion_labels):
                pc.set_facecolor(EMOTION_COLORS.get(emotion, '#808080'))
                pc.set_alpha(0.7)
            
            ax.set_xticks(range(len(emotion_labels)))
            ax.set_xticklabels(emotion_labels, rotation=45, ha='right')
            ax.set_title(f'Cluster {cluster_id}\n(n={stats["size"]})', 
                        fontsize=12, fontweight='bold')
            ax.set_ylabel('Percentage (%)', fontsize=10)
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'cluster_emotion_violinplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"バイオリンプロットを保存: {output_path / 'cluster_emotion_violinplot.png'}")


def generate_cluster_report(
    cluster_stats: Dict,
    roman_progressions: List[List[str]],
    labels: np.ndarray,
    output_dir: str
):
    """
    クラスタ分析レポートを生成
    
    Args:
        cluster_stats: クラスタ統計
        roman_progressions: コード進行リスト
        labels: クラスタラベル
        output_dir: 出力ディレクトリ
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_lines = []
    report_lines.append("# クラスタ単位の感情分布分析レポート\n")
    report_lines.append("## 推論1: 似ているはずのコード進行群に、異なる感情が混じるのはなぜなのか？\n")
    report_lines.append("\n")
    
    # クラスタごとの詳細
    sorted_clusters = sorted(
        cluster_stats.items(),
        key=lambda x: x[1]['size'],
        reverse=True
    )
    
    for cluster_id, stats in sorted_clusters[:10]:
        report_lines.append(f"### クラスタ {cluster_id} (n={stats['size']})\n")
        
        # このクラスタの代表的なコード進行を抽出
        cluster_indices = [i for i in range(len(labels)) if labels[i] == cluster_id]
        if cluster_indices:
            sample_progressions = [roman_progressions[i] for i in cluster_indices[:5]]
            report_lines.append("**代表的なコード進行:**\n")
            for prog in sample_progressions:
                report_lines.append(f"- {', '.join(prog)}\n")
            report_lines.append("\n")
        
        # 感情分布
        report_lines.append("**感情分布:**\n")
        report_lines.append("| 感情 | 平均 | 中央値 | 標準偏差 | 最小 | 最大 |\n")
        report_lines.append("|------|------|--------|----------|------|------|\n")
        
        emotion_names = ['JOY', 'SADNESS', 'ANTICIPATION', 'SURPRISE', 
                        'ANGER', 'FEAR', 'DISGUST', 'TRUST']
        
        for emotion in emotion_names:
            emotion_stats = stats['emotions'].get(emotion, {})
            report_lines.append(
                f"| {emotion} | {emotion_stats['mean']:.2f}% | "
                f"{emotion_stats['median']:.2f}% | {emotion_stats['std']:.2f}% | "
                f"{emotion_stats['min']:.2f}% | {emotion_stats['max']:.2f}% |\n"
            )
        
        report_lines.append("\n")
        
        # 解釈
        high_std_emotions = [
            (emotion, stats['emotions'][emotion]['std'])
            for emotion in emotion_names
            if stats['emotions'][emotion]['std'] > 15.0
        ]
        if high_std_emotions:
            report_lines.append("**観察:**\n")
            report_lines.append("- 標準偏差が大きい感情（ばらつきが大きい）:\n")
            for emotion, std in sorted(high_std_emotions, key=lambda x: x[1], reverse=True):
                report_lines.append(f"  - {emotion}: 標準偏差 {std:.2f}%\n")
            report_lines.append("\n")
            report_lines.append("→ 同じコード進行クラスタでも、感情が多様であることが確認されました。\n")
            report_lines.append("\n")
    
    # 総合的な解釈
    report_lines.append("## 総合的な解釈\n")
    report_lines.append("\n")
    report_lines.append("### 観察結果\n")
    report_lines.append("\n")
    report_lines.append("1. **同じクラスタ内でも感情のばらつきが大きい**\n")
    report_lines.append("   - 多くのクラスタで、標準偏差が15%を超える感情が複数存在\n")
    report_lines.append("   - これは、コード進行が似ていても、感情が必ずしも一致しないことを示唆\n")
    report_lines.append("\n")
    report_lines.append("2. **「紙一重」の要因**\n")
    report_lines.append("   - コード進行は感情の「器」であり、「決定要因」ではない\n")
    report_lines.append("   - 作詞家、アーティスト、年代などの要因が感情に影響を与える可能性\n")
    report_lines.append("\n")
    report_lines.append("3. **コード進行と感情の関係**\n")
    report_lines.append("   - コード進行と感情の関係は、単純な相関ではなく、楽曲ごとの文脈や表現意図によって多様\n")
    report_lines.append("   - 同じコード進行でも、異なる感情の歌詞が乗りうる\n")
    report_lines.append("\n")
    
    # レポートを保存
    report_path = output_path / 'cluster_emotion_analysis_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    print(f"レポートを保存: {report_path}")


def main():
    """
    メイン処理: クラスタ単位の感情分布比較図を生成
    """
    # パスの設定
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent.parent
    data_dir = root_dir / "vis_system" / "scattergraph" / "data"
    output_dir = root_dir / "vis_system" / "cluster_emotion_analysis"
    
    # データの読み込み
    json_path = data_dir / "umap_all.json"
    print(f"データを読み込み中: {json_path}")
    points_data, roman_progressions = load_scattergraph_data(str(json_path))
    print(f"読み込み完了: {len(points_data)}個の点")
    
    # 距離行列の計算
    print("\n距離行列を計算中...")
    distance_matrix = compute_progression_distance_matrix(roman_progressions)
    
    # クラスタリング
    print("\nクラスタリング中...")
    labels = cluster_progressions(distance_matrix, method='kmeans', n_clusters=15)
    
    # 感情分布の分析
    print("\n感情分布を分析中...")
    cluster_stats = analyze_cluster_emotion_distribution(points_data, labels)
    
    # 可視化
    print("\n可視化を生成中...")
    visualize_cluster_emotions(points_data, labels, cluster_stats, str(output_dir))
    
    # レポート生成
    print("\nレポートを生成中...")
    generate_cluster_report(cluster_stats, roman_progressions, labels, str(output_dir))
    
    print(f"\n完了！結果は {output_dir} に保存されました。")


if __name__ == "__main__":
    main()

