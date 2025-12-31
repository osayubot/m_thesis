"""
アーティストごとの散布図を生成
フレーズが20以上のアーティストに絞って、outputディレクトリに画像を保存
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

# 日本語フォントの設定
import platform
import matplotlib.font_manager as fm

# macOSで日本語フォントを検索
def get_japanese_font():
    if platform.system() == 'Darwin':  # macOS
        # 利用可能な日本語フォントを検索
        fonts = [f.name for f in fm.fontManager.ttflist]
        for font_name in ['Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'Hiragino Maru Gothic Pro']:
            if font_name in fonts:
                return font_name
    elif platform.system() == 'Windows':
        return 'MS Gothic'
    return 'DejaVu Sans'

plt.rcParams['font.family'] = get_japanese_font()

# センチメントから色へのマッピング
SENTIMENT_COLORS = {
    'positive': '#FF0000',   # 真っ赤（ポジティブ）
    'negative': '#0000FF',   # 真っ青（ネガティブ）
    'neutral': '#9E9E9E',   # グレー（中立）
}

# 後方互換性のため
EMOTION_COLORS = SENTIMENT_COLORS


def load_topic_analysis_data(json_path: str) -> Dict:
    """topic_analysis.jsonを読み込む"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def count_phrases_per_artist(data: Dict, min_cluster_size: int = 20) -> Dict[str, int]:
    """
    アーティストごとのフレーズ（曲）数をカウント
    
    Args:
        data: topic_analysis.jsonのデータ
        min_cluster_size: 使用するmin_cluster_size
    
    Returns:
        アーティスト名 -> フレーズ数の辞書
    """
    artist_counts = defaultdict(int)
    
    analysis = data['analyses'].get(str(min_cluster_size))
    if not analysis:
        return {}
    
    topic_songs = analysis.get('topic_songs', {})
    
    for topic_id, songs in topic_songs.items():
        for song in songs:
            artist = song.get('artist', '')
            if artist:
                artist_counts[artist] += 1
    
    return dict(artist_counts)


def create_emotion_pie_chart(
    x: float, y: float,
    emotions: List[Dict],
    edge_color: str,
    size: float = 1.0
) -> List[Wedge]:
    """
    センチメントデータから円グラフのパッチを作成
    
    Args:
        x, y: 中心座標
        emotions: センチメントデータのリスト（各要素は{'label': 'positive'/'negative', 'score': 0.0-1.0}の辞書、または古い形式の{emotion_name: value}の辞書）
        edge_color: 縁の色（基準進行の色）
        size: 円のサイズ
    
    Returns:
        Wedgeパッチのリスト
    """
    if not emotions:
        # センチメントデータがない場合は、基準進行の色を薄くした色で円を作成（縁の色も基準進行の色）
        # 縁の色を薄くした色を円の色として使用
        base_color = edge_color if edge_color != '#888888' else '#C0C0C0'
        circle = Circle((x, y), size, color=base_color, alpha=0.3, 
                       edgecolor=edge_color, linewidth=3.0)
        return [circle]
    
    # すべてのセンチメントを集計
    sentiment_totals = defaultdict(float)
    for item in emotions:
        # 新しいsentiment形式か古いemotion形式かを判定
        if isinstance(item, dict) and 'label' in item:
            # 新しいsentiment形式: {'label': 'positive'/'negative', 'score': 0.0-1.0}
            label = item.get('label', 'neutral').lower()
            score = item.get('score', 0.0)
            sentiment_totals[label] += score
        else:
            # 古いemotion形式（後方互換性）: {emotion_name: value}
            for emotion_name, value in item.items():
                # 感情名をsentimentにマッピング
                if emotion_name in ['JOY', 'TRUST', 'ANTICIPATION', 'SURPRISE']:
                    sentiment_totals['positive'] += value
                elif emotion_name in ['SADNESS', 'ANGER', 'FEAR', 'DISGUST']:
                    sentiment_totals['negative'] += value
                else:
                    sentiment_totals['neutral'] += value
    
    # 合計値で正規化
    total = sum(sentiment_totals.values())
    if total == 0:
        # センチメントデータがすべて0の場合は、基準進行の色を薄くした色で円を作成
        base_color = edge_color if edge_color != '#888888' else '#C0C0C0'
        circle = Circle((x, y), size, color=base_color, alpha=0.3,
                       edgecolor=edge_color, linewidth=3.0)
        return [circle]
    
    # 円グラフの各セクションを作成
    wedges = []
    start_angle = 0
    
    # 値が0より大きいセンチメントのみを使用
    valid_sentiments = {k: v for k, v in sentiment_totals.items() if v > 0}
    sorted_sentiments = sorted(valid_sentiments.items(), key=lambda x: x[1], reverse=True)
    
    for sentiment_label, value in sorted_sentiments:
        angle = (value / total) * 360.0
        color = SENTIMENT_COLORS.get(sentiment_label, '#808080')
        
        wedge = Wedge(
            (x, y), size,
            start_angle, start_angle + angle,
            color=color,
            alpha=0.7,
            edgecolor='white',  # セグメント間の線は白色
            linewidth=0.5  # 細くする
        )
        wedges.append(wedge)
        start_angle += angle
    
    # 円の外側の縁だけに基準進行の色を適用（外側の円を追加）
    if wedges:
        outer_circle = Circle((x, y), size, fill=False, edgecolor=edge_color, linewidth=3.0)
        wedges.append(outer_circle)
    
    if not wedges:
        # センチメントデータがすべて0の場合は、基準進行の色を薄くした色で円を作成
        base_color = edge_color if edge_color != '#888888' else '#C0C0C0'
        circle = Circle((x, y), size, color=base_color, alpha=0.3,
                       edgecolor=edge_color, linewidth=3.0)
        return [circle]
    
    return wedges


def get_artist_topics(analysis: Dict, artist: str) -> Set[int]:
    """
    指定されたアーティストの曲が含まれるトピックIDのセットを取得
    
    Args:
        analysis: analyses[min_cluster_size]のデータ
        artist: アーティスト名
    
    Returns:
        トピックIDのセット
    """
    artist_topics = set()
    
    topic_songs = analysis.get('topic_songs', {})
    for topic_id, songs in topic_songs.items():
        for song in songs:
            if song.get('artist', '') == artist:
                artist_topics.add(int(topic_id))
                break  # このトピックにはこのアーティストの曲が含まれている
    
    return artist_topics


def create_artist_scatterplot(
    data: Dict,
    artist: str,
    min_cluster_size: int = 20,
    output_path: str = None,
    figsize: tuple = (12, 12),
    analyzed_data_dir: str = "data/analyzed"
) -> None:
    """
    アーティストごとの散布図を生成
    
    Args:
        data: topic_analysis.jsonのデータ
        artist: アーティスト名
        min_cluster_size: 使用するmin_cluster_size
        output_path: 出力パス
        figsize: 図のサイズ
    """
    analysis = data['analyses'].get(str(min_cluster_size))
    if not analysis:
        print(f"Warning: No analysis data for min_cluster_size={min_cluster_size}")
        return
    
    points = analysis.get('points', [])
    reference_progressions = data.get('reference_progressions', {})
    reference_colors = data.get('reference_colors', {})
    
    # このアーティストの曲が含まれるトピックIDを取得
    artist_topics = get_artist_topics(analysis, artist)
    
    if not artist_topics:
        print(f"Warning: No topics found for artist: {artist}")
        return
    
    # 散布図を作成
    fig, ax = plt.subplots(figsize=figsize)
    
    # 基準進行のポイントをプロット
    reference_points_x = []
    reference_points_y = []
    reference_labels = []
    reference_colors_list = []
    
    # トピックのポイント情報を収集
    topic_data = []  # (x, y, topic_id, size, edge_color, lyrics_list)
    topic_songs_dict = analysis.get('topic_songs', {})
    
    for point in points:
        point_type = point.get('type')
        x = point.get('x', 0)
        y = point.get('y', 0)
        
        if point_type == 'reference':
            # 基準進行のポイント（常に表示）
            reference_points_x.append(x)
            reference_points_y.append(y)
            ref_name = point.get('reference_name', '')
            reference_labels.append(ref_name)
            reference_colors_list.append(point.get('color', '#000000'))
        elif point_type == 'topic':
            # トピックのポイント（このアーティストの曲が含まれる場合のみ表示）
            topic_id = point.get('topic_id')
            if topic_id in artist_topics:
                size = point.get('size', 1)
                circle_size = max(2, min(8, np.sqrt(size) * 0.3))  # 円グラフのサイズを調整
                
                # 最も近い基準進行の色を縁の色として使用
                closest_ref = point.get('closest_reference', '')
                edge_color = reference_colors.get(closest_ref, '#888888') if closest_ref else '#888888'
                
                # このトピックの歌詞情報を取得
                topic_songs = topic_songs_dict.get(str(topic_id), [])
                lyrics_list = []
                for song in topic_songs:
                    if song.get('artist', '') == artist:
                        lyric_preview = song.get('lyric_preview', '')
                        if lyric_preview:
                            lyrics_list.append(lyric_preview)
                
                topic_data.append((x, y, topic_id, circle_size, edge_color, lyrics_list, topic_songs))
    
    # 基準進行をプロット（大きめのマーカー）
    for i, (x, y, label, color) in enumerate(zip(reference_points_x, reference_points_y, 
                                                   reference_labels, reference_colors_list)):
        ax.scatter(x, y, s=300, c=color, marker='s', edgecolors='black', 
                  linewidths=2, label=f'Reference: {label}', zorder=5)
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', 
                   fontsize=10, fontweight='bold')
    
    # トピックを円グラフとしてプロット
    all_wedges = []
    
    for x, y, topic_id, circle_size, edge_color, lyrics_list, topic_songs in topic_data:
        # 感情データを取得
        emotions = load_emotion_data_for_topic(topic_id, topic_songs, artist, analyzed_data_dir)
        
        wedges = create_emotion_pie_chart(x, y, emotions, edge_color, circle_size)
        all_wedges.extend(wedges)
        
        # 歌詞を表示（最初の1つだけ、円の近くに）
        if lyrics_list:
            # 最初の歌詞を取得（最大30文字）
            lyric_text = lyrics_list[0]
            if len(lyric_text) > 30:
                lyric_text = lyric_text[:30] + '...'
            
            # 円の右側に歌詞を配置（日本語フォントを明示的に指定）
            from matplotlib import font_manager
            japanese_font_name = get_japanese_font()
            # FontPropertiesを使用してフォントを確実に適用
            font_prop = font_manager.FontProperties(family=japanese_font_name, size=8)
            ax.text(x + circle_size * 1.5, y, lyric_text, 
                   fontproperties=font_prop,
                   ha='left', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.9, edgecolor='gray', linewidth=0.5),
                   zorder=4)
    
    # パッチを個別に追加（edgecolorを確実に適用するため）
    if all_wedges:
        for wedge in all_wedges:
            ax.add_patch(wedge)
    
    # すべての座標を結合して軸の範囲を設定
    topic_points_x = [x for x, _, _, _, _, _, _ in topic_data]
    topic_points_y = [y for _, y, _, _, _, _, _ in topic_data]
    all_x = reference_points_x + topic_points_x if topic_points_x else reference_points_x
    all_y = reference_points_y + topic_points_y if topic_points_y else reference_points_y
    
    if all_x and all_y:
        # 軸の範囲を適切に設定（少し余白を追加）
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # 範囲が0の場合はデフォルト範囲を使用
        if x_range == 0:
            x_range = 10
        if y_range == 0:
            y_range = 10
        
        margin_x = x_range * 0.1
        margin_y = y_range * 0.1
        
        ax.set_xlim(x_min - margin_x, x_max + margin_x)
        ax.set_ylim(y_min - margin_y, y_max + margin_y)
    
    # 軸の設定
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_title(f'Scatter Plot for Artist: {artist}\n(min_cluster_size={min_cluster_size})', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # 凡例
    legend_elements = []
    for ref_name, color in reference_colors.items():
        legend_elements.append(
            mpatches.Patch(color=color, label=f'Reference: {ref_name}')
        )
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved scatter plot for {artist} to {output_path}")
    else:
        plt.show()
    
    plt.close()


def load_emotion_data_for_topic(
    topic_id: int,
    topic_songs: List[Dict],
    artist: str,
    analyzed_data_dir: str = "data/analyzed"
) -> List[Dict[str, float]]:
    """
    トピックに含まれるアーティストの曲から感情データを取得
    
    Args:
        topic_id: トピックID
        topic_songs: トピックに含まれる曲のリスト
        artist: アーティスト名
        analyzed_data_dir: 分析済みデータディレクトリ
    
    Returns:
        感情データのリスト（各要素は{emotion_name: value}の辞書）
    """
    emotions = []
    from pathlib import Path
    
    script_dir = Path(__file__).parent.parent.parent
    data_dir = script_dir / analyzed_data_dir
    
    if not data_dir.exists():
        return []
    
    # このアーティストの曲のみを対象
    for song in topic_songs:
        if song.get('artist', '') != artist:
            continue
        
        spotify_id = song.get('spotify_id', '')
        if not spotify_id:
            continue
        
        # データファイルを検索
        json_file = data_dir / f"{spotify_id}.json"
        if not json_file.exists():
            continue
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                song_data = json.load(f)
            
            # 感情データを収集
            analyzed = song_data.get('analyzed_chord_progressions_and_lyrics', [])
            for section in analyzed:
                # sentimentを優先、なければemotion（後方互換性）
                sentiment = section.get('sentiment', {})
                emotion = section.get('emotion', {}) if not sentiment else None
                
                if sentiment:
                    # 新しいsentiment形式
                    if sentiment.get('score', 0.0) > 0:
                        emotions.append(sentiment)
                elif emotion:
                    # 古いemotion形式（後方互換性）
                    if any(v > 0 for v in emotion.values()):
                        # emotionをsentiment形式に変換
                        max_emotion = max(emotion.items(), key=lambda x: x[1])
                        emotion_name = max_emotion[0]
                        max_emotion_value = max_emotion[1]
                        if emotion_name in ['JOY', 'TRUST', 'ANTICIPATION', 'SURPRISE']:
                            emotions.append({'label': 'positive', 'score': max_emotion_value})
                        elif emotion_name in ['SADNESS', 'ANGER', 'FEAR', 'DISGUST']:
                            emotions.append({'label': 'negative', 'score': max_emotion_value})
                        else:
                            emotions.append({'label': 'neutral', 'score': max_emotion_value})
        except Exception as e:
            continue
    
    return emotions


def main(
    json_path: str = None,
    output_dir: str = None,
    min_cluster_size: int = 20,
    min_phrases: int = 20,
    analyzed_data_dir: str = "data/analyzed"
) -> None:
    """
    メイン処理
    
    Args:
        json_path: topic_analysis.jsonのパス
        output_dir: 出力ディレクトリ
        min_cluster_size: 使用するmin_cluster_size
        min_phrases: 最小フレーズ数（これ以上のアーティストのみ処理）
    """
    # パスの解決
    if json_path is None:
        script_dir = Path(__file__).parent.parent.parent
        json_path = script_dir / "vis_system" / "scattergraph_bertopic" / "data" / "topic_analysis.json"
    else:
        json_path = Path(json_path)
    
    if output_dir is None:
        script_dir = Path(__file__).parent.parent.parent
        output_dir = script_dir / "vis_system" / "scattergraph_bertopic" / "output"
    else:
        output_dir = Path(output_dir)
    
    # ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("アーティストごとの散布図生成")
    print("=" * 60)
    print(f"JSONファイル: {json_path}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"min_cluster_size: {min_cluster_size}")
    print(f"最小フレーズ数: {min_phrases}")
    print("=" * 60)
    
    # データを読み込む
    print("\nデータを読み込んでいます...")
    if not json_path.exists():
        print(f"Error: JSONファイルが見つかりません: {json_path}")
        return
    
    data = load_topic_analysis_data(str(json_path))
    print("データ読み込み完了")
    
    # アーティストごとのフレーズ数をカウント
    print("\nアーティストごとのフレーズ数をカウントしています...")
    artist_counts = count_phrases_per_artist(data, min_cluster_size)
    
    # 最小フレーズ数以上のアーティストをフィルタリング
    filtered_artists = {
        artist: count for artist, count in artist_counts.items() 
        if count >= min_phrases
    }
    
    print(f"\n総アーティスト数: {len(artist_counts)}")
    print(f"フレーズ数{min_phrases}以上のアーティスト数: {len(filtered_artists)}")
    
    # フレーズ数でソート
    sorted_artists = sorted(filtered_artists.items(), key=lambda x: x[1], reverse=True)
    
    # 各アーティストの散布図を生成
    print("\n散布図を生成しています...")
    for i, (artist, count) in enumerate(sorted_artists, 1):
        # ファイル名に使用できない文字を置換
        safe_artist_name = artist.replace('/', '_').replace('\\', '_').replace(':', '_')
        output_path = output_dir / f"{safe_artist_name}_scatterplot.png"
        
        print(f"[{i}/{len(sorted_artists)}] Processing {artist} ({count} phrases)...")
        
        try:
            create_artist_scatterplot(
                data=data,
                artist=artist,
                min_cluster_size=min_cluster_size,
                output_path=str(output_path),
                figsize=(12, 12),
                analyzed_data_dir=analyzed_data_dir
            )
        except Exception as e:
            print(f"  Error processing {artist}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("完了!")
    print("=" * 60)
    print(f"生成された画像数: {len(sorted_artists)}")
    print(f"出力ディレクトリ: {output_dir}")


if __name__ == "__main__":
    import sys
    
    json_path = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    min_cluster_size = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    min_phrases = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    
    main(
        json_path=json_path,
        output_dir=output_dir,
        min_cluster_size=min_cluster_size,
        min_phrases=min_phrases
    )

