"""
MDSによる座標計算と円グラフ可視化
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.collections import PatchCollection
from sklearn.manifold import MDS
import matplotlib.patches as mpatches

from .musical_distance import compute_distance_matrix, circular_distance
from ..analyze_data.roman_numeral import section_to_roman_progression

# ============================================================================
# 基準進行（典型コード進行）の定義
# ============================================================================
# 新しい典型コード進行を追加する場合は、以下の辞書に追加してください。
# 進行はローマ数字で表記します（例: ['IV', 'V', 'I']）
# 
# 追加方法:
#   1. REFERENCE_PROGRESSIONS に新しいエントリを追加
#   2. REFERENCE_COLORS に対応する色を追加（16進数カラーコード）
#   3. FILE_NAME_MAP に対応するファイル名を追加（基準進行ベースモード用）
# ============================================================================

REFERENCE_PROGRESSIONS = {
    'odo': ['IV', 'V', 'iii', 'vi'],      # 王道進行: Ⅳ→Ⅴ→Ⅲm→Ⅵm
    'komuro': ['vi', 'IV', 'V', 'I'],     # 小室進行: VIm → IV → V → I
    'marusa': ['IVM7', 'III7', 'vi7', 'I7'],  # マルサ進行: ⅣM7 - Ⅲ7 - Ⅵm7 - Ⅰ7
}

# 基準進行の表示色（散布図での枠線の色）
REFERENCE_COLORS = {
    'odo': '#FF0000',      # 赤（王道進行）
    'komuro': '#0000FF',   # 青（小室進行）
    'marusa': '#00FF00',   # 緑（マルサ進行）
}

FILE_NAME_MAP = {
    'odo': 'mds_odo_pie_data.json',
    'komuro': 'mds_komuro_pie_data.json',
    'marusa': 'mds_marusa_pie_data.json',
}

# 感情から色へのマッピング（Plutchik's model of emotions ベース）
EMOTION_COLORS = {
    'JOY': '#FFFF73',        # (255,255,115)
    'SADNESS': '#5150F8',    # (81,80,248)
    'ANTICIPATION': '#F3AB63', # (243,171,99)
    'SURPRISE': '#74BBF9',   # (116,187,249)
    'ANGER': '#E93323',      # (233,51,35)
    'FEAR': '#429429',       # (66,148,41)
    'DISGUST': '#EB60F8',    # (235,96,248)
    'TRUST': '#88FC6E',      # (136,252,110)
}

def emotion_to_color(emotion_dict: Dict[str, float]) -> str:
    """
    感情辞書から色を決定（最も強い感情の色を使用）
    """
    if not emotion_dict:
        return '#808080'  # グレー（感情データなし）
    
    max_emotion = max(emotion_dict.items(), key=lambda x: x[1])
    emotion_name = max_emotion[0]
    return EMOTION_COLORS.get(emotion_name, '#808080')

def load_analyzed_data(data_dir: str, max_files: Optional[int] = None) -> List[Dict]:
    """
    分析済みデータを読み込む
    
    Args:
        data_dir: データディレクトリ
        max_files: 最大ファイル数（Noneの場合は全て）
    
    Returns:
        楽曲データのリスト
    """
    data_path = Path(data_dir)
    json_files = list(data_path.glob("*.json"))
    
    if max_files:
        json_files = json_files[:max_files]
    
    all_songs = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_songs.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    return all_songs

def extract_chord_progressions_with_lyrics(songs: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    コード進行と紐づく歌詞を抽出
    
    Returns:
        (progressions_data, songs_list)
        progressions_data: 各コード進行とその歌詞のリスト
        songs_list: 曲情報のリスト（元のsongsと同じ順序）
    """
    progressions_data = []
    songs_list = []
    
    for song in songs:
        # 曲情報を保存
        song_info = {
            'title': song.get('title', ''),
            'artist': song.get('artist', ''),
            'lyricist': song.get('lyricist', ''),
            'composer': song.get('composer', ''),
            'spotify_id': song.get('spotify_id', ''),
            'jtotal_path': song.get('jtotal_path', ''),
            'release_date': song.get('release_date', ''),
            'spotify_popularity': song.get('spotify_popularity', None)
        }
        songs_list.append(song_info)
        
        analyzed = song.get('analyzed_chord_progressions_and_lyrics', [])
        
        for section in analyzed:
            chord_prog = section.get('normalized_chord_progression', [])
            if not chord_prog or chord_prog == ['N.C']:
                continue
            
            # ローマ数字に変換
            key = section.get('key')
            if not key:
                continue
            
            # normalized_chord_progressionを使用してローマ数字に変換
            # section_to_roman_progressionはchord_progressionを使うので、
            # 一時的にnormalized_chord_progressionをchord_progressionとして渡す
            section_for_roman = section.copy()
            section_for_roman['chord_progression'] = chord_prog
            roman_prog = section_to_roman_progression(section_for_roman, key)
            if not roman_prog:
                continue
            
            # フィルタリング条件1: コード進行は4つの連なりのもののみを採用
            if len(roman_prog) != 4:
                continue
            
            # 歌詞を取得（歌詞がないセクションはスキップ）
            lyric = section.get('lyric', '').strip()
            emotion = section.get('emotion', {})
            
            # 歌詞がない場合はスキップ（歌詞数0を避けるため）
            if not lyric:
                continue
            
            # フィルタリング条件2: 感情については、0.5以上のものがある歌詞のみ採用
            if emotion:
                # 感情値の最大値を取得
                max_emotion_value = max(emotion.values()) if emotion.values() else 0.0
                if max_emotion_value < 0.5:
                    continue
            else:
                # 感情データがない場合はスキップ
                continue
            
            # 同じコード進行をグループ化
            prog_key = tuple(roman_prog)
            found = False
            for prog_data in progressions_data:
                # listとtupleの比較を正しく行う
                existing_prog = tuple(prog_data['roman_progression'])
                if existing_prog == prog_key:
                    # 既存のコード進行に歌詞を追加
                    prog_data['lyrics'].append({
                        'lyric': lyric,
                        'emotion': emotion,
                        'color': emotion_to_color(emotion),
                        'song_index': len(songs_list) - 1  # どの曲に属するか
                    })
                    found = True
                    break
            
            if not found:
                # 新しいコード進行を追加
                lyrics_list = [{
                    'lyric': lyric,
                    'emotion': emotion,
                    'color': emotion_to_color(emotion),
                    'song_index': len(songs_list) - 1
                }]
                
                progressions_data.append({
                    'chord_progression': section.get('chord_progression', []),
                    'normalized_chord_progression': chord_prog,
                    'roman_progression': list(prog_key),
                    'lyrics': lyrics_list,
                    'key': key,
                })
    
    return progressions_data, songs_list

def compute_distance_vectors(
    progressions_data: List[Dict],
    reference_progressions: Dict[str, List[str]] = None
) -> np.ndarray:
    """
    各コード進行から基準進行への距離ベクトルを計算
    
    Args:
        progressions_data: コード進行データ
        reference_progressions: 基準進行の辞書（Noneの場合はデフォルトを使用）
    
    Returns:
        距離ベクトルの配列（n×m、n=コード進行数、m=基準進行数）
    """
    if reference_progressions is None:
        reference_progressions = REFERENCE_PROGRESSIONS
    
    # ローマ数字のコード進行を取得
    roman_progressions = [prog['roman_progression'] for prog in progressions_data]
    
    # 基準進行のリスト（順序を保持）
    ref_names = list(reference_progressions.keys())
    ref_progs = [reference_progressions[name] for name in ref_names]
    
    print(f"Computing distance vectors to {len(ref_names)} reference progressions...")
    print(f"Reference progressions: {ref_names}")
    
    # 各コード進行について、各基準進行への距離を計算
    distance_vectors = []
    for i, prog in enumerate(roman_progressions):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(roman_progressions)}...")
        
        vector = []
        for ref_prog in ref_progs:
            dist = circular_distance(prog, ref_prog)
            vector.append(dist)
        distance_vectors.append(vector)
    
    return np.array(distance_vectors), ref_names

def compute_reference_based_mds_coordinates(
    progressions_data: List[Dict],
    n_components: int = 2,
    reference_progressions: Dict[str, List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    基準進行からの距離ベクトルを使ってMDSで座標を計算
    
    Args:
        progressions_data: コード進行データ
        n_components: 次元数（2または3）
        reference_progressions: 基準進行の辞書（Noneの場合はデフォルトを使用）
    
    Returns:
        (座標配列（n×n_components）, 基準進行名のリスト)
    """
    # 距離ベクトルを計算
    distance_vectors, ref_names = compute_distance_vectors(
        progressions_data, reference_progressions
    )
    
    # 距離ベクトル間のユークリッド距離で距離行列を作成
    print("Computing distance matrix from distance vectors...")
    n = len(distance_vectors)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            # ユークリッド距離
            dist = np.linalg.norm(distance_vectors[i] - distance_vectors[j])
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
    
    # MDSで座標を計算
    print("Computing MDS coordinates...")
    mds = MDS(
        n_components=n_components,
        dissimilarity='precomputed',
        random_state=42,
        n_init=10,
        max_iter=1000
    )
    coordinates = mds.fit_transform(dist_matrix)
    
    return coordinates, ref_names

def compute_mds_coordinates(progressions_data: List[Dict], n_components: int = 2) -> np.ndarray:
    """
    MDSで座標を計算（従来の方法：全コード進行間の距離行列を使用）
    
    Args:
        progressions_data: コード進行データ
        n_components: 次元数（2または3）
    
    Returns:
        座標配列（n×n_components）
    """
    # ローマ数字のコード進行を取得
    roman_progressions = [prog['roman_progression'] for prog in progressions_data]
    
    # 距離行列を計算
    print("Computing distance matrix...")
    dist_matrix = compute_distance_matrix(roman_progressions)
    
    # MDSで座標を計算
    print("Computing MDS coordinates...")
    print("  This may take several minutes depending on the data size...")
    import time
    mds_start = time.time()
    mds = MDS(
        n_components=n_components,
        dissimilarity='precomputed',
        random_state=42,
        n_init=10,
        max_iter=1000
    )
    coordinates = mds.fit_transform(dist_matrix)
    mds_time = time.time() - mds_start
    print(f"  MDS computation completed in {mds_time:.1f}s")
    
    return coordinates

def create_pie_chart_patches(
    x: float, y: float, 
    lyrics: List[Dict],
    size: float = 1.0
) -> List[mpatches.Wedge]:
    """
    円グラフのパッチを作成
    
    Args:
        x, y: 中心座標
        lyrics: 歌詞のリスト
        size: 円のサイズ（歌詞数に基づく）
    
    Returns:
        Wedgeパッチのリスト
    """
    if not lyrics:
        # 歌詞がない場合は小さなグレーの円
        return [Circle((x, y), size * 0.1, color='gray', alpha=0.3)]
    
    n = len(lyrics)
    if n == 0:
        return []
    
    # 各歌詞の割合（等分）
    angles_per_lyric = 360.0 / n
    
    wedges = []
    start_angle = 0
    
    for i, lyric_data in enumerate(lyrics):
        color = lyric_data.get('color', '#808080')
        angle = angles_per_lyric
        
        wedge = Wedge(
            (x, y), size,
            start_angle, start_angle + angle,
            color=color,
            alpha=0.7,
            edgecolor='white',
            linewidth=0.5
        )
        wedges.append(wedge)
        start_angle += angle
    
    return wedges

def normalize_roman(roman: str) -> str:
    """
    ローマ数字を正規化（大文字小文字を統一、7thコードの表記を統一）
    
    Args:
        roman: ローマ数字文字列
    
    Returns:
        正規化されたローマ数字
    """
    # 大文字小文字を統一（基本は小文字、ただし最初の文字は大文字）
    # 7thコードの表記を統一（M7, m7, 7など）
    normalized = roman.strip()
    
    # 7thコードの表記を統一（M7, m7, 7 → 7）
    # ただし、これは後で比較する際に考慮する
    
    return normalized

def is_same_progression(prog1: List[str], prog2: List[str]) -> bool:
    """
    2つのコード進行が同じかどうかを判定（循環性を考慮、大文字小文字を無視）
    
    Args:
        prog1: コード進行1
        prog2: コード進行2
    
    Returns:
        同じかどうか
    """
    if len(prog1) != len(prog2):
        return False
    
    # 大文字小文字を無視して比較する関数
    def normalize_for_comparison(prog: List[str]) -> List[str]:
        normalized = []
        for r in prog:
            # 大文字小文字を統一（小文字に変換）
            r_lower = r.lower()
            # 7thコードの表記を統一（M7, m7, 7 → 7として扱う）
            # IVM7, IVm7, IV7 → iv7 として統一
            if 'm7' in r_lower or '7' in r_lower or 'M7' in r:
                # 7thコードがある場合、基本のローマ数字部分を抽出
                # 例: IVM7 → iv, III7 → iii, vi7 → vi, I7 → i
                # M7, m7, 7をすべて除去してから7を追加
                base = r_lower.replace('m7', '').replace('M7', '').rstrip('7')
                normalized.append(base + '7')
            else:
                normalized.append(r_lower)
        return normalized
    
    prog1_norm = normalize_for_comparison(prog1)
    prog2_norm = normalize_for_comparison(prog2)
    
    # 完全一致
    if prog1_norm == prog2_norm:
        return True
    
    # 循環的に一致するかチェック
    for i in range(len(prog1_norm)):
        rotated = prog1_norm[i:] + prog1_norm[:i]
        if rotated == prog2_norm:
            return True
    
    return False

def export_to_json_format(
    progressions_data: List[Dict],
    coordinates: np.ndarray,
    songs_list: List[Dict],
    output_path: str,
    reference_progression: Optional[List[str]] = None,
    reference_name: Optional[str] = None,
    reference_progressions: Optional[Dict[str, List[str]]] = None
) -> Dict:
    """
    all.htmlが期待するJSON形式でデータを出力
    
    Returns:
        {
            "points": [
                {
                    "x": float,
                    "y": float,
                    "r": float,
                    "pie": [
                        {"label": "JOY", "c": "#FFD700", "v": 25.0},
                        ...
                    ]
                },
                ...
            ],
            "emotionSongs": {
                "JOY": [
                    {"title": "...", "artist": "...", "lyricist": "...", "composer": "..."},
                    ...
                ],
                ...
            }
        }
    """
    # 座標の範囲を取得（0-100に正規化するため）
    if len(coordinates) == 0:
        return {"points": [], "emotionSongs": {}}
    
    x_coords = coordinates[:, 0]
    y_coords = coordinates[:, 1]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    x_range = x_max - x_min if x_max > x_min else 1
    y_range = y_max - y_min if y_max > y_min else 1
    
    # 原点中心の場合、範囲を対称にする
    if abs(x_min + x_max) < 0.01 and abs(y_min + y_max) < 0.01:
        # 原点中心なので、最大絶対値を使って範囲を決める
        x_abs_max = max(abs(x_min), abs(x_max))
        y_abs_max = max(abs(y_min), abs(y_max))
        x_range = x_abs_max * 2 if x_abs_max > 0 else 1
        y_range = y_abs_max * 2 if y_abs_max > 0 else 1
        x_min = -x_abs_max
        y_min = -y_abs_max
    
    # 感情ごとの曲リストを構築
    emotion_songs = defaultdict(list)  # {emotion: [song_info, ...]}
    
    points = []
    
    for i, (prog_data, coord) in enumerate(zip(progressions_data, coordinates)):
        lyrics = prog_data['lyrics']
        if not lyrics:
            continue
        
        # 座標を0-100に正規化
        x_norm = ((coord[0] - x_min) / x_range) * 100
        y_norm = ((coord[1] - y_min) / y_range) * 100
        
        # 各歌詞を等分（10個なら10%ずつ）
        # 各歌詞の最も強い感情を取得して、同じ感情のものをまとめる
        emotion_groups = defaultdict(list)  # {emotion: [lyric_data, ...]}
        emotion_song_indices = defaultdict(set)  # {emotion: {song_index, ...}}
        emotion_lyrics_map = defaultdict(list)  # {emotion: [{'lyric': lyric_text, 'song_index': song_index}, ...]} - 曲情報も含める
        
        # 有効な歌詞のみをフィルタリング（感情データがあり、歌詞テキストもあるもの）
        # フィルタリング条件: 感情については、0.5以上のものがある歌詞のみ採用
        valid_lyrics = []
        for lyric_data in lyrics:
            emotion = lyric_data.get('emotion', {})
            lyric_text = lyric_data.get('lyric', '').strip()
            
            # 感情データと歌詞テキストの両方が必要
            if not emotion or not lyric_text:
                continue
            
            # 感情値の最大値が0.5以上であることを確認
            max_emotion_value = max(emotion.values()) if emotion.values() else 0.0
            if max_emotion_value < 0.5:
                continue
            
            valid_lyrics.append(lyric_data)
        
        # 有効な歌詞がない場合はスキップ
        if not valid_lyrics:
            continue
        
        for lyric_data in valid_lyrics:
            emotion = lyric_data.get('emotion', {})
            lyric_text = lyric_data.get('lyric', '').strip()
            
            # 最も強い感情を取得
            max_emotion = max(emotion.items(), key=lambda x: x[1])
            emotion_name = max_emotion[0]
            
            # この歌詞が属する曲のインデックスを記録
            song_index = lyric_data.get('song_index')
            
            # 同じ歌詞が既にこの感情グループに存在するかチェック（重複除去）
            lyric_exists = any(
                item['lyric'] == lyric_text 
                for item in emotion_lyrics_map[emotion_name]
            )
            
            if not lyric_exists:
                emotion_groups[emotion_name].append(lyric_data)
                if lyric_text:
                    emotion_lyrics_map[emotion_name].append({
                        'lyric': lyric_text,
                        'song_index': song_index
                    })
            
            if song_index is not None:
                emotion_song_indices[emotion_name].add(song_index)
        
        if not emotion_groups:
            continue
        
        # 重複を除去した後のユニークな歌詞数を計算
        unique_lyrics = set()
        for lyric_list in emotion_lyrics_map.values():
            for lyric_item in lyric_list:
                unique_lyrics.add(lyric_item['lyric'])
        total_unique_lyrics = len(unique_lyrics)
        
        # 各感情グループの割合を計算（ユニークな歌詞数に基づく）
        percentage_per_lyric = 100.0 / total_unique_lyrics if total_unique_lyrics > 0 else 0
        
        # pieデータを作成（%で合計100になるように）
        pie_data = []
        for emotion_name in EMOTION_COLORS.keys():
            lyric_count = len(emotion_groups.get(emotion_name, []))
            if lyric_count > 0:
                percentage = lyric_count * percentage_per_lyric
                # 歌詞と曲情報を含むリスト（最大10個まで）
                lyrics_with_songs = emotion_lyrics_map[emotion_name][:10]
                # 曲情報を追加
                lyrics_list = []
                for lyric_item in lyrics_with_songs:
                    lyric_entry = {
                        'lyric': lyric_item['lyric'],
                        'song_index': lyric_item.get('song_index')
                    }
                    # 曲情報があれば追加
                    if lyric_item.get('song_index') is not None and lyric_item['song_index'] < len(songs_list):
                        song_info = songs_list[lyric_item['song_index']]
                        lyric_entry['song'] = {
                            'title': song_info.get('title', ''),
                            'artist': song_info.get('artist', ''),
                            'composer': song_info.get('composer', ''),
                            'lyricist': song_info.get('lyricist', ''),
                            'release_date': song_info.get('release_date', ''),
                            'spotify_popularity': song_info.get('spotify_popularity', None)
                        }
                    lyrics_list.append(lyric_entry)
                
                pie_data.append({
                    'label': emotion_name,
                    'c': EMOTION_COLORS[emotion_name],
                    'v': float(round(percentage, 2)),
                    'lyrics': lyrics_list
                })
                
                # この感情に対応する曲を追加
                for song_index in emotion_song_indices[emotion_name]:
                    if song_index < len(songs_list):
                        song_info = songs_list[song_index]
                        # 重複チェック（titleとartistで判定）
                        is_duplicate = any(
                            s.get('title') == song_info.get('title') and 
                            s.get('artist') == song_info.get('artist')
                            for s in emotion_songs[emotion_name]
                        )
                        if not is_duplicate:
                            emotion_songs[emotion_name].append(song_info)
        
        # 合計が100になるように調整（丸め誤差対策）
        if pie_data:
            total_pie = sum(p['v'] for p in pie_data)
            if abs(total_pie - 100) > 0.01:  # 0.01%以上の誤差がある場合
                diff = 100 - total_pie
                if pie_data:
                    pie_data[-1]['v'] = float(round(pie_data[-1]['v'] + diff, 2))
        
        # 半径を計算（ユニークな歌詞数に基づく）- より明確なサイズ差を出す（小さめに調整）
        lyric_count = total_unique_lyrics
        min_r = 4
        max_r = 20
        # 対数スケールでより明確な差を出す（1個=4, 10個=12, 50個=18, 100個=20程度）
        if lyric_count <= 1:
            r = min_r
        else:
            # 対数スケール: log(lyric_count) / log(max_lyric_count) * (max_r - min_r) + min_r
            # 最大歌詞数を100と仮定
            max_lyric_count = 100
            log_scale = np.log(lyric_count) / np.log(max_lyric_count)
            r = min_r + (max_r - min_r) * min(1.0, log_scale)
        
        
        # コード進行を取得（ローマ数字表記）
        roman_progression = prog_data.get('roman_progression', [])
        chord_progression = prog_data.get('chord_progression', [])
        
        # コード進行の文字列表現を作成
        if roman_progression:
            progression_str = ' - '.join(roman_progression)
        elif chord_progression:
            progression_str = ' - '.join(chord_progression)
        else:
            progression_str = 'N/A'
        
        # 基準進行と一致するかチェック
        stroke_color = None
        matched_reference_name = None
        
        # 複数の基準進行をチェック（reference_progressionsが提供された場合）
        if reference_progressions and roman_progression:
            for ref_name, ref_prog in reference_progressions.items():
                if is_same_progression(roman_progression, ref_prog):
                    # 基準進行の色を取得（デフォルトは黒）
                    stroke_color = REFERENCE_COLORS.get(ref_name, "#000000")
                    matched_reference_name = ref_name
                    print(f"  Found matching progression for {ref_name}: {progression_str}")
                    break
        # 単一の基準進行をチェック（後方互換性のため）
        elif reference_progression and roman_progression:
            if is_same_progression(roman_progression, reference_progression):
                # 基準進行の色を取得（デフォルトは黒）
                stroke_color = REFERENCE_COLORS.get(reference_name, "#000000") if reference_name else "#000000"
                matched_reference_name = reference_name
                if reference_name:
                    print(f"  Found matching progression for {reference_name}: {progression_str}")
        
        point_data = {
            'x': float(round(x_norm, 2)),
            'y': float(round(y_norm, 2)),
            'r': float(round(r, 2)),
            'pie': pie_data,
            'lyricCount': lyric_count,
            'progression': progression_str,
            'roman_progression': roman_progression,
            'chord_progression': chord_progression
        }
        
        # 基準進行と一致する場合はstrokeColorを追加
        if stroke_color:
            point_data['strokeColor'] = stroke_color
            if matched_reference_name:
                point_data['referenceName'] = matched_reference_name
        
        points.append(point_data)
    
    # 感情ごとの曲リストを辞書形式に変換
    emotion_songs_dict = {}
    for emotion_name in EMOTION_COLORS.keys():
        emotion_songs_dict[emotion_name] = emotion_songs[emotion_name]
    
    result = {
        'points': points,
        'emotionSongs': emotion_songs_dict
    }
    
    # JSONファイルに保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Exported JSON to {output_path}")
    return result

def visualize_mds_with_pie_charts(
    progressions_data: List[Dict],
    coordinates: np.ndarray,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 20),
    show_lyrics: bool = True,
    max_lyric_length: int = 30
):
    """
    MDS座標上に円グラフを表示
    
    Args:
        progressions_data: コード進行データ
        coordinates: MDS座標
        output_path: 出力パス（Noneの場合は表示のみ）
        figsize: 図のサイズ
        show_lyrics: 歌詞を表示するか
        max_lyric_length: 表示する歌詞の最大長
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # サイズの範囲を正規化（最小0.05、最大0.3）
    lyric_counts = [len(prog['lyrics']) for prog in progressions_data]
    if lyric_counts:
        min_count = min(lyric_counts)
        max_count = max(lyric_counts)
        if max_count > min_count:
            sizes = [0.05 + 0.25 * (count - min_count) / (max_count - min_count) 
                    for count in lyric_counts]
        else:
            sizes = [0.15] * len(lyric_counts)
    else:
        sizes = [0.15] * len(progressions_data)
    
    # 各コード進行に対して円グラフを作成
    all_wedges = []
    for i, (prog_data, coord, size) in enumerate(zip(progressions_data, coordinates, sizes)):
        x, y = coord[0], coord[1]
        lyrics = prog_data['lyrics']
        
        # 円グラフのサイズを調整（歌詞数に基づく）
        pie_size = size  # 既に正規化済み
        
        wedges = create_pie_chart_patches(x, y, lyrics, pie_size)
        all_wedges.extend(wedges)
        
        # 歌詞をテキストで表示
        if show_lyrics and lyrics:
            # 最初の数個の歌詞を表示
            lyrics_to_show = lyrics[:3]  # 最大3つ
            for j, lyric_data in enumerate(lyrics_to_show):
                lyric_text = lyric_data['lyric']
                if len(lyric_text) > max_lyric_length:
                    lyric_text = lyric_text[:max_lyric_length] + '...'
                
                # 円の周りに配置
                angle_offset = (360.0 / len(lyrics)) * j - 90  # 上から開始
                angle_rad = np.radians(angle_offset)
                # テキストの位置を円の外側に配置
                text_x = x + (pie_size * 1.5 + 0.02) * np.cos(angle_rad)
                text_y = y + (pie_size * 1.5 + 0.02) * np.sin(angle_rad)
                
                ax.text(
                    text_x, text_y,
                    lyric_text,
                    fontsize=6,
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'),
                    rotation=angle_offset if abs(angle_offset) < 90 else angle_offset + 180
                )
    
    # パッチを追加
    pc = PatchCollection(all_wedges, match_original=True)
    ax.add_collection(pc)
    
    # 軸の設定
    ax.set_xlabel('MDS Dimension 1', fontsize=12)
    ax.set_ylabel('MDS Dimension 2', fontsize=12)
    ax.set_title('Chord Progressions with Lyrics (MDS Visualization)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # 凡例（感情の色）
    legend_elements = [
        mpatches.Patch(color=color, label=emotion) 
        for emotion, color in EMOTION_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()

def main(
    data_dir: str = "data/analyzed",
    output_path: Optional[str] = None,
    json_output_path: Optional[str] = None,
    max_files: Optional[int] = 100,
    show_lyrics: bool = True,
    export_json: bool = True,
    reference_progressions: Optional[Dict[str, List[str]]] = None
):
    """
    メイン処理
    
    Args:
        data_dir: 分析済みデータディレクトリ
        output_path: 画像出力パス（Noneの場合は表示のみ）
        json_output_path: JSON出力パス（Noneの場合は自動生成）
        max_files: 最大ファイル数
        show_lyrics: 歌詞を表示するか（画像出力時）
        export_json: JSONを出力するか
        reference_progressions: 基準進行の辞書（Noneの場合はデフォルトを使用）
    """
    # データディレクトリのパスを解決
    data_path = Path(data_dir)
    if not data_path.is_absolute():
        # 相対パスの場合、スクリプトディレクトリからの相対パスとして扱う
        script_dir = Path(__file__).parent.parent.parent
        data_path = script_dir / data_dir
    
    if not data_path.exists():
        print(f"Error: Data directory not found: {data_path}")
        return
    
    # データを読み込む
    print(f"Loading data from {data_path}...")
    songs = load_analyzed_data(str(data_path), max_files)
    print(f"Loaded {len(songs)} songs")
    
    # コード進行と歌詞を抽出
    print("Extracting chord progressions with lyrics...")
    progressions_data, songs_list = extract_chord_progressions_with_lyrics(songs)
    print(f"Found {len(progressions_data)} unique chord progressions")
    
    if len(progressions_data) == 0:
        print("No chord progressions found!")
        return
    
    # 歌詞が少ないコード進行を除外（可視化のため）
    progressions_data = [p for p in progressions_data if len(p['lyrics']) > 0]
    print(f"After filtering (with lyrics): {len(progressions_data)} progressions")
    
    if len(progressions_data) < 2:
        print("Not enough progressions for MDS!")
        return
    
    # JSON出力
    if export_json:
        script_dir = Path(__file__).parent.parent.parent
        output_dir = script_dir / "vis_system" / "scattergraph" / "data"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 直接距離行列ベースのMDS（すべてのコード進行間の距離を使用）
        print(f"\n{'='*60}")
        print("Computing MDS coordinates from direct distance matrix...")
        print(f"{'='*60}")
        
        # 直接距離行列を使ってMDS座標を計算
        coordinates = compute_mds_coordinates(progressions_data, n_components=2)
        
        # JSONファイル名を決定（1つのファイルにまとめる）
        json_output_path = str(output_dir / "mds_all.json")
        
        # 基準進行を特別表示するため、すべての基準進行を渡す
        if reference_progressions is None:
            reference_progressions = REFERENCE_PROGRESSIONS
        
        print(f"Exporting JSON to {json_output_path}...")
        export_to_json_format(
            progressions_data,
            coordinates,
            songs_list,
            json_output_path,
            reference_progressions=reference_progressions
        )
    
    # 画像可視化（オプション）
    if output_path:
        print(f"Creating visualization image...")
        visualize_mds_with_pie_charts(
            progressions_data,
            coordinates,
            output_path=output_path,
            show_lyrics=show_lyrics
        )
    
    print("Done!")

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/analyzed"
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    main(data_dir=data_dir, output_path=output_path)

