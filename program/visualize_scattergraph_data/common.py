"""
共通関数・定数（t-SNE / UMAP可視化用）
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict

from .musical_distance import compute_distance_matrix, circular_distance
from ..analyze_data.roman_numeral import section_to_roman_progression

# ============================================================================
# 基準進行（典型コード進行）の定義
# ============================================================================
# 新しい典型コード進行を追加する場合は、以下の手順に従ってください。
# 
# 【追加手順】
# 1. REFERENCE_PROGRESSIONS に新しいエントリを追加
#    - キー: 進行の識別子（英数字、小文字推奨、例: 'my_progression'）
#    - 値: ローマ数字のコード進行リスト（4つのコード、例: ['IV', 'V', 'I', 'vi']）
# 
# 2. REFERENCE_COLORS に対応する色を追加
#    - キー: REFERENCE_PROGRESSIONSと同じ識別子
#    - 値: 16進数カラーコード（例: '#FF5733'）
# 
# 3. フロントエンド（vis_system/scattergraph/index.html）も更新
#    - PROGRESSION_NAMES に表示名を追加（例: 'my_progression': 'マイ進行'）
#    - PROGRESSION_COLORS に色を追加（rgba形式、例: 'my_progression': 'rgba(255, 87, 51, 0.8)'）
# 
# 【動作】
# - 基準進行は、データに存在しない場合でも空のデータポイントとして自動的に追加されます
# - t-SNE/UMAPの座標計算に含まれ、散布図上で星（⭐）として表示されます
# - マウスオーバーで「⭐ XX進行」と表示されます
# ============================================================================

REFERENCE_PROGRESSIONS = {
    'odo': ['IV', 'V', 'iii', 'vi'],      # 王道進行: Ⅳ→Ⅴ→Ⅲm→Ⅵm
    'komuro': ['vi', 'IV', 'V', 'I'],     # 小室進行: VIm → IV → V → I
    'marusa': ['IVM7', 'III7', 'vi7', 'I7'],  # マルサ進行: ⅣM7 - Ⅲ7 - Ⅵm7 - Ⅰ7
}

# 基準進行の表示色（散布図での星の色、16進数カラーコード）
REFERENCE_COLORS = {
    'odo': '#FF9800',      # オレンジ（王道進行）
    'komuro': '#4CAF50',   # 緑（小室進行）
    'marusa': '#9C27B0',   # 紫（丸サ進行）
}

# プルチックの感情の輪 - 8つの基本感情の色定義
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

# 後方互換性のため（既存コードで使用されている可能性がある）
SENTIMENT_COLORS = {
    'positive': '#FF0000',   # 真っ赤（ポジティブ）
    'negative': '#0000FF',   # 真っ青（ネガティブ）
    'neutral': '#9E9E9E',   # グレー（中立）
}

def sentiment_to_color(sentiment_dict: Dict[str, Union[str, float]]) -> str:
    """
    センチメント辞書から色を決定
    """
    if not sentiment_dict:
        return '#808080'  # グレー（センチメントデータなし）
    
    label = sentiment_dict.get('label', '').lower()
    return SENTIMENT_COLORS.get(label, '#808080')

def emotion_to_color(emotion_dict: Dict[str, float]) -> str:
    """
    感情辞書から色を決定（8感情の最大値を使用）
    """
    # sentiment形式の場合はsentiment_to_colorに委譲
    if isinstance(emotion_dict, dict) and 'label' in emotion_dict:
        return sentiment_to_color(emotion_dict)
    
    # emotion形式の場合、最大値の感情の色を返す
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
            # emotionデータを優先、なければsentiment（後方互換性）
            emotion = section.get('emotion', {})
            sentiment = section.get('sentiment', {}) if not emotion else None
            
            # 歌詞がない場合はスキップ（歌詞数0を避けるため）
            if not lyric:
                continue
            
            # フィルタリング条件2: emotionまたはsentimentデータが必要
            # emotionの場合は最大値が0.55以上、sentimentの場合はscoreが0.55以上
            # 閾値0.55は、感情が明確に表現された高品質なフレーズのみを抽出し、
            # 計算時間を短縮しながら統計的信頼性も維持できる
            if emotion:
                # emotion形式: {JOY: 0.5, SADNESS: 0.3, ...}
                max_emotion_value = max(emotion.values()) if emotion.values() else 0.0
                if max_emotion_value < 0.55:
                    continue
            elif sentiment:
                # sentiment形式（後方互換性）: {'label': 'positive' or 'negative', 'score': 0.0-1.0}
                sentiment_score = sentiment.get('score', 0.0)
                if sentiment_score < 0.55:
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
                        'sentiment': sentiment,  # 後方互換性のため保持
                        'color': emotion_to_color(emotion) if emotion else sentiment_to_color(sentiment),
                        'song_index': len(songs_list) - 1,  # どの曲に属するか
                        'typical_chord_distance': section.get('typical_chord_distance', {})  # 基準進行との距離を保存
                    })
                    # typical_chord_distanceを更新（最小値を保持）
                    tcd = section.get('typical_chord_distance', {})
                    if tcd:
                        existing_tcd = prog_data.get('typical_chord_distance', {})
                        if not existing_tcd:
                            prog_data['typical_chord_distance'] = tcd.copy()
                        else:
                            # 各基準進行の最小距離を保持
                            for ref_name in ['odo', 'komuro', 'marusa']:
                                if ref_name in tcd and (ref_name not in existing_tcd or tcd[ref_name] < existing_tcd[ref_name]):
                                    existing_tcd[ref_name] = tcd[ref_name]
                    found = True
                    break
            
            if not found:
                # 新しいコード進行を追加
                lyrics_list = [{
                    'lyric': lyric,
                    'emotion': emotion,
                    'sentiment': sentiment,  # 後方互換性のため保持
                    'color': emotion_to_color(emotion) if emotion else sentiment_to_color(sentiment),
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

def add_reference_progressions(progressions_data: List[Dict], 
                                reference_progressions: Optional[Dict[str, List[str]]] = None) -> List[Dict]:
    """
    基準進行を空のデータポイントとして追加する
    
    この関数は、REFERENCE_PROGRESSIONSで定義された基準進行を、
    データに存在しない場合でも空のデータポイントとして追加します。
    追加された基準進行は、t-SNE/UMAPの座標計算に含まれ、
    散布図上で星（⭐）として表示されます。
    
    Args:
        progressions_data: 既存のコード進行データ
        reference_progressions: 基準進行の辞書（Noneの場合はREFERENCE_PROGRESSIONSを使用）
    
    Returns:
        基準進行を追加したコード進行データ（既に存在する場合はスキップ）
    """
    if reference_progressions is None:
        reference_progressions = REFERENCE_PROGRESSIONS
    
    # 各基準進行を処理
    for ref_name, ref_prog in reference_progressions.items():
        # 既存のデータに基準進行と一致するものがあるかチェック（循環性を考慮）
        found_existing = False
        for prog_data in progressions_data:
            roman_prog = prog_data.get('roman_progression', [])
            if roman_prog and is_same_progression(roman_prog, ref_prog):
                # 既存のデータに基準進行フラグを設定
                prog_data['isReferenceProgression'] = True
                prog_data['referenceProgressionName'] = ref_name
                # typical_chord_distanceも更新（自分自身なので距離0）
                if 'typical_chord_distance' not in prog_data:
                    prog_data['typical_chord_distance'] = {}
                prog_data['typical_chord_distance'][ref_name] = 0.0
                print(f"  Reference progression {ref_name} already exists in data, marking as reference...")
                found_existing = True
                break
        
        # 既に存在する場合はスキップ
        if found_existing:
            continue
        
        # 空のデータとして追加
        reference_prog_data = {
            'chord_progression': [],  # 空
            'normalized_chord_progression': [],  # 空
            'roman_progression': list(ref_prog),
            'lyrics': [],  # 空の歌詞リスト
            'typical_chord_distance': {ref_name: 0.0},  # 自分自身なので距離0
            'isReferenceProgression': True,
            'referenceProgressionName': ref_name
        }
        
        progressions_data.append(reference_prog_data)
        print(f"  Added reference progression {ref_name}: {' - '.join(ref_prog)}")
    
    return progressions_data

def compute_distance_vectors(
    progressions_data: List[Dict],
    reference_progressions: Dict[str, List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    各コード進行から基準進行への距離ベクトルを計算
    
    Args:
        progressions_data: コード進行データ
        reference_progressions: 基準進行の辞書（Noneの場合はデフォルトを使用）
    
    Returns:
        (距離ベクトルの配列（n×m、n=コード進行数、m=基準進行数）, 基準進行名のリスト)
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
                        {"label": "positive", "c": "#FF0000", "v": 25.0},
                        ...
                    ]
                },
                ...
            ],
            "emotionSongs": {
                "positive": [
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
    
    # sentimentごとの曲リストを構築
    emotion_songs = defaultdict(list)  # {sentiment_label: [song_info, ...]}
    
    points = []
    
    for i, (prog_data, coord) in enumerate(zip(progressions_data, coordinates)):
        lyrics = prog_data.get('lyrics', [])
        is_reference = prog_data.get('isReferenceProgression', False)
        
        # 基準進行の場合は空の歌詞でも処理する
        if not lyrics and not is_reference:
            continue
        
        # 座標を0-100に正規化
        x_norm = ((coord[0] - x_min) / x_range) * 100
        y_norm = ((coord[1] - y_min) / y_range) * 100
        
        # 基準進行の場合は空のpieデータを作成
        if is_reference:
            ref_name = prog_data.get('referenceProgressionName', '')
            roman_progression = prog_data.get('roman_progression', [])
            progression_str = ' - '.join(roman_progression) if roman_progression else 'N/A'
            
            point_data = {
                'x': float(round(x_norm, 2)),
                'y': float(round(y_norm, 2)),
                'r': float(round(12.0, 2)),  # 基準進行は大きめのサイズ
                'pie': [],  # 空のpieデータ
                'lyricCount': 0,
                'progression': progression_str,
                'roman_progression': roman_progression,
                'chord_progression': prog_data.get('chord_progression', []),
                'isReferenceProgression': True,
                'referenceProgressionName': ref_name,
                'progressionType': ref_name
            }
            if ref_name:
                point_data['strokeColor'] = REFERENCE_COLORS.get(ref_name, "#000000")
                point_data['referenceName'] = ref_name
            points.append(point_data)
            continue
        
        # 各歌詞のemotionデータから8感情のスコアを集計
        # emotion_groups: {emotion_name: [lyric_data, ...]}
        emotion_groups = defaultdict(list)  # {emotion_name: [lyric_data, ...]}
        emotion_song_indices = defaultdict(set)  # {emotion_name: {song_index, ...}}
        emotion_lyrics_map = defaultdict(list)  # {emotion_name: [{'lyric': lyric_text, 'song_index': song_index}, ...]}
        
        # 有効な歌詞のみをフィルタリング（emotionデータがあり、歌詞テキストもあるもの）
        # フィルタリング条件: emotionの最大値が0.55以上
        # 閾値0.55は、感情が明確に表現された高品質なフレーズのみを抽出し、
        # 計算時間を短縮しながら統計的信頼性も維持できる
        valid_lyrics = []
        for lyric_data in lyrics:
            emotion = lyric_data.get('emotion', {})
            # 後方互換性: emotionがない場合はsentimentから変換を試みる
            if not emotion:
                sentiment = lyric_data.get('sentiment', {})
                if sentiment:
                    # sentimentからemotionへの変換は行わず、スキップ
                    continue
            
            lyric_text = lyric_data.get('lyric', '').strip()
            
            # emotionデータと歌詞テキストの両方が必要
            if not emotion or not lyric_text:
                continue
            
            # emotionの最大値が0.55以上であることを確認
            max_emotion_value = max(emotion.values()) if emotion.values() else 0.0
            if max_emotion_value < 0.55:
                continue
            
            valid_lyrics.append(lyric_data)
        
        # 有効な歌詞がない場合はスキップ
        if not valid_lyrics:
            continue
        
        # 各歌詞のemotionデータから、各感情のスコアを集計
        for lyric_data in valid_lyrics:
            emotion = lyric_data.get('emotion', {})
            lyric_text = lyric_data.get('lyric', '').strip()
            song_index = lyric_data.get('song_index')
            
            # 各感情について、スコアが0.55以上のものをグループに追加
            for emotion_name, emotion_score in emotion.items():
                if emotion_score >= 0.55:
                    # 同じ歌詞が既にこのemotionグループに存在するかチェック（重複除去）
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
        
        # 各感情ごとのユニークな歌詞数を計算（重複を除去）
        emotion_unique_lyrics = {}
        for emotion_name in EMOTION_COLORS.keys():
            emotion_unique_set = set()
            for lyric_item in emotion_lyrics_map.get(emotion_name, []):
                emotion_unique_set.add(lyric_item['lyric'])
            emotion_unique_lyrics[emotion_name] = len(emotion_unique_set)
        
        # pieデータを作成（%で合計100になるように）
        pie_data = []
        for emotion_name in EMOTION_COLORS.keys():
            # 各感情のユニークな歌詞数を使用してパーセンテージを計算
            unique_count = emotion_unique_lyrics.get(emotion_name, 0)
            if unique_count > 0 and total_unique_lyrics > 0:
                percentage = (unique_count / total_unique_lyrics) * 100.0
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
                
                # このemotionに対応する曲を追加
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
        # ただし、負の値にならないようにする
        if pie_data:
            total_pie = sum(p['v'] for p in pie_data)
            if abs(total_pie - 100) > 0.01:  # 0.01%以上の誤差がある場合
                diff = 100 - total_pie
                if pie_data:
                    # 最後の要素にdiffを加えるが、負の値にならないようにする
                    new_value = pie_data[-1]['v'] + diff
                    pie_data[-1]['v'] = float(round(max(0.0, new_value), 2))
                    # もし負の値になった場合、他の要素から調整する
                    if new_value < 0:
                        # 負の値の分を他の要素から均等に減らす
                        negative_amount = abs(new_value)
                        pie_data[-1]['v'] = 0.0
                        # 正の値を持つ要素から均等に減らす
                        positive_items = [p for p in pie_data[:-1] if p['v'] > 0]
                        if positive_items:
                            per_item_reduction = negative_amount / len(positive_items)
                            for item in positive_items:
                                item['v'] = float(round(max(0.0, item['v'] - per_item_reduction), 2))
        
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
        
        # 基準進行と一致するかチェック、または最も近い基準進行を判定
        stroke_color = None
        matched_reference_name = None
        progression_type = None  # クラスタリング用
        
        # typical_chord_distanceを使って最も近い基準進行を判定
        tcd = prog_data.get('typical_chord_distance', {})
        if tcd:
            # 距離が最小の基準進行を探す
            min_distance = float('inf')
            closest_ref = None
            for ref_name in ['odo', 'komuro', 'marusa']:
                if ref_name in tcd:
                    distance = tcd[ref_name]
                    if distance < min_distance:
                        min_distance = distance
                        closest_ref = ref_name
            if closest_ref:
                progression_type = closest_ref
                stroke_color = REFERENCE_COLORS.get(closest_ref, "#000000")
                matched_reference_name = closest_ref
        
        # 完全一致もチェック（reference_progressionsが提供された場合）
        is_reference_progression = False
        reference_progression_name = None
        if reference_progressions and roman_progression:
            for ref_name, ref_prog in reference_progressions.items():
                if is_same_progression(roman_progression, ref_prog):
                    # 基準進行の色を取得（デフォルトは黒）
                    stroke_color = REFERENCE_COLORS.get(ref_name, "#000000")
                    matched_reference_name = ref_name
                    progression_type = ref_name
                    is_reference_progression = True
                    reference_progression_name = ref_name
                    print(f"  Found matching progression for {ref_name}: {progression_str}")
                    break
        
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
        
        # 基準進行の情報を追加
        if stroke_color:
            point_data['strokeColor'] = stroke_color
        if matched_reference_name:
            point_data['referenceName'] = matched_reference_name
        if progression_type:
            point_data['progressionType'] = progression_type
        if is_reference_progression:
            point_data['isReferenceProgression'] = True
            point_data['referenceProgressionName'] = reference_progression_name
        
        points.append(point_data)
    
    # sentimentごとの曲リストを辞書形式に変換
    emotion_songs_dict = {}
    for sentiment_label in SENTIMENT_COLORS.keys():
        emotion_songs_dict[sentiment_label] = emotion_songs.get(sentiment_label, [])
    
    result = {
        'points': points,
        'emotionSongs': emotion_songs_dict  # キー名は後方互換性のため保持
    }
    
    # JSONファイルに保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Exported JSON to {output_path}")
    return result

