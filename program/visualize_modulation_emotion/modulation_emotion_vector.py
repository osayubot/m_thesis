"""
転調前後の感情ベクトル可視化

転調前後のセクションの感情データを取得し、2D空間に投影して矢印で可視化する。
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')  # バックエンドを設定（GUI不要）
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib is not installed. Visualization will be skipped.")

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn is not installed. PCA projection will use simple method.")


# 感情ラベル（順序を保持）
EMOTION_LABELS = ["JOY", "SADNESS", "ANTICIPATION", "SURPRISE", "ANGER", "FEAR", "DISGUST", "TRUST"]

# 感情の色定義
EMOTION_COLORS = {
    'JOY': '#FFFF73',
    'SADNESS': '#5150F8',
    'ANTICIPATION': '#F3AB63',
    'SURPRISE': '#74BBF9',
    'ANGER': '#E93323',
    'FEAR': '#429429',
    'DISGUST': '#EB60F8',
    'TRUST': '#88FC6E',
}

# 転調タイプの色定義
MODULATION_TYPE_COLORS = {
    'semitone_up': '#FF6B6B',      # 赤（半音上）
    'tone_up': '#FFA500',          # オレンジ（全音上）
    'semitone_down': '#4ECDC4',    # シアン（半音下）
    'tone_down': '#95E1D3',        # ミント（全音下）
    'relative_major_minor': '#9B59B6',  # 紫（関係調：長調↔短調）
    'fifth_related': '#3498DB',    # 青（5度関係）
    'other': '#95A5A6',            # グレー（その他）
}


def key_to_note_value(key: str) -> Optional[int]:
    """
    キー名（例: "C", "C#", "Am", "Am#m"）を半音の値（0-11）に変換
    
    Returns:
        メジャーキーの場合: 0-11の値
        マイナーキーの場合: 0-11の値（短3度下）
        解析できない場合: None
    """
    if not key:
        return None
    
    # マイナーキーの場合は 'm' を除去
    is_minor = key.lower().endswith('m')
    base_key = key.rstrip('mM').strip()
    
    # ノート名から半音値を取得するマップ
    note_map = {
        'C': 0, 'C#': 1, 'Db': 1,
        'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4,
        'F': 5, 'F#': 6, 'Gb': 6,
        'G': 7, 'G#': 8, 'Ab': 8,
        'A': 9, 'A#': 10, 'Bb': 10,
        'B': 11
    }
    
    # ベースキーを取得
    if base_key in note_map:
        value = note_map[base_key]
        # マイナーキーの場合は短3度下（3半音下、つまり9半音上）
        if is_minor:
            value = (value + 9) % 12
        return value
    
    return None


def classify_modulation_type(from_key: str, to_key: str) -> str:
    """
    転調タイプを分類
    
    Args:
        from_key: 転調前のキー（例: "C", "Am"）
        to_key: 転調後のキー（例: "C#", "F"）
    
    Returns:
        転調タイプの文字列
    """
    from_val = key_to_note_value(from_key)
    to_val = key_to_note_value(to_key)
    
    if from_val is None or to_val is None:
        return 'other'
    
    # 半音関係を計算
    semitone_diff = (to_val - from_val) % 12
    if semitone_diff == 0:
        return 'other'  # 同じキー（マイナー↔メジャーかもしれないが、ここでは簡略化）
    
    # 半音上
    if semitone_diff == 1:
        return 'semitone_up'
    
    # 全音上
    if semitone_diff == 2:
        return 'tone_up'
    
    # 半音下（11半音上）
    if semitone_diff == 11:
        return 'semitone_down'
    
    # 全音下（10半音上）
    if semitone_diff == 10:
        return 'tone_down'
    
    # 関係調（長調↔短調）の判定
    # 相対調の関係: C ↔ Am, D ↔ Bm, E ↔ C#m など
    # ベースノートを取得して比較
    from_is_minor = from_key.lower().endswith('m')
    to_is_minor = to_key.lower().endswith('m')
    if from_is_minor != to_is_minor:
        # ベースノート名を取得（例: "C" や "Am" から "C" や "A" を抽出）
        from_base_key = from_key.rstrip('mM').strip()
        to_base_key = to_key.rstrip('mM').strip()
        
        # ベースノートが同じか、または3半音の関係があるかチェック
        from_base_val = key_to_note_value(from_base_key)
        to_base_val = key_to_note_value(to_base_key)
        
        if from_base_val is not None and to_base_val is not None:
            # 相対調の関係: 長調の3半音下が相対短調のベース
            # 例: C(0)の相対短調はAmで、AmのベースはA(9) = C(0) + 9 mod 12 = 9
            # しかし、A(9) - 3 = 6でC(0)にはならない...
            # 正しくは: Cメジャーの相対短調はAmで、AmのベースはA(9)、C(0) + 3 = 3 = D# ≠ A
            # 実際の相対調: C ↔ Am は同じキーシグネチャを持つが、ベースノートは異なる
            # 簡略化: ベースノートが同じで長調↔短調の関係がある場合を関係調とみなす
            # （例: C ↔ Cm は同主調だが、ここでは関係調として扱わない）
            # 実際の相対調判定は複雑なので、semitone_diffが3または9の場合に限定
            base_semitone_diff = (to_base_val - from_base_val) % 12
            if base_semitone_diff == 3 or base_semitone_diff == 9:
                return 'relative_major_minor'
    
    # 5度関係（7半音 = 完全5度上、5半音 = 完全4度上 = 完全5度下）
    if semitone_diff == 7 or semitone_diff == 5:
        return 'fifth_related'
    
    return 'other'


def extract_emotion_vector(emotion_dict: Dict[str, float]) -> np.ndarray:
    """
    感情辞書を8次元ベクトルに変換
    
    Args:
        emotion_dict: 感情スコアの辞書 {JOY: 0.5, SADNESS: 0.3, ...}
    
    Returns:
        8次元のnumpy配列（EMOTION_LABELSの順序で）
    """
    vector = np.zeros(len(EMOTION_LABELS))
    for i, label in enumerate(EMOTION_LABELS):
        vector[i] = emotion_dict.get(label, 0.0)
    return vector


def load_modulation_data(data_dir: str) -> List[Dict]:
    """
    分析済みデータから転調情報と感情データを抽出
    
    Args:
        data_dir: 分析済みデータのディレクトリパス
    
    Returns:
        転調情報のリスト。各要素は以下のキーを持つ:
        - song_id: 楽曲ID
        - song_title: 楽曲タイトル
        - from_key: 転調前のキー
        - to_key: 転調後のキー
        - modulation_type: 転調タイプ
        - before_emotion: 転調前の感情ベクトル（8次元）
        - after_emotion: 転調後の感情ベクトル（8次元）
        - section_before: 転調前のセクションインデックス
        - section_after: 転調後のセクションインデックス
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    modulation_events = []
    
    # すべてのJSONファイルを読み込む
    json_files = list(data_path.glob("*.json"))
    print(f"Loading {len(json_files)} analyzed song files...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                song = json.load(f)
            
            # セクション情報を取得（analyzed_chord_progressions_and_lyricsフィールドを使用）
            sections = song.get('analyzed_chord_progressions_and_lyrics', [])
            if len(sections) < 2:
                continue
            
            song_id = song.get('spotify_id', json_file.stem)
            song_title = song.get('title', 'Unknown')
            
            # 連続するセクション間で転調を検出
            for i in range(len(sections) - 1):
                sec_before = sections[i]
                sec_after = sections[i + 1]
                
                key_before = sec_before.get('key')
                key_after = sec_after.get('key')
                
                # キーが異なり、両方とも有効な値の場合
                if key_before and key_after and key_before != key_after:
                    emotion_before = sec_before.get('emotion', {})
                    emotion_after = sec_after.get('emotion', {})
                    
                    # 感情データが有効か確認（少なくとも1つの感情が0より大きい）
                    if emotion_before and emotion_after:
                        has_valid_before = any(v > 0 for v in emotion_before.values())
                        has_valid_after = any(v > 0 for v in emotion_after.values())
                        
                        if has_valid_before and has_valid_after:
                            modulation_type = classify_modulation_type(key_before, key_after)
                            
                            before_vector = extract_emotion_vector(emotion_before)
                            after_vector = extract_emotion_vector(emotion_after)
                            
                            modulation_events.append({
                                'song_id': song_id,
                                'song_title': song_title,
                                'from_key': key_before,
                                'to_key': key_after,
                                'modulation_type': modulation_type,
                                'before_emotion': before_vector,
                                'after_emotion': after_vector,
                                'section_before': i,
                                'section_after': i + 1,
                            })
        
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue
    
    print(f"Found {len(modulation_events)} modulation events with valid emotion data")
    return modulation_events


def project_to_2d(emotion_vectors: np.ndarray, method: str = 'pca') -> Tuple[np.ndarray, Optional[object]]:
    """
    感情ベクトルを2D空間に投影
    
    Args:
        emotion_vectors: (N, 8)のnumpy配列
        method: 投影方法 ('pca' または 'joy_sadness')
    
    Returns:
        (2D座標 (N, 2), 投影モデル（PCAの場合はPCAオブジェクト、それ以外はNone）)
    """
    if method == 'pca':
        if HAS_SKLEARN:
            pca = PCA(n_components=2)
            coords_2d = pca.fit_transform(emotion_vectors)
            return coords_2d, pca
        else:
            # sklearnがない場合の簡易実装（共分散行列の固有値分解）
            # ここでは簡略化してJOY-SADNESS軸を使用
            method = 'joy_sadness'
    
    if method == 'joy_sadness':
        # JOY-SADNESS軸を使用
        joy_idx = EMOTION_LABELS.index('JOY')
        sadness_idx = EMOTION_LABELS.index('SADNESS')
        coords_2d = emotion_vectors[:, [joy_idx, sadness_idx]]
        return coords_2d, None
    
    raise ValueError(f"Unknown projection method: {method}")


def visualize_modulation_emotion_vectors(
    modulation_events: List[Dict],
    output_path: str,
    projection_method: str = 'pca',
    arrow_alpha: float = 0.6,
    point_size: int = 50,
    arrow_width: float = 0.001,
    max_arrows: Optional[int] = None,
):
    """
    転調前後の感情ベクトルを矢印で可視化
    
    Args:
        modulation_events: load_modulation_data()で取得した転調イベントのリスト
        output_path: 出力画像のパス
        projection_method: 投影方法 ('pca' または 'joy_sadness')
        arrow_alpha: 矢印の透明度（0.0-1.0）
        point_size: 点のサイズ
        arrow_width: 矢印の幅
        max_arrows: 表示する最大矢印数（Noneの場合はすべて表示）
    """
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for visualization")
        return
    
    if not modulation_events:
        print("No modulation events to visualize")
        return
    
    # 転調タイプごとにグループ化
    events_by_type = defaultdict(list)
    for event in modulation_events:
        events_by_type[event['modulation_type']].append(event)
    
    print(f"Modulation events by type:")
    for mod_type, events in events_by_type.items():
        print(f"  {mod_type}: {len(events)} events")
    
    # すべての感情ベクトルを収集
    all_before = np.array([e['before_emotion'] for e in modulation_events])
    all_after = np.array([e['after_emotion'] for e in modulation_events])
    all_vectors = np.vstack([all_before, all_after])
    
    # 2D投影
    coords_2d, projection_model = project_to_2d(all_vectors, method=projection_method)
    before_coords = coords_2d[:len(modulation_events)]
    after_coords = coords_2d[len(modulation_events):]
    
    # プロット設定
    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    
    # 転調タイプごとに描画
    for mod_type, events in events_by_type.items():
        color = MODULATION_TYPE_COLORS.get(mod_type, '#95A5A6')
        
        # このタイプのインデックスを取得
        type_indices = [i for i, e in enumerate(modulation_events) if e['modulation_type'] == mod_type]
        
        if max_arrows and len(type_indices) > max_arrows:
            # ランダムにサンプリング
            import random
            type_indices = random.sample(type_indices, max_arrows)
        
        for idx in type_indices:
            event = modulation_events[idx]
            before_xy = before_coords[idx]
            after_xy = after_coords[idx]
            
            # 矢印を描画
            dx = after_xy[0] - before_xy[0]
            dy = after_xy[1] - before_xy[1]
            
            ax.arrow(
                before_xy[0], before_xy[1],
                dx, dy,
                head_width=arrow_width * max(abs(dx), abs(dy)),
                head_length=arrow_width * max(abs(dx), abs(dy)) * 1.5,
                fc=color,
                ec=color,
                alpha=arrow_alpha,
                length_includes_head=True,
                linewidth=1.5,
            )
            
            # 転調前の点を描画
            ax.scatter(
                before_xy[0], before_xy[1],
                c=color,
                s=point_size * 0.7,
                alpha=0.8,
                edgecolors='black',
                linewidths=0.5,
            )
            
            # 転調後の点を描画
            ax.scatter(
                after_xy[0], after_xy[1],
                c=color,
                s=point_size,
                alpha=0.9,
                edgecolors='black',
                linewidths=1.0,
                marker='s',  # 四角
            )
    
    # 軸ラベル
    if projection_method == 'pca':
        variance_explained = None
        if projection_model and hasattr(projection_model, 'explained_variance_ratio_'):
            variance_explained = projection_model.explained_variance_ratio_
            ax.set_xlabel(f'PC1 ({variance_explained[0]*100:.1f}% variance)', fontsize=12)
            ax.set_ylabel(f'PC2 ({variance_explained[1]*100:.1f}% variance)', fontsize=12)
        else:
            ax.set_xlabel('PC1', fontsize=12)
            ax.set_ylabel('PC2', fontsize=12)
    elif projection_method == 'joy_sadness':
        ax.set_xlabel('JOY', fontsize=12)
        ax.set_ylabel('SADNESS', fontsize=12)
    
    ax.set_title('Emotion Vector Changes Before and After Modulation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 凡例
    legend_elements = [
        mpatches.Patch(color=color, label=mod_type.replace('_', ' ').title())
        for mod_type, color in MODULATION_TYPE_COLORS.items()
        if mod_type in events_by_type
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()


if __name__ == '__main__':
    # テスト実行
    import sys
    if len(sys.argv) < 2:
        print("Usage: python modulation_emotion_vector.py <data_dir> [output_path]")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'modulation_emotion_vectors.png'
    
    events = load_modulation_data(data_dir)
    visualize_modulation_emotion_vectors(events, output_path)

