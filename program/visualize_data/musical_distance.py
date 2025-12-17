"""
音楽的レーベンシュタイン距離の実装
機能的な近さ、循環性を考慮
（文脈調整は無効化: use_context=False）
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

# 機能分類（Tonic, Predominant, Dominant）
FUNCTIONAL_GROUPS = {
    'T': ['i', 'iii', 'vi', 'I', 'III', 'VI'],  # Tonic
    'PD': ['ii', 'iv', 'II', 'IV'],              # Predominant
    'D': ['v', 'vii', 'V', 'VII', 'V7', 'v7'],  # Dominant
}

# 機能マッピング（ローマ数字から機能へ）
def get_function(roman: str) -> Optional[str]:
    """ローマ数字から機能を取得"""
    # 品質記号を除去（7, sus, °, oなど）
    roman_base = roman.rstrip('7sus°o').strip()
    
    # 大文字小文字を統一して比較
    roman_lower = roman_base.lower()
    
    if roman_lower in ['i', 'iii', 'vi']:
        return 'T'
    elif roman_lower in ['ii', 'iv']:
        return 'PD'
    elif roman_lower in ['v', 'vii']:
        return 'D'
    return None

def functional_similarity_cost(a: str, b: str) -> float:
    """機能的な近さに基づく置換コスト"""
    if a == b:
        return 0.0
    
    func_a = get_function(a)
    func_b = get_function(b)
    
    if func_a == func_b and func_a is not None:
        return 0.2  # 同機能
    elif func_a is None or func_b is None:
        return 1.0  # 機能不明
    else:
        # 近傍機能（T-PD, PD-D）は中程度のコスト
        if (func_a == 'T' and func_b == 'PD') or (func_a == 'PD' and func_b == 'T'):
            return 0.5
        elif (func_a == 'PD' and func_b == 'D') or (func_a == 'D' and func_b == 'PD'):
            return 0.5
        elif (func_a == 'T' and func_b == 'D') or (func_a == 'D' and func_b == 'T'):
            return 0.8  # T-Dは遠い
        return 1.0

def is_natural_progression(prev: Optional[str], curr: str, next_chord: Optional[str]) -> bool:
    """自然な進行かどうかを判定（T→PD→D→Tなど）"""
    if prev is None or next_chord is None:
        return True  # 文脈がない場合は自然とみなす
    
    func_prev = get_function(prev)
    func_curr = get_function(curr)
    func_next = get_function(next_chord)
    
    if func_prev is None or func_curr is None or func_next is None:
        return True
    
    # 自然な流れ: T→PD→D→T
    natural_flows = [
        ('T', 'PD', 'D'),
        ('PD', 'D', 'T'),
        ('D', 'T', 'PD'),
        ('T', 'T', 'PD'),  # 同じ機能の継続も自然
        ('PD', 'PD', 'D'),
        ('D', 'D', 'T'),
    ]
    
    return (func_prev, func_curr, func_next) in natural_flows

def contextual_substitution_cost(
    a: str, b: str, 
    prev_a: Optional[str] = None, 
    next_a: Optional[str] = None,
    prev_b: Optional[str] = None,
    next_b: Optional[str] = None
) -> float:
    """文脈を考慮した置換コスト"""
    base_cost = functional_similarity_cost(a, b)
    
    # 文脈が自然な場合はコストを下げる
    if is_natural_progression(prev_a, a, next_a) and is_natural_progression(prev_b, b, next_b):
        return base_cost * 0.9
    elif not is_natural_progression(prev_a, a, next_a) or not is_natural_progression(prev_b, b, next_b):
        return base_cost * 1.1
    
    return base_cost

def musical_levenshtein_distance(
    seq1: List[str], 
    seq2: List[str],
    use_context: bool = True
) -> float:
    """
    音楽的レーベンシュタイン距離を計算
    
    Args:
        seq1: コード進行1（ローマ数字のリスト）
        seq2: コード進行2（ローマ数字のリスト）
        use_context: 文脈を考慮するか
    
    Returns:
        距離（0以上）
    """
    m, n = len(seq1), len(seq2)
    
    # 空の場合は長さ分のコスト
    if m == 0:
        return float(n)
    if n == 0:
        return float(m)
    
    # DPテーブル
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    
    # 初期化
    for i in range(m + 1):
        dp[i][0] = float(i) * 0.5  # 削除コスト（繰り返し・装飾は低コスト）
    for j in range(n + 1):
        dp[0][j] = float(j) * 0.5  # 挿入コスト
    
    # 動的計画法
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # 置換コスト
            if use_context:
                prev_a = seq1[i-2] if i > 1 else None
                next_a = seq1[i] if i < m else None
                prev_b = seq2[j-2] if j > 1 else None
                next_b = seq2[j] if j < n else None
                sub_cost = contextual_substitution_cost(
                    seq1[i-1], seq2[j-1],
                    prev_a, next_a, prev_b, next_b
                )
            else:
                sub_cost = functional_similarity_cost(seq1[i-1], seq2[j-1])
            
            # 削除コスト（構造的欠落は高コスト、繰り返しは低コスト）
            del_cost = 0.5 if i > 1 and seq1[i-1] == seq1[i-2] else 0.7
            
            # 挿入コスト
            ins_cost = 0.5 if j > 1 and seq2[j-1] == seq2[j-2] else 0.7
            
            dp[i][j] = min(
                dp[i-1][j] + del_cost,      # 削除
                dp[i][j-1] + ins_cost,      # 挿入
                dp[i-1][j-1] + sub_cost     # 置換
            )
    
    return dp[m][n]

def circular_distance(seq1: List[str], seq2: List[str]) -> float:
    """
    循環性を考慮した距離（全回転を試して最小距離を返す）
    文脈調整は無効化（use_context=False）
    """
    if len(seq1) == 0 or len(seq2) == 0:
        return musical_levenshtein_distance(seq1, seq2, use_context=False)
    
    min_dist = float('inf')
    
    # seq1の全回転を試す
    for i in range(len(seq1)):
        rotated = seq1[i:] + seq1[:i]
        dist = musical_levenshtein_distance(rotated, seq2, use_context=False)
        min_dist = min(min_dist, dist)
    
    # seq2の全回転も試す（必要に応じて）
    for i in range(len(seq2)):
        rotated = seq2[i:] + seq2[:i]
        dist = musical_levenshtein_distance(seq1, rotated, use_context=False)
        min_dist = min(min_dist, dist)
    
    return min_dist

def compute_distance_matrix(progressions: List[List[str]]) -> np.ndarray:
    """
    コード進行のリストから距離行列を計算
    
    Args:
        progressions: ローマ数字のコード進行のリスト
    
    Returns:
        距離行列（n×n）
    """
    n = len(progressions)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = circular_distance(progressions[i], progressions[j])
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
    
    return dist_matrix

