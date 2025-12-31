"""
音楽的レーベンシュタイン距離の実装
機能的な近さ、循環性、テンションを考慮
（文脈調整は無効化: use_context=False）
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import re
import time
from multiprocessing import Pool, cpu_count
from functools import partial

# 機能分類（Tonic, Predominant, Dominant）
FUNCTIONAL_GROUPS = {
    'T': ['i', 'iii', 'vi', 'I', 'III', 'VI'],  # Tonic
    'PD': ['ii', 'iv', 'II', 'IV'],              # Predominant
    'D': ['v', 'vii', 'V', 'VII', 'V7', 'v7'],  # Dominant
}

# テンションの不協和度レベル（高いほど複雑）
TENSION_COMPLEXITY = {
    'none': 0,      # テンションなし（トライアド）
    '7': 1,         # 7th（M7, m7, 7）
    'sus': 1,       # sus2, sus4
    'add': 1.5,     # add9, add11
    '9': 2,         # 9th
    '11': 2.5,      # 11th
    '13': 3,        # 13th
    'aug': 1.5,     # augmented
    'dim': 1.5,     # diminished
}


def parse_roman_numeral(roman: str) -> Tuple[str, str, float]:
    """
    ローマ数字を解析して、基本部分とテンション情報を抽出
    
    Args:
        roman: ローマ数字文字列（例: "IVM7", "V9", "vim7", "IVadd9"）
    
    Returns:
        (基本ローマ数字, テンションタイプ, 不協和度)
    """
    if not roman:
        return ('', 'none', 0)
    
    original = roman
    
    # テンションパターンを順番に検出（長いものから）
    tension_patterns = [
        (r'(add\d+)', 'add'),      # add9, add11, add13
        (r'(sus\d*)', 'sus'),      # sus, sus2, sus4
        (r'(M7|maj7)', '7'),       # Major 7th
        (r'(m7|min7)', '7'),       # minor 7th
        (r'13', '13'),             # 13th
        (r'11', '11'),             # 11th
        (r'9', '9'),               # 9th
        (r'7', '7'),               # dominant 7th
        (r'(aug|\+)', 'aug'),      # augmented
        (r'(dim|°|o)', 'dim'),     # diminished
    ]
    
    tension_type = 'none'
    complexity = 0
    
    # テンションを検出
    for pattern, t_type in tension_patterns:
        if re.search(pattern, roman, re.IGNORECASE):
            tension_type = t_type
            complexity = TENSION_COMPLEXITY.get(t_type, 0)
            break
    
    # テンション・品質記号をすべて除去して基本ローマ数字を抽出
    # 正規表現で除去
    base = re.sub(
        r'(M7|m7|maj7|min7|add\d+|sus\d*|aug|dim|°|o|\+|13|11|9|7)+',
        '',
        roman,
        flags=re.IGNORECASE
    ).strip()
    
    # 空になった場合は元の文字列の先頭部分を使用
    if not base:
        # ローマ数字部分を抽出（I, II, III, IV, V, VI, VII）
        match = re.match(r'^(i{1,3}|iv|vi{0,2}|v|vii?|I{1,3}|IV|VI{0,2}|V|VII?)', roman, re.IGNORECASE)
        if match:
            base = match.group(1)
        else:
            base = roman
    
    return (base, tension_type, complexity)


def get_function(roman: str) -> Optional[str]:
    """ローマ数字から機能を取得（テンション対応版）"""
    # 基本ローマ数字を抽出
    base, _, _ = parse_roman_numeral(roman)
    
    # 大文字小文字を統一して比較
    roman_lower = base.lower()
    
    if roman_lower in ['i', 'iii', 'vi']:
        return 'T'
    elif roman_lower in ['ii', 'iv']:
        return 'PD'
    elif roman_lower in ['v', 'vii']:
        return 'D'
    return None


def tension_similarity_cost(tension_a: str, complexity_a: float, 
                           tension_b: str, complexity_b: float) -> float:
    """
    テンションの類似性に基づくコスト
    
    Returns:
        0.0〜0.15 のコスト
    """
    # 同じテンションタイプ
    if tension_a == tension_b:
        return 0.0
    
    # 両方テンションなし
    if tension_a == 'none' and tension_b == 'none':
        return 0.0
    
    # 片方だけテンションあり
    if tension_a == 'none' or tension_b == 'none':
        # 不協和度の差に基づくコスト（最大0.15）
        complexity_diff = abs(complexity_a - complexity_b)
        return min(0.15, complexity_diff * 0.05)
    
    # 両方テンションあり（異なるタイプ）
    # 不協和度の差に基づくコスト
    complexity_diff = abs(complexity_a - complexity_b)
    return min(0.1, complexity_diff * 0.03)


def functional_similarity_cost(a: str, b: str, consider_tension: bool = True) -> float:
    """
    機能的な近さに基づく置換コスト（テンション考慮版）
    
    Args:
        a: コード1（ローマ数字）
        b: コード2（ローマ数字）
        consider_tension: テンションを考慮するか
    
    Returns:
        置換コスト（0.0〜1.0）
    """
    if a == b:
        return 0.0
    
    # テンション情報を抽出
    base_a, tension_a, complexity_a = parse_roman_numeral(a)
    base_b, tension_b, complexity_b = parse_roman_numeral(b)
    
    # 基本ローマ数字が同じ場合（テンションのみ異なる）
    if base_a.lower() == base_b.lower():
        if consider_tension:
            return tension_similarity_cost(tension_a, complexity_a, tension_b, complexity_b)
        else:
            return 0.0
    
    # 機能を取得
    func_a = get_function(a)
    func_b = get_function(b)
    
    # 基本コスト（機能に基づく）
    if func_a == func_b and func_a is not None:
        base_cost = 0.2  # 同機能
    elif func_a is None or func_b is None:
        base_cost = 1.0  # 機能不明
    else:
        # 近傍機能（T-PD, PD-D）は中程度のコスト
        if (func_a == 'T' and func_b == 'PD') or (func_a == 'PD' and func_b == 'T'):
            base_cost = 0.5
        elif (func_a == 'PD' and func_b == 'D') or (func_a == 'D' and func_b == 'PD'):
            base_cost = 0.5
        elif (func_a == 'T' and func_b == 'D') or (func_a == 'D' and func_b == 'T'):
            base_cost = 0.8  # T-Dは遠い
        else:
            base_cost = 1.0
    
    # テンションコストを追加
    if consider_tension:
        tension_cost = tension_similarity_cost(tension_a, complexity_a, tension_b, complexity_b)
        return base_cost + tension_cost
    
    return base_cost

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
    next_b: Optional[str] = None,
    consider_tension: bool = True
) -> float:
    """文脈を考慮した置換コスト"""
    base_cost = functional_similarity_cost(a, b, consider_tension=consider_tension)
    
    # 文脈が自然な場合はコストを下げる
    if is_natural_progression(prev_a, a, next_a) and is_natural_progression(prev_b, b, next_b):
        return base_cost * 0.9
    elif not is_natural_progression(prev_a, a, next_a) or not is_natural_progression(prev_b, b, next_b):
        return base_cost * 1.1
    
    return base_cost

def musical_levenshtein_distance(
    seq1: List[str], 
    seq2: List[str],
    use_context: bool = True,
    consider_tension: bool = True
) -> float:
    """
    音楽的レーベンシュタイン距離を計算（テンション考慮版）
    
    Args:
        seq1: コード進行1（ローマ数字のリスト）
        seq2: コード進行2（ローマ数字のリスト）
        use_context: 文脈を考慮するか
        consider_tension: テンションを考慮するか（デフォルト: True）
    
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
                    prev_a, next_a, prev_b, next_b,
                    consider_tension=consider_tension
                )
            else:
                sub_cost = functional_similarity_cost(
                    seq1[i-1], seq2[j-1],
                    consider_tension=consider_tension
                )
            
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

def circular_distance(seq1: List[str], seq2: List[str], consider_tension: bool = True) -> float:
    """
    距離計算（循環性の考慮は削除して計算時間を短縮）
    文脈調整は無効化（use_context=False）
    
    注意: 循環性を考慮しないため、同じ進行でも開始位置が違う場合（例: IV-V-iii-vi と V-iii-vi-IV）
    は正確に比較できませんが、計算時間が約87.5%短縮されます。
    
    Args:
        seq1: コード進行1
        seq2: コード進行2
        consider_tension: テンションを考慮するか（デフォルト: True）
    """
    # 循環性の計算を削除して、直接距離を計算（約8倍高速化）
        return musical_levenshtein_distance(seq1, seq2, use_context=False, consider_tension=consider_tension)

def _compute_pair_distance(args: Tuple[int, int, List[str], List[str]]) -> Tuple[int, int, float]:
    """
    1ペアの距離を計算（並列化用のヘルパー関数）
    
    Args:
        args: (i, j, prog_i, prog_j) のタプル
    
    Returns:
        (i, j, distance) のタプル
    """
    i, j, prog_i, prog_j = args
    dist = circular_distance(prog_i, prog_j)
    return (i, j, dist)

def compute_distance_matrix(progressions: List[List[str]], show_progress: bool = True, n_jobs: Optional[int] = None) -> np.ndarray:
    """
    コード進行のリストから距離行列を計算（並列化対応）
    
    Args:
        progressions: ローマ数字のコード進行のリスト
        show_progress: 進捗を表示するか
        n_jobs: 並列化するプロセス数（Noneの場合はCPUコア数を使用）
    
    Returns:
        距離行列（n×n）
    """
    n = len(progressions)
    dist_matrix = np.zeros((n, n))
    
    # 総計算回数（対称行列なので半分）
    total_pairs = n * (n - 1) // 2
    
    if show_progress:
        print(f"Computing distance matrix for {n} progressions...")
        print(f"Total pairs to compute: {total_pairs:,}")
        if n_jobs is None or n_jobs > 1:
            num_cores = cpu_count() if n_jobs is None else n_jobs
            print(f"Using {num_cores} CPU cores for parallel computation...")
        print("This may take a while...")
    
    start_time = time.time()
    
    # 並列化するかどうかを決定
    if n_jobs == 1 or (n_jobs is None and total_pairs < 10000):
        # 小規模データまたは明示的にシリアル実行を指定された場合
        computed = 0
        last_progress = -1
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = circular_distance(progressions[i], progressions[j])
                dist_matrix[i][j] = dist
                dist_matrix[j][i] = dist
                
                computed += 1
                # 5%ごとに進捗を表示
                progress = int(100 * computed / total_pairs)
                if show_progress and progress != last_progress and progress % 5 == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    rate = computed / elapsed if elapsed > 0 else 0
                    remaining = (total_pairs - computed) / rate if rate > 0 else 0
                    
                    print(f"  Progress: {progress}% ({computed:,}/{total_pairs:,} pairs computed, "
                          f"elapsed: {elapsed:.1f}s, estimated remaining: {remaining:.1f}s)")
                    last_progress = progress
    else:
        # 並列化実行
        num_cores = cpu_count() if n_jobs is None else n_jobs
        
        # すべてのペアを準備
        pairs = [(i, j, progressions[i], progressions[j]) 
                 for i in range(n) for j in range(i + 1, n)]
        
        # 並列計算
        with Pool(processes=num_cores) as pool:
            results = []
            completed = 0
            last_progress = -1
            
            # チャンクサイズを調整（進捗表示のため）
            chunk_size = max(1, len(pairs) // (num_cores * 10))
            
            for result in pool.imap(_compute_pair_distance, pairs, chunksize=chunk_size):
                i, j, dist = result
                dist_matrix[i][j] = dist
                dist_matrix[j][i] = dist
                
                completed += 1
                # 5%ごとに進捗を表示
                progress = int(100 * completed / total_pairs)
                if show_progress and progress != last_progress and progress % 5 == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (total_pairs - completed) / rate if rate > 0 else 0
                    
                    print(f"  Progress: {progress}% ({completed:,}/{total_pairs:,} pairs computed, "
                          f"elapsed: {elapsed:.1f}s, estimated remaining: {remaining:.1f}s)")
                    last_progress = progress
    
    if show_progress:
        total_time = time.time() - start_time
        print(f"  Completed: 100% ({total_pairs:,} pairs computed in {total_time:.1f}s)")
    
    return dist_matrix

