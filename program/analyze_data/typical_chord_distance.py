"""
Typical chord progression distance utilities.

For each section's `normalized_chord_progression` (already transposed to C or Am),
compute musical distance to representative J-POP progressions:
- odo (王道進行): IV - V - iii - vi
- komuro (小室進行): vi - IV - V - I
- marusa (丸サ進行): IVM7 - III7 - vi7 - I7

Output format:
  typical_chord_distance: {"odo": 0.2, "komuro": 1.4, "marusa": 2.4}
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import re

from program.analyze_data.roman_numeral import chord_to_roman

# ---------------------------------------------------------------------
# Minimal "musical distance" implementation (integrated from the previous
# analyze_data/circular_distance.py). We only keep what we actually use here:
# - musical Levenshtein distance with functional similarity + tension
# - circular (rotation) minimization
# (Context-aware cost is intentionally omitted; we always use use_context=False.)
# ---------------------------------------------------------------------

TENSION_COMPLEXITY = {
    "none": 0.0,
    "7": 1.0,
    "sus": 1.0,
    "add": 1.5,
    "9": 2.0,
    "11": 2.5,
    "13": 3.0,
    "aug": 1.5,
    "dim": 1.5,
}


def _parse_roman_numeral(roman: str) -> Tuple[str, str, float]:
    if not roman:
        return ("", "none", 0.0)

    tension_patterns = [
        (r"(add\d+)", "add"),
        (r"(sus\d*)", "sus"),
        (r"(M7|maj7)", "7"),
        (r"(m7|min7)", "7"),
        (r"13", "13"),
        (r"11", "11"),
        (r"9", "9"),
        (r"7", "7"),
        (r"(aug|\+)", "aug"),
        (r"(dim|°|o)", "dim"),
    ]

    tension_type = "none"
    complexity = 0.0
    for pattern, t_type in tension_patterns:
        if re.search(pattern, roman, re.IGNORECASE):
            tension_type = t_type
            complexity = float(TENSION_COMPLEXITY.get(t_type, 0.0))
            break

    base = re.sub(
        r"(M7|m7|maj7|min7|add\d+|sus\d*|aug|dim|°|o|\+|13|11|9|7)+",
        "",
        roman,
        flags=re.IGNORECASE,
    ).strip()

    if not base:
        match = re.match(
            r"^(i{1,3}|iv|vi{0,2}|v|vii?|I{1,3}|IV|VI{0,2}|V|VII?)",
            roman,
            re.IGNORECASE,
        )
        base = match.group(1) if match else roman

    return (base, tension_type, float(complexity))


def _get_function(roman: str) -> Optional[str]:
    base, _, _ = _parse_roman_numeral(roman)
    r = base.lower()
    if r in ["i", "iii", "vi"]:
        return "T"
    if r in ["ii", "iv"]:
        return "PD"
    if r in ["v", "vii"]:
        return "D"
    return None


def _tension_similarity_cost(tension_a: str, complexity_a: float, tension_b: str, complexity_b: float) -> float:
    if tension_a == tension_b:
        return 0.0
    if tension_a == "none" and tension_b == "none":
        return 0.0
    if tension_a == "none" or tension_b == "none":
        return min(0.15, abs(complexity_a - complexity_b) * 0.05)
    return min(0.1, abs(complexity_a - complexity_b) * 0.03)


def _functional_similarity_cost(a: str, b: str, consider_tension: bool = True) -> float:
    if a == b:
        return 0.0

    base_a, tension_a, complexity_a = _parse_roman_numeral(a)
    base_b, tension_b, complexity_b = _parse_roman_numeral(b)

    if base_a.lower() == base_b.lower():
        return _tension_similarity_cost(tension_a, complexity_a, tension_b, complexity_b) if consider_tension else 0.0

    func_a = _get_function(a)
    func_b = _get_function(b)

    if func_a == func_b and func_a is not None:
        base_cost = 0.2
    elif func_a is None or func_b is None:
        base_cost = 1.0
    else:
        if (func_a == "T" and func_b == "PD") or (func_a == "PD" and func_b == "T"):
            base_cost = 0.5
        elif (func_a == "PD" and func_b == "D") or (func_a == "D" and func_b == "PD"):
            base_cost = 0.5
        elif (func_a == "T" and func_b == "D") or (func_a == "D" and func_b == "T"):
            base_cost = 0.8
        else:
            base_cost = 1.0

    if consider_tension:
        return base_cost + _tension_similarity_cost(tension_a, complexity_a, tension_b, complexity_b)
    return base_cost


def _musical_levenshtein_distance(seq1: List[str], seq2: List[str], consider_tension: bool = True) -> float:
    m, n = len(seq1), len(seq2)
    if m == 0:
        return float(n)
    if n == 0:
        return float(m)

    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = float(i) * 0.5
    for j in range(n + 1):
        dp[0][j] = float(j) * 0.5

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub_cost = _functional_similarity_cost(seq1[i - 1], seq2[j - 1], consider_tension=consider_tension)
            del_cost = 0.5 if i > 1 and seq1[i - 1] == seq1[i - 2] else 0.7
            ins_cost = 0.5 if j > 1 and seq2[j - 1] == seq2[j - 2] else 0.7
            dp[i][j] = min(
                dp[i - 1][j] + del_cost,
                dp[i][j - 1] + ins_cost,
                dp[i - 1][j - 1] + sub_cost,
            )
    return float(dp[m][n])


def _circular_distance(seq1: List[str], seq2: List[str], consider_tension: bool = True) -> float:
    if len(seq1) == 0 or len(seq2) == 0:
        return _musical_levenshtein_distance(seq1, seq2, consider_tension=consider_tension)

    min_dist = float("inf")
    for i in range(len(seq1)):
        rotated = seq1[i:] + seq1[:i]
        min_dist = min(min_dist, _musical_levenshtein_distance(rotated, seq2, consider_tension=consider_tension))
    for i in range(len(seq2)):
        rotated = seq2[i:] + seq2[:i]
        min_dist = min(min_dist, _musical_levenshtein_distance(seq1, rotated, consider_tension=consider_tension))
    return float(min_dist)


# Reference progressions in chord tokens (normalized-chord style).
# Notes:
# - Our chord normalization collapses maj7 -> major triad, but keeps dominant "7".
# - So for marusa, "IVM7" becomes "IV" (triad) in this representation.
REF_CHORDS = {
    "odo": ["F", "G", "Em", "Am"],         # IV - V - iii - vi in C
    "komuro": ["Am", "F", "G", "C"],       # vi - IV - V - I in C
    "marusa": ["F", "E7", "Am", "C7"],     # IVM7 - III7 - vi7 - I7 in C (M7 collapsed)
}


def _to_roman_progression(chords: List[str], key_label: str) -> List[str]:
    out: List[str] = []
    for ch in chords:
        r = chord_to_roman(ch, key_label)
        if r:
            out.append(r)
    return out


def compute_typical_chord_distance(
    normalized_chord_progression: List[str],
    normalized_key_label: str,
) -> Optional[Dict[str, float]]:
    """
    Args:
      normalized_chord_progression: chord tokens transposed to C (major) or Am (minor)
      normalized_key_label: "C" or "Am" used to interpret chords as roman numerals
    """
    if not normalized_chord_progression or not normalized_key_label:
        return None

    seq = _to_roman_progression(normalized_chord_progression, normalized_key_label)
    if not seq:
        return None

    out: Dict[str, float] = {}
    for name, ref_chords in REF_CHORDS.items():
        ref_seq = _to_roman_progression(ref_chords, normalized_key_label)
        if not ref_seq:
            continue
        d = _circular_distance(seq, ref_seq, consider_tension=True)
        out[name] = round(float(d), 3)

    return out if out else None


