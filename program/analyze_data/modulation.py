"""
転調検出関数
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix, hstack
from .data_extraction import root_hist_12
from .data_extraction import extract_ufret_chords, extract_jtotal_chords

def make_window_features(norm_chords, vec_all):
    """Create feature vector for a window of chords."""
    text = " ".join(norm_chords)
    X_bag = vec_all.transform([text])
    X_root = csr_matrix(root_hist_12(norm_chords).reshape(1, -1))
    return hstack([X_bag, X_root]).tocsr()

def sliding_window_probs(chords_norm, vec_all, clf, W=16, H=4):
    """
    Sliding window analysis to get key probabilities for each window.
    
    Returns:
      probs: (T, K) where T = number of windows, K = num classes
      spans: list[(start_idx, end_idx)] chord indices
    """
    spans = []
    probs = []
    n = len(chords_norm)
    
    if n < W:
        Xw = make_window_features(chords_norm, vec_all)
        p = clf.predict_proba(Xw)[0]
        return np.vstack([p]), [(0, n)]
    
    for s in range(0, n - W + 1, H):
        e = s + W
        win = chords_norm[s:e]
        Xw = make_window_features(win, vec_all)
        p = clf.predict_proba(Xw)[0]
        probs.append(p)
        spans.append((s, e))
    
    return np.vstack(probs), spans

def viterbi_hmm(log_emission, switch_penalty=4.0):
    """
    Viterbi algorithm for HMM smoothing.
    
    log_emission: (T, K)  各窓のlog P(key | window)
    遷移は「同じキー=コスト0」「別キー=コストswitch_penalty」
    
    Returns: best_path (T,) state indices
    """
    T, K = log_emission.shape
    dp = np.full((T, K), -np.inf, dtype=np.float64)
    back = np.zeros((T, K), dtype=np.int32)
    
    dp[0] = log_emission[0]
    back[0] = -1
    
    for t in range(1, T):
        prev = dp[t-1]
        best_prev_any = np.max(prev)
        for k in range(K):
            stay = prev[k]
            switch = best_prev_any - switch_penalty
            if stay >= switch:
                dp[t, k] = stay + log_emission[t, k]
                back[t, k] = k
            else:
                dp[t, k] = switch + log_emission[t, k]
                back[t, k] = int(np.argmax(prev))
    
    path = np.zeros(T, dtype=np.int32)
    path[T-1] = int(np.argmax(dp[T-1]))
    for t in range(T-1, 0, -1):
        path[t-1] = back[t, path[t]]
    
    return path

def detect_modulations(state_path, spans, classes, min_run_windows=3):
    """
    Extract modulation points from state path.
    
    state_path: (T,) Viterbi後のキーindex列
    spans: windowごとの(chord start,end)
    
    Returns: list of dicts with modulation points
    """
    T = len(state_path)
    runs = []
    i = 0
    while i < T:
        j = i
        while j < T and state_path[j] == state_path[i]:
            j += 1
        runs.append((i, j, state_path[i]))
        i = j
    
    mods = []
    filtered = [r for r in runs if (r[1]-r[0]) >= min_run_windows]
    for a, b in zip(filtered, filtered[1:]):
        end_win = a[1]-1
        next_win = b[0]
        chord_pos = spans[next_win][0]
        mods.append({
            "from_key": classes[a[2]],
            "to_key": classes[b[2]],
            "at_chord_index": chord_pos,
            "from_windows": (a[0], a[1]),
            "to_windows": (b[0], b[1]),
        })
    return mods

def build_chord_index_map(song_json, use_ufret=True):
    """
    Build mapping from flattened chord index to section/chord position.
    
    Args:
        song_json: Song data dictionary
        use_ufret: If True, use ufret_chord_progressions_and_lyrics, else jtotal
    
    Returns: list of (section_idx, chord_idx_in_section, chord_str, lyric)
    """
    mapping = []
    field_name = "ufret_chord_progressions_and_lyrics" if use_ufret else "jtotal_chord_progressions_and_lyrics"
    secs = song_json.get(field_name, [])
    for si, sec in enumerate(secs):
        lyric = sec.get("lyric", "")
        for ci, chord in enumerate(sec.get("chord_progression", [])):
            mapping.append((si, ci, chord, lyric))
    return mapping

def modulation_analysis_for_song(song_json, vec_all, clf, W=16, H=4, switch_penalty=4.0, min_run_windows=3, use_ufret=True):
    """
    Analyze modulation (key changes) in a song.
    
    Args:
        song_json: Song data dictionary
        vec_all: Fitted TfidfVectorizer
        clf: Trained LogisticRegression classifier
        W: Window size (number of chords)
        H: Step size for sliding window
        switch_penalty: HMM transition penalty for key changes
        min_run_windows: Minimum consecutive windows to consider a key change
        use_ufret: If True, use ufret_chord_progressions_and_lyrics, else jtotal
    
    Returns:
        Dictionary with analysis results including modulations
    """
    if use_ufret:
        chords_norm = extract_ufret_chords(song_json)
    else:
        chords_norm = extract_jtotal_chords(song_json)
    if len(chords_norm) < 12:
        return {"error": "too_few_chords"}
    
    probs, spans = sliding_window_probs(chords_norm, vec_all, clf, W=W, H=H)
    
    eps = 1e-12
    log_em = np.log(np.clip(probs, eps, 1.0))
    path = viterbi_hmm(log_em, switch_penalty=switch_penalty)
    
    classes = clf.classes_
    mods = detect_modulations(path, spans, classes, min_run_windows=min_run_windows)
    
    keys_per_window = [classes[i] for i in path]
    return {
        "W": W, "H": H,
        "switch_penalty": switch_penalty,
        "min_run_windows": min_run_windows,
        "keys_per_window": keys_per_window,
        "spans": spans,
        "modulations": mods,
    }

