"""
キー割り当て関数
"""
from __future__ import annotations
from collections import Counter
import numpy as np
from .data_extraction import (
    extract_jtotal_chords, extract_jtotal_chords_with_section_spans,
    extract_ufret_chords, extract_ufret_chords_with_section_spans
)
from .transposition import estimate_transposition_shift, transpose_key, lyric_initial_match
from .modulation import sliding_window_probs, viterbi_hmm, build_chord_index_map
from .emotion_analysis import analyze_emotion

def assign_keys_to_ufret_with_transposition(song_json, vec_all, clf, W=16, H=4, switch_penalty=4.0):
    """
    Assign keys to ufret sections using transposition-corrected jtotal key labels.
    """
    jt_chords = extract_jtotal_chords(song_json)
    uf_chords = extract_ufret_chords(song_json)
    
    if len(jt_chords) < 12 or len(uf_chords) < 12:
        return []
    
    jt_key = song_json.get("jtotal_original_play_key") or song_json.get("original_play_key")
    if not jt_key:
        return []
    
    shift = estimate_transposition_shift(jt_chords, uf_chords, N=16)
    if shift is None:
        shift = 0
    
    ufret_key = transpose_key(jt_key, shift)
    if not ufret_key:
        ufret_key = jt_key
    
    probs, spans = sliding_window_probs(uf_chords, vec_all, clf, W=W, H=H)
    classes = clf.classes_
    
    eps = 1e-12
    log_em = np.log(np.clip(probs, eps, 1.0))
    path = viterbi_hmm(log_em, switch_penalty=switch_penalty)
    
    ufret_secs = song_json.get("ufret_chord_progressions_and_lyrics", [])
    jtotal_secs = song_json.get("jtotal_chord_progressions_and_lyrics", [])
    
    if not ufret_secs:
        return []
    
    mapping = []
    for si, sec in enumerate(ufret_secs):
        for ci, chord in enumerate(sec.get("chord_progression", [])):
            mapping.append((si, ci, chord))
    
    n_chords = len(mapping)
    chord_key_votes = [[] for _ in range(n_chords)]
    
    for widx, (s, e) in enumerate(spans):
        k = classes[path[widx]]
        for i in range(s, min(e, n_chords)):
            chord_key_votes[i].append(k)
    
    analyzed_sections = []
    flat_idx = 0
    
    for sec_idx, uf_sec in enumerate(ufret_secs):
        chord_prog = uf_sec.get("chord_progression", [])
        section_start_idx = flat_idx
        section_end_idx = flat_idx + len(chord_prog)
        
        sec_votes = []
        for i in range(section_start_idx, min(section_end_idx, n_chords)):
            if chord_key_votes[i]:
                sec_votes.append(Counter(chord_key_votes[i]).most_common(1)[0][0])
        
        if sec_votes:
            c = Counter(sec_votes)
            section_key, count = c.most_common(1)[0]
            conf = count / len(sec_votes)
        else:
            section_key = ufret_key
            conf = 0.5
        
        lyric_match = False
        if sec_idx < len(jtotal_secs):
            jt_sec = jtotal_secs[sec_idx]
            lyric_match = lyric_initial_match(jt_sec, uf_sec)
        
        analyzed_sec = {
            "chord_progression": chord_prog.copy(),
            "lyric": uf_sec.get("lyric", "")
        }
        
        if "chord_word_pair" in uf_sec:
            analyzed_sec["chord_word_pair"] = uf_sec["chord_word_pair"].copy()
        
        analyzed_sec["key"] = section_key
        analyzed_sec["key_confidence"] = round(float(conf), 3)
        analyzed_sec["lyric_match_quality"] = lyric_match
        
        # Add emotion analysis
        lyric = uf_sec.get("lyric", "")
        emotion_scores = analyze_emotion(lyric)
        if emotion_scores:
            analyzed_sec["emotion"] = emotion_scores
        
        analyzed_sections.append(analyzed_sec)
        flat_idx = section_end_idx
    
    return analyzed_sections

def assign_keys_with_probabilities(song_json, vec_all, clf, W=16, H=4, switch_penalty=4.0, use_ufret=True):
    """
    Assign keys to sections using probability averaging.
    """
    if use_ufret:
        chords_norm = extract_ufret_chords(song_json)
        field_name = "ufret_chord_progressions_and_lyrics"
    else:
        chords_norm = extract_jtotal_chords(song_json)
        field_name = "jtotal_chord_progressions_and_lyrics"
    
    if len(chords_norm) < 12:
        return []
    
    probs, spans = sliding_window_probs(chords_norm, vec_all, clf, W=W, H=H)
    classes = clf.classes_
    
    eps = 1e-12
    log_em = np.log(np.clip(probs, eps, 1.0))
    path = viterbi_hmm(log_em, switch_penalty=switch_penalty)
    
    secs = song_json.get(field_name, [])
    if not secs:
        return []
    
    mapping = build_chord_index_map(song_json, use_ufret=use_ufret)
    n_chords = len(mapping)
    
    analyzed_sections = []
    
    flat_idx = 0
    for sec_idx, sec in enumerate(secs):
        chord_prog = sec.get("chord_progression", [])
        section_start_idx = flat_idx
        section_end_idx = flat_idx + len(chord_prog)
        
        analyzed_sec = {
            "chord_progression": chord_prog.copy(),
            "lyric": sec.get("lyric", "")
        }
        
        if "chord_word_pair" in sec:
            analyzed_sec["chord_word_pair"] = sec["chord_word_pair"].copy()
        
        section_probs = np.zeros(len(classes), dtype=np.float32)
        total_weight = 0.0
        
        for win_idx, (win_start, win_end) in enumerate(spans):
            overlap_start = max(section_start_idx, win_start)
            overlap_end = min(section_end_idx, win_end)
            overlap_size = max(0, overlap_end - overlap_start)
            
            if overlap_size > 0:
                weight = overlap_size / (section_end_idx - section_start_idx)
                section_probs += probs[win_idx] * weight
                total_weight += weight
        
        if total_weight > 0:
            section_probs /= total_weight
            best_key_idx = int(np.argmax(section_probs))
            section_key = classes[best_key_idx]
            confidence = float(section_probs[best_key_idx])
        else:
            section_key = classes[path[0]] if len(path) > 0 else None
            confidence = 0.5
        
        analyzed_sec["key"] = section_key
        analyzed_sec["key_confidence"] = round(confidence, 3)
        analyzed_sec["key_method"] = "window_hmm_probavg_v1"
        analyzed_sec["key_span"] = {
            "unit": "chord",
            "start": section_start_idx,
            "end": section_end_idx
        }
        
        # Add emotion analysis
        lyric = sec.get("lyric", "")
        emotion_scores = analyze_emotion(lyric)
        if emotion_scores:
            analyzed_sec["emotion"] = emotion_scores
        
        analyzed_sections.append(analyzed_sec)
        flat_idx = section_end_idx
    
    return analyzed_sections

def assign_keys_to_jtotal_sections(song_json, vec_all, clf, W=16, H=4, switch_penalty=4.0):
    """
    Assign keys to jtotal sections using normalized chord indices.
    """
    chords_norm, section_spans, _ = extract_jtotal_chords_with_section_spans(song_json)
    if len(chords_norm) < 12:
        return []
    
    probs, spans = sliding_window_probs(chords_norm, vec_all, clf, W=W, H=H)
    classes = clf.classes_
    
    eps = 1e-12
    log_em = np.log(np.clip(probs, eps, 1.0))
    path = viterbi_hmm(log_em, switch_penalty=switch_penalty)
    
    n = len(chords_norm)
    chord_key_votes = [[] for _ in range(n)]
    
    for widx, (s, e) in enumerate(spans):
        k = classes[path[widx]]
        for i in range(s, min(e, n)):
            chord_key_votes[i].append(k)
    
    analyzed_sections = []
    secs = song_json.get("jtotal_chord_progressions_and_lyrics", [])
    
    for sec_idx, sec in enumerate(secs):
        start, end = section_spans[sec_idx]
        
        if end <= start:
            section_key = None
            conf = 0.0
        else:
            sec_votes = []
            for i in range(start, end):
                if chord_key_votes[i]:
                    sec_votes.append(Counter(chord_key_votes[i]).most_common(1)[0][0])
            
            if sec_votes:
                c = Counter(sec_votes)
                section_key, count = c.most_common(1)[0]
                conf = count / len(sec_votes)
            else:
                section_key = None
                conf = 0.0
        
        analyzed_sec = {
            "chord_progression": sec.get("chord_progression", []).copy(),
            "lyric": sec.get("lyric", "")
        }
        
        if "chord_word_pair" in sec:
            analyzed_sec["chord_word_pair"] = sec["chord_word_pair"].copy()
        
        analyzed_sec["key"] = section_key
        analyzed_sec["key_confidence"] = round(float(conf), 3)
        
        # Add emotion analysis
        lyric = sec.get("lyric", "")
        emotion_scores = analyze_emotion(lyric)
        if emotion_scores:
            analyzed_sec["emotion"] = emotion_scores
        
        analyzed_sections.append(analyzed_sec)
    
    return analyzed_sections

def assign_keys_to_ufret_sections(song_json, vec_all, clf, W=16, H=4, switch_penalty=4.0):
    """
    Assign keys to ufret sections.

    Note:
    - The original implementation used Viterbi-smoothed hard labels per window and
      section-level majority vote. With a high `switch_penalty`, that tends to
      "stick" to a single key for long stretches and can overstate confidence.
    - We now default to probability-averaging (same as `assign_keys_with_probabilities`)
      which better reflects ambiguity (e.g., relative major/minor like G vs Em).
    """
    analyzed_sections = assign_keys_with_probabilities(
        song_json, vec_all, clf,
        W=W, H=H, switch_penalty=switch_penalty,
        use_ufret=True
    )
    # Keep explicit method label for downstream/debugging
    for sec in analyzed_sections:
        if isinstance(sec, dict):
            sec.setdefault("key_method", "window_probavg_v1")
    return analyzed_sections

