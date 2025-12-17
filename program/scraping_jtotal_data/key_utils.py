#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
キー関連のユーティリティ関数
"""


def get_key_semitones(key: str) -> int:
    """キーを半音数に変換（C=0）"""
    key_map = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
        'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }
    
    # フラットをbに統一
    key = key.replace('♭', 'b')
    
    return key_map.get(key, 0)

