#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_scrapingパッケージ
"""

from .html_extraction import (
    extract_key_info_from_page,
    extract_lyrics,
    clean_html,
    extract_jtotal_chord_progressions_and_lyrics
)
from .key_utils import get_key_semitones
from .spotify_utils import add_spotify_info, get_spotify_track_info
from .data_loader import load_data_path_files

__all__ = [
    'extract_key_info_from_page',
    'extract_lyrics',
    'clean_html',
    'extract_jtotal_chord_progressions_and_lyrics',
    'get_key_semitones',
    'add_spotify_info',
    'get_spotify_track_info',
    'load_data_path_files',
]

