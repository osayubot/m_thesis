#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
U-FRETのHTMLからデータを抽出する関数
"""

from bs4 import BeautifulSoup


def extract_ufret_chord_progressions_and_lyrics(html_content):
    """Extract chord progression and lyrics from HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')
    rows = soup.find_all('div', class_='chord-row')
    
    result = []
    for row in rows:
        chords = []
        lyrics = []
        chord_word_pairs = []
        
        for p in row.find_all('p', class_='chord'):
            chord = p.find('rt')
            chord_text = chord.text if chord else ""
            
            lyric_spans = p.find_all('span', class_='col')
            lyric_text = ''.join([span.text for span in lyric_spans if span.text.strip()])
            
            if chord_text:
                chords.append(chord_text)
            
            if lyric_text:
                lyrics.append(lyric_text)
            
            # chord_and_word用のペアを追加
            if chord_text or lyric_text:
                chord_word_pairs.append({
                    "chord": chord_text,
                    "word": lyric_text
                })
        
        if chords or lyrics:
            item = {
                "chord_progression": chords,
                "lyric": ''.join(lyrics)
            }
            
            # chord_word_pairを追加（chordとwordの配列形式）
            # lyricが空の場合は追加しない
            lyric_text = ''.join(lyrics)
            if lyric_text and chord_word_pairs:
                item["chord_word_pair"] = {
                    "chord": [pair["chord"] for pair in chord_word_pairs],
                    "word": [pair["word"] for pair in chord_word_pairs]
                }
            
            result.append(item)
    
    return result


def extract_chord_and_word(html_content):
    """Extract chord and word pairs from HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')
    rows = soup.find_all('div', class_='chord-row')
    
    result = []
    for row in rows:
        chord_lyric_pairs = []
        
        for p in row.find_all('p', class_='chord'):
            chord = p.find('rt')
            chord_text = chord.text if chord else ""
            
            lyric_spans = p.find_all('span', class_='col')
            lyric_text = ''.join([span.text for span in lyric_spans if span.text.strip()])
            
            if chord_text or lyric_text:
                chord_lyric_pairs.append({
                    "chord": chord_text,
                    "word": lyric_text
                })
        
        result.extend(chord_lyric_pairs)
    
    return result


def parse_composer_info(composer_text):
    """Parse composer information"""
    lyricist = ""
    composer = ""
    
    lines = composer_text.split('/')
    
    for line in lines:
        line = line.strip()
        if '作詞' in line:
            lyricist = line.split(':')[-1].strip()
        elif '作曲' in line:
            composer = line.split(':')[-1].strip()
    
    return lyricist, composer

