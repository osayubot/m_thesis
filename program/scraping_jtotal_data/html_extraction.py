#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HTML抽出関連の関数
"""

import re
from typing import Optional
from bs4 import BeautifulSoup


def extract_key_info_from_page(page) -> Optional[dict]:
    """ページからOriginal Key、Capo、PlayKeyを抽出（DOMから直接取得）"""
    try:
        # h3タグのテキストを取得
        h3_text = page.evaluate("""
            () => {
                const h3 = document.querySelector('h3');
                return h3 ? h3.textContent : null;
            }
        """)
        
        if h3_text:
            # Original Keyを抽出（マイナーキーも含む: Am, Dmなど）
            original_key_match = re.search(r'Original\s+Key[：:]\s*([A-G][#♭b]?m?)', h3_text, re.IGNORECASE)
            original_key = original_key_match.group(1).strip() if original_key_match else None
            
            # Capoを抽出（負の値も含む）
            capo_match = re.search(r'Capo[：:]\s*([+-]?\d+)', h3_text, re.IGNORECASE)
            capo = int(capo_match.group(1)) if capo_match else 0
            
            # PlayKeyを抽出（マイナーキーも含む: Am, Dmなど）
            original_play_key_match = re.search(r'Play[：:]\s*([A-G][#♭b]?m?)', h3_text, re.IGNORECASE)
            original_play_key = original_play_key_match.group(1).strip() if original_play_key_match else None
            
            return {
                'original_key': original_key,
                'capo': capo,
                'original_play_key': original_play_key,
                'h3_text': h3_text
            }
    except Exception as e:
        print(f"キー情報の抽出に失敗: {e}")
    
    return None



def extract_lyrics(html_content: str) -> list:
    """HTMLから歌詞を抽出（<a>タグがない行を配列として返す）"""
    soup = BeautifulSoup(html_content, 'lxml')
    lyrics_lines = []
    
    # <tt>タグを探す
    tt_tag = soup.find('tt')
    if not tt_tag:
        return []
    
    # HTMLを文字列として取得し、<br>で分割
    html_str = str(tt_tag)
    # <!--HPSTART-->と<!--HPEND-->の間を取得
    start_marker = '<!--HPSTART-->'
    end_marker = '<!--HPEND-->'
    
    start_idx = html_str.find(start_marker)
    end_idx = html_str.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        content = html_str[start_idx + len(start_marker):end_idx]
    else:
        # マーカーがない場合は全体を使用
        content = html_str
    
    # <br>タグで分割して各行を処理
    lines = re.split(r'<br\s*/?>', content, flags=re.IGNORECASE)
    
    for line in lines:
        if not line.strip():
            continue
        
        # 行をパース
        line_soup = BeautifulSoup(line, 'lxml')
        
        # <a>タグがない行は歌詞
        if not line_soup.find_all('a'):
            lyric = line_soup.get_text(separator='', strip=True)
            # 全角スペースを削除
            lyric = lyric.replace('　', ' ')
            if lyric:
                lyrics_lines.append(lyric)
    
    # 配列として返す
    return lyrics_lines


def clean_html(html_content: str) -> str:
    """HTMLを整形して、不要なhref属性を削除し、全角スペースを削除し、<br>タグの前後で改行"""
    soup = BeautifulSoup(html_content, 'lxml')
    
    # <tt>タグを探す
    tt_tag = soup.find('tt')
    if not tt_tag:
        return html_content
    
    # https://music.j-total.netを含む<a>タグを削除
    for a_tag in tt_tag.find_all('a'):
        href = a_tag.get('href', '')
        if href and 'music.j-total.net' in href:
            # タグ全体を削除
            a_tag.decompose()
        elif href and 'javascript:' in href.lower():
            # JavaScriptで始まるhref属性を削除
            del a_tag['href']
            # target属性も削除（もしあれば）
            if 'target' in a_tag.attrs:
                del a_tag['target']
    
    # コメント（<!--HPSTART-->など）を削除
    from bs4 import Comment
    for comment in tt_tag.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    
    # 全角スペースを削除（NavigableStringのみ）
    for element in tt_tag.descendants:
        if element.__class__.__name__ == 'NavigableString':
            element.replace_with(element.replace('　', ''))
    
    # HTMLを文字列に変換（prettify()は使わない）
    cleaned_html = str(tt_tag)
    
    # <tt>タグの後に改行を追加
    cleaned_html = re.sub(r'(<tt>)', r'\1\n', cleaned_html)
    
    # <br>や<br/>の前後で改行を入れる
    # <br>や<br/>の前に改行がない場合は追加
    cleaned_html = re.sub(r'(?<![\n\r])(<br\s*/?>)', r'\n\1', cleaned_html)
    # <br>や<br/>の後に改行がない場合は追加
    cleaned_html = re.sub(r'(<br\s*/?>)(?![\n\r])', r'\1\n', cleaned_html)
    
    # <br/><br/>に挟まれている部分（コード行）を1行にまとめる
    lines = cleaned_html.split('\n')
    result_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # <br/>で始まる行を探す
        if re.match(r'^\s*<br\s*/?>\s*$', line):
            result_lines.append(line)
            i += 1
            # 次の<br/>まで、<a>タグを含む行を1行にまとめる
            code_lines = []
            while i < len(lines):
                next_line = lines[i]
                # 次の<br/>に到達したら終了
                if re.match(r'^\s*<br\s*/?>\s*$', next_line):
                    break
                # <a>タグを含む行、または空白や記号のみの行（コード行の続き）を収集
                if '<a>' in next_line or '</a>' in next_line:
                    # 改行と余分な空白を削除して結合
                    code_lines.append(next_line.strip())
                elif next_line.strip() and not any(c.isalnum() or ord(c) > 127 for c in next_line.strip()):
                    # 空白や記号のみの行（コード行の続き、例: "/"）も含める
                    code_lines.append(next_line.strip())
                else:
                    # <a>タグがない行（歌詞など）はそのまま追加
                    if code_lines:
                        # コード行を1行にまとめて追加
                        result_lines.append('  ' + ' '.join(code_lines))
                        code_lines = []
                    result_lines.append(next_line)
                i += 1
            # 残っているコード行を1行にまとめて追加
            if code_lines:
                result_lines.append('  ' + ' '.join(code_lines))
        else:
            result_lines.append(line)
            i += 1
    
    cleaned_html = '\n'.join(result_lines)
    
    return cleaned_html


def extract_jtotal_chord_progressions_and_lyrics(html_content: str) -> list:
    """HTMLからコード進行と歌詞を抽出（歌詞ベース、直前の<br>に囲まれたコード進行とペアにする）"""
    soup = BeautifulSoup(html_content, 'lxml')
    result = []
    
    # <tt>タグを探す
    tt_tag = soup.find('tt')
    if not tt_tag:
        return result
    
    # HTMLを文字列として取得し、<br>で分割
    html_str = str(tt_tag)
    # <!--HPSTART-->と<!--HPEND-->の間を取得
    start_marker = '<!--HPSTART-->'
    end_marker = '<!--HPEND-->'
    
    start_idx = html_str.find(start_marker)
    end_idx = html_str.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        content = html_str[start_idx + len(start_marker):end_idx]
    else:
        # マーカーがない場合は全体を使用
        content = html_str
    
    # <br>タグで分割して各行を処理
    # <br>、<br/>、<br />のすべてのパターンに対応
    lines = re.split(r'<br\s*/?>', content, flags=re.IGNORECASE)
    
    # 歌詞ベースで抽出
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        
        # 行をパース
        line_soup = BeautifulSoup(line, 'lxml')
        
        # <a>タグがあるかチェック
        has_a_tags = bool(line_soup.find_all('a'))
        
        # 歌詞を抽出（<a>タグがない行は歌詞）
        if not has_a_tags:
            # すべての<a>タグを削除してからテキストを取得
            for a_tag in line_soup.find_all('a'):
                a_tag.decompose()
            
            lyric = line_soup.get_text(separator='', strip=True)
            # 全角スペースを削除（日本語のフォーマット問題対応）
            lyric = lyric.replace('　', '')
            lyric = remove_spaces_between_japanese(lyric)
            # 繰り返し記号を削除（※印くりかえし、△印くりかえし、◇印くりかえしなど、「…」が含まれる場合も対応）
            lyric = re.sub(r'[（(][※△◇]印くりかえし.*?[）)]', '', lyric)
            # 先頭・末尾の「※」「△」「◇」を削除
            lyric = re.sub(r'^[※△◇]+', '', lyric)
            lyric = re.sub(r'[※△◇]+$', '', lyric)
            # 半角スペースと「/」の削除は削除（英語歌詞を壊すため）
            # 「/」のみの行は空行として扱う
            if lyric.strip() == '/':
                lyric = ''
            
            # (N.C.)が含まれている行は除外（歌詞ではない）
            if '(N.C.)' in lyric or '(N.C' in lyric:
                lyric = ''
            
            if lyric:
                # この歌詞の直前の<br>に囲まれたコード進行を探す
                chords = []
                j = i - 1
                # 直前から遡ってコード行を探す（空行または歌詞行に到達するまで）
                # 複数行のコードを前から順に取得するため、一旦リストに格納
                temp_chords = []
                while j >= 0:
                    prev_line = lines[j]
                    if not prev_line.strip():
                        # 空行（<br/>の後）に到達したら終了
                        break
                    
                    prev_line_soup = BeautifulSoup(prev_line, 'lxml')
                    prev_has_a_tags = bool(prev_line_soup.find_all('a'))
                    
                    # コード行の場合（<a>タグがある）
                    if prev_has_a_tags:
                        # コードを抽出（行内の順序を保持）
                        line_chords = []
                        for a_tag in prev_line_soup.find_all('a'):
                            chord_text = a_tag.get_text(strip=True)
                            # リンクテキストがコードっぽいかチェック（「動画」などのリンクを除外）
                            if chord_text and not any(word in chord_text for word in ['動画', '初心者', 'こちら', 'Ver']):
                                line_chords.append(chord_text)
                        if line_chords:
                            temp_chords.insert(0, line_chords)  # 行を前から順に追加
                        j -= 1
                    else:
                        # 歌詞行に到達したら終了
                        break
                
                # 前から順に結合
                for line_chords in temp_chords:
                    chords.extend(line_chords)
                
                # コード進行と歌詞をペアにして追加
                result.append({
                    'chord_progression': chords,
                    'lyric': lyric
                })
        
        i += 1
    
    return result

import re

def remove_spaces_between_japanese(text: str) -> str:
    """
    半角スペースについて：
    - 前後が日本語（ひらがな・カタカナ・漢字）の場合のみ削除
    - それ以外（英語を含む）はスペースを残す
    """
    # 日本語（漢字・ひらがな・カタカナ）
    jp = r"[一-龥ぁ-んァ-ン]"

    # パターン：
    #   日本語 + 半角スペース + 日本語
    pattern = re.compile(f"({jp}) ({jp})")

    # スペースを削除
    while True:
        new_text = pattern.sub(r"\1\2", text)
        if new_text == text:
            break
        text = new_text

    return text
