#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
スクレイピング関数
"""

import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from .utils import should_exclude_link, extract_path_from_url


def scrape_paths_from_search_page(url: str) -> list:
    """
    検索結果ページからパス一覧を取得（タイトル、パス、アーティストのペア）
    
    Args:
        url: 検索結果ページのURL
    
    Returns:
        タイトル、パス、アーティストのリスト
    """
    items = []  # {title, path, artist}のリスト
    path_to_title = {}  # パスをキーとして情報を保持（重複排除用）
    
    try:
        # ページを取得
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # エンコーディングを設定（J-Total MusicはShift_JIS）
        # HTMLのmetaタグから確認
        if 'charset=Shift_JIS' in response.text[:2000] or 'charset=shift_jis' in response.text[:2000].lower():
            response.encoding = 'shift_jis'
        elif response.encoding is None or response.encoding.lower() in ['iso-2022-jp']:
            # コンテンツからエンコーディングを推測
            try:
                import chardet
                detected = chardet.detect(response.content)
                if detected and detected.get('encoding'):
                    response.encoding = detected['encoding']
                else:
                    response.encoding = 'shift_jis'
            except ImportError:
                response.encoding = 'shift_jis'
        
        # HTMLをパース（エンコーディングを明示的に指定）
        # BeautifulSoup 4.4.0以降ではfrom_encodingは非推奨だが、明示的に指定する
        try:
            soup = BeautifulSoup(response.content, 'lxml', from_encoding=response.encoding)
        except TypeError:
            # from_encodingがサポートされていない場合は、textを使用
            soup = BeautifulSoup(response.text, 'lxml')
        
        # width="600"のテーブル要素を抜き出す
        result_tables = soup.find_all('table', attrs={'width': '600'})
        
        # 各テーブルを処理
        for table in result_tables:
            # このテーブル内の<a>タグで、https://music.j-total.net/data/で始まるリンクを探す
            links = table.find_all('a', href=True)
            data_link = None
            path = None
            
            for link in links:
                link_href = link.get('href', '')
                # 完全なURLに変換
                full_url = urljoin(url, link_href)
                
                # https://music.j-total.net/data/で始まるリンクを探す
                if full_url.startswith('https://music.j-total.net/data/'):
                    # それ以降の文字をpathとして保存
                    path = full_url.replace('https://music.j-total.net/data/', '')
                    # .html拡張子を削除
                    if path.endswith('.html'):
                        path = path[:-5]
                    
                    # 除外条件をチェック
                    link_text = link.get_text(strip=True)
                    if not should_exclude_link(link_text, link_href, path):
                        data_link = link
                        break
            
            # /data/で始まるリンクが見つからない場合は、このテーブルをスキップ
            if not data_link or not path:
                continue
            
            # <b>タグの中身をtitleとして保存
            title = ''
            bold_tag = table.find('b')
            if bold_tag:
                title = bold_tag.get_text(strip=True)
            
            # <td>タグの中で「歌：」が含まれていれば、その「歌：」の後の文字をartistとして保存
            # 「歌：」が含まれていないテーブルは無視（スキップ）
            artist = ''
            td_tags = table.find_all('td')
            for td in td_tags:
                td_text = td.get_text(strip=True)
                if '歌：' in td_text:
                    # 「歌：」の後の文字を抽出
                    artist_match = re.search(r'歌[：:]([^/]+)', td_text)
                    if artist_match:
                        artist = artist_match.group(1).strip()
                        break  # 最初に見つかったものを使用
            
            # 「歌：」が含まれていない（artistが空）場合は、このテーブルをスキップ
            if not artist:
                continue
            
            # タイトルから「〜XXXX〜」形式の文字列を削除（最後に付いているもの）
            if title:
                # 「〜」で始まり「〜」で終わる文字列を削除（最後に付いているもの）
                title = re.sub(r'〜[^〜]*〜$', '', title).strip()
                # 末尾の「〜」だけが残っている場合も削除
                title = title.rstrip('〜').strip()
            
            # パスが既に存在する場合は、情報が空でない場合のみ更新
            if path not in path_to_title:
                path_to_title[path] = {
                    'title': title,
                    'artist': artist
                }
            else:
                # 既存の情報を更新（空でない場合のみ）
                existing = path_to_title[path]
                if isinstance(existing, dict):
                    if title and not existing.get('title'):
                        existing['title'] = title
                    if artist and not existing.get('artist'):
                        existing['artist'] = artist
                else:
                    # 既存が文字列の場合は辞書に変換
                    path_to_title[path] = {
                        'title': existing if existing else title,
                        'artist': artist
                    }
        
        # タイトルとパスのペアリストを作成
        items = []
        for path in sorted(path_to_title.keys()):
            info = path_to_title.get(path, {})
            if isinstance(info, dict):
                items.append({
                    'title': info.get('title', ''),
                    'artist': info.get('artist', ''),
                    'path': path
                })
            else:
                # 後方互換性（既存の文字列形式の場合）
                items.append({
                    'title': info if info else '',
                    'artist': '',
                    'path': path,
                })
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        raise
    
    return items

