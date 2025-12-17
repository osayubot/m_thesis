#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ユーティリティ関数
"""

from urllib.parse import urlparse


def should_exclude_link(link_text: str, link_href: str, path: str = '') -> bool:
    """
    リンクを除外するかどうかを判定
    
    Args:
        link_text: リンクのテキスト
        link_href: リンクのhref属性
        path: 抽出されたパス
    
    Returns:
        除外する場合True
    """
    # 除外条件をチェック
    exclude_keywords = [
        '童謡・わらべうた',
        '動画視聴Ver.',
        '動画sync Ver.',
        '初心者向け簡単コードVer.',
        '弾き語り向け詳細コードVer.',
        'セルフカバーVer.'
    ]
    
    # リンクテキストに除外キーワードが含まれているかチェック
    for keyword in exclude_keywords:
        if keyword in link_text:
            return True
    
    # hrefに除外キーワードが含まれているかチェック
    for keyword in exclude_keywords:
        if keyword in link_href:
            return True
    
    # パス名に-mvや-eが含まれている場合は除外（動画sync Ver.や初心者向け簡単コードVer.）
    if path:
        if path.endswith('-mv') or path.endswith('-e'):
            return True
        # パス名に-mvや-eが含まれている場合も除外
        if '/-mv' in path or '/-e' in path or path.startswith('-mv') or path.startswith('-e'):
            return True
    
    return False


def extract_path_from_url(url: str) -> str:
    """
    URLからパスを抽出
    例: https://music.j-total.net/data/001a/118_aimyong/015.html
    -> 001a/118_aimyong/015
    
    Args:
        url: J-Total MusicのURL
    
    Returns:
        パス文字列（拡張子なし）
    """
    # URLをパース
    parsed = urlparse(url)
    path = parsed.path
    
    # /data/ 以降のパスを取得
    if '/data/' in path:
        path_part = path.split('/data/')[1]
        # 拡張子を削除
        if path_part.endswith('.html'):
            path_part = path_part[:-5]
        return path_part
    
    return ''


def detect_path_prefix(paths: list) -> str:
    """
    パスリストから共通のプレフィックスを検出（最も多く出現するものを選択）
    
    Args:
        paths: パスのリスト
    
    Returns:
        共通プレフィックス（例: '001a/118_aimyong' または '032mi/009_Mr_Children'）
    """
    if not paths:
        return ''
    
    # 各パスのプレフィックス（最初の2セグメント）をカウント
    prefix_count = {}
    for path in paths:
        parts = path.split('/')
        if len(parts) >= 2:
            prefix = '/'.join(parts[:2])
            prefix_count[prefix] = prefix_count.get(prefix, 0) + 1
    
    # 最も多く出現するプレフィックスを返す
    if prefix_count:
        return max(prefix_count.items(), key=lambda x: x[1])[0]
    
    return ''

