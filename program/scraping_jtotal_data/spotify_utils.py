#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spotify関連のユーティリティ関数
"""

import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from typing import Optional, Dict, Any, List, Tuple
import sys
import re
import json
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# ============================================================
# Spotify クライアントはグローバルで 1 回だけ生成（重要）
# ============================================================
if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
    client_credentials_manager = SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    )
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
else:
    sp = None


def normalize_string(s: str) -> str:
    """
    文字列を正規化（小文字化、空白除去）
    日本語文字（ひらがな、カタカナ、漢字）も保持します
    """
    if not s:
        return ""
    # 小文字化して、空白と記号を除去（日本語文字は保持）
    # \w は Unicode文字、数字、アンダースコアにマッチ
    # 日本語文字（ひらがな、カタカナ、漢字）も含まれる
    normalized = re.sub(r'[^\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', '', s.lower())
    return normalized


def is_english_string(s: str) -> bool:
    """
    文字列が英語のみかどうかを判定（日本語文字が含まれていないか）
    """
    if not s:
        return False
    # 日本語文字（ひらがな、カタカナ、漢字）が含まれていないかチェック
    return not bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', s))


def match_english_names(name1: str, name2: str) -> bool:
    """
    英語名同士を比較（姓名が逆になっている場合も考慮）
    正規化した後、短い方の文字が長い方に全て含まれていればOK（並び順は問わず）
    """
    if not name1 or not name2:
        return False
    
    norm1 = normalize_string(name1)
    norm2 = normalize_string(name2)
    
    # 完全一致の場合はTrue
    if norm1 == norm2:
        return True
    
    # 英語文字列の場合のみ、部分一致チェック
    if is_english_string(norm1) and is_english_string(norm2):
        # 短い方と長い方を判定
        if len(norm1) <= len(norm2):
            shorter = norm1
            longer = norm2
        else:
            shorter = norm2
            longer = norm1
        
        # 短い方が空の場合はFalse
        if not shorter:
            return False
        
        # 短い方の各文字が長い方に含まれているかチェック（出現回数も考慮）
        from collections import Counter
        shorter_chars = Counter(shorter)
        longer_chars = Counter(longer)
        
        # 短い方の各文字が、長い方に十分な数含まれているかチェック
        for char, count in shorter_chars.items():
            if longer_chars.get(char, 0) < count:
                return False
        
        return True
    
    return False


def extract_artist_en_from_path(jtotal_path: str) -> Optional[str]:
    """
    jtotal_pathから英語のアーティスト名を抽出
    
    例:
        "001a/012_anzenchitai/002" -> "anzenchitai"
        "001a/011_THE_ALFEE/001" -> "THE ALFEE"
    """
    if not jtotal_path:
        return None
    
    # パスを分割
    parts = jtotal_path.split('/')
    if len(parts) < 2:
        return None
    
    # アーティスト名のディレクトリ部分を取得（例: "012_anzenchitai" または "011_THE_ALFEE"）
    artist_dir = parts[1] if len(parts) > 1 else None
    
    if not artist_dir:
        return None
    
    # 数字とアンダースコアのプレフィックスを除去（例: "012_" や "011_"）
    # アンダースコアで分割して、最初の部分が数字の場合はスキップ
    dir_parts = artist_dir.split('_', 1)
    if len(dir_parts) > 1:
        # 数字部分をスキップ（例: "012_anzenchitai" -> "anzenchitai"）
        artist_en = dir_parts[1]
    else:
        # アンダースコアがない場合、数字部分を除去
        artist_en = re.sub(r'^\d+', '', artist_dir)
    
    # アンダースコアをスペースに変換（例: "THE_ALFEE" -> "THE ALFEE"）
    artist_en = artist_en.replace('_', ' ')
    
    # 空文字列の場合はNone
    if not artist_en:
        return None
    
    return artist_en


def get_spotify_track_info(track_name: str, artist_name_ja: str, artist_name_en: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    """
    Spotifyからトラック情報を取得（normalize したうえでアーティスト名が一致するものだけ採用）
    
    ①まず日本語のアーティスト名で検索し、Spotifyで返ってきたアーティスト名（日本語/英語）と
    日本語名または英語名を比較して一致するものを返す。
    ②見つからなかった場合、英語名で検索し、同様に比較する。
    
    Args:
        track_name: 曲名
        artist_name_ja: 日本語のアーティスト名
        artist_name_en: 英語のアーティスト名（jtotal_pathから抽出可能な場合）
    
    Returns:
        (一致するトラック情報, 検索結果の最初の2件) のタプル
        一致するものがある場合: (track_info, None)
        一致しない場合: (None, [track_info1, track_info2])
    """

    # 認証情報なし
    if sp is None:
        print("Warning: Spotify credentials not configured", file=sys.stderr)
        return None, None

    def search_and_match(query_artist_name: str, search_name_type: str) -> Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        """検索とマッチングを行う内部関数"""
        try:
            query = f"track:{track_name} artist:{query_artist_name}"
            print(f"  [検索クエリ] {search_name_type}: {query}", file=sys.stderr)
            results = sp.search(q=query, type='track', limit=20)

            if not results['tracks']['items']:
                print(f"  [DEBUG] 検索結果が空です", file=sys.stderr)
                return None, None

            tracks = results['tracks']['items']
            print(f"  [DEBUG] 検索結果数: {len(tracks)}", file=sys.stderr)
            
            # 検索結果の最初の2件を保存用に整形
            def format_track_info(track: Dict[str, Any]) -> Dict[str, Any]:
                """トラック情報を整形"""
                return {
                    "id": track.get('id'),
                    'artist_id': track.get('artists', [{}])[0].get('id') if track.get('artists') else None,
                    'name': track.get('name'),
                    'artist': track.get('artists', [{}])[0].get('name') if track.get('artists') else None,
                    'artist_en': track.get('artists', [{}])[0].get('name') if track.get('artists') else None,
                    'album': track.get('album', {}).get('name'),
                    'release_date': track.get('album', {}).get('release_date'),
                    'duration_ms': track.get('duration_ms'),
                    'popularity': track.get('popularity'),
                    'preview_url': track.get('preview_url'),
                }
            
            # 最初の2件を保存
            top_results = [format_track_info(track) for track in tracks[:2]]
            
            # 比較対象のアーティスト名を正規化
            normalized_artist_ja = normalize_string(artist_name_ja) if artist_name_ja else ""
            normalized_artist_en = normalize_string(artist_name_en) if artist_name_en else ""
            
            # デバッグ: 検索結果のアーティスト名を確認
            print(f"  [DEBUG] {search_name_type}検索: {track_name} - {query_artist_name}", file=sys.stderr)
            print(f"  [DEBUG] 検索結果数: {len(tracks)}", file=sys.stderr)
            for i, track in enumerate(tracks[:3]):  # 最初の3件だけ表示
                spotify_artist = track.get('artists', [{}])[0].get('name', 'Unknown')
                normalized_spotify = normalize_string(spotify_artist)
                print(f"  [DEBUG] 結果{i+1}: {track.get('name')} - {spotify_artist}", file=sys.stderr)
            
            # トラック名を正規化（比較用）
            normalized_track_name = normalize_string(track_name)
            
            # アーティスト名が一致するトラックを探す
            for track in tracks:
                spotify_track_name = track.get('name', '')
                normalized_spotify_track_name = normalize_string(spotify_track_name)
                
                # トラック名が一致しているかチェック
                track_name_matches = normalized_track_name == normalized_spotify_track_name
                
                spotify_artists = [artist.get('name', '') for artist in track.get('artists', [])]
                
                # 全てのアーティストをチェック（複数アーティストの場合も対応）
                for spotify_artist in spotify_artists:
                    normalized_spotify_artist = normalize_string(spotify_artist)
                    # 正規化していない元の文字列も取得
                    original_spotify_artist = spotify_artist
                    
                    # マッチング条件:
                    # 1. 日本語名または英語名のどちらかが一致すればマッチ
                    # 2. トラック名が一致している場合、アーティスト名が部分一致（短い方が長い方に含まれる）でもマッチ
                    is_match = False
                    
                    if normalized_spotify_artist:
                        # 正規化した日本語名と一致
                        if normalized_artist_ja and normalized_spotify_artist == normalized_artist_ja:
                            is_match = True
                        
                        # 正規化した英語名と一致（完全一致）
                        if not is_match and normalized_artist_en and normalized_spotify_artist == normalized_artist_en:
                            is_match = True
                        
                        # 英語名同士の部分一致チェック（姓名逆順対応）
                        if not is_match and normalized_artist_en:
                            if match_english_names(original_spotify_artist, artist_name_en):
                                is_match = True
                    
                    # 正規化していない元の文字列でもチェック
                    if not is_match:
                        # 元の日本語名と一致（大文字小文字を無視）
                        if artist_name_ja and original_spotify_artist.lower() == artist_name_ja.lower():
                            is_match = True
                        
                        # 元の英語名と一致（大文字小文字を無視）
                        if not is_match and artist_name_en and original_spotify_artist.lower() == artist_name_en.lower():
                            is_match = True
                    
                    # トラック名が一致している場合、アーティスト名の部分一致も許容
                    if not is_match and track_name_matches:
                        # 日本語名の場合：短い方が長い方に含まれているかチェック
                        if normalized_artist_ja and normalized_spotify_artist:
                            shorter_ja = normalized_artist_ja if len(normalized_artist_ja) <= len(normalized_spotify_artist) else normalized_spotify_artist
                            longer_ja = normalized_artist_ja if len(normalized_artist_ja) > len(normalized_spotify_artist) else normalized_spotify_artist
                            if shorter_ja and longer_ja and shorter_ja in longer_ja:
                                is_match = True
                        
                        # 英語名の場合：match_english_namesを使用（姓名逆順対応）
                        if not is_match and artist_name_en:
                            if match_english_names(original_spotify_artist, artist_name_en):
                                is_match = True
                    
                    if is_match:
                        # 一致したトラックを返す
                        track_info = {
                            "id": track['id'],
                            'artist_id': track['artists'][0]['id'],
                            'name': track['name'],
                            'artist': track['artists'][0]['name'],
                            'artist_en': track['artists'][0]['name'],
                            'album': track['album']['name'],
                            'release_date': track['album']['release_date'],
                            'duration_ms': track['duration_ms'],
                            'popularity': track['popularity'],
                            'preview_url': track.get('preview_url'),
                        }
                        return track_info, None
            
            # 一致しなかった場合、検索結果の最初の2件を返す
            print(f"  [DEBUG] 一致しなかったため、検索結果の最初の{len(top_results)}件を返します", file=sys.stderr)
            return None, top_results
        except Exception as e:
            print(f"Error in search_and_match: {e}", file=sys.stderr)
            return None, None

    # ①まず日本語のアーティスト名で検索
    all_search_results = None
    if artist_name_ja:
        result, search_results = search_and_match(artist_name_ja, "日本語")
        if result:
            return result, None
        # 一致しなかったが検索結果がある場合は保存
        if search_results:
            all_search_results = search_results
    
    # ②見つからなかった場合、英語名で検索
    if artist_name_en:
        result, search_results = search_and_match(artist_name_en, "英語")
        if result:
            return result, None
        # 一致しなかったが検索結果がある場合は保存（日本語検索の結果を上書き）
        if search_results:
            all_search_results = search_results
    
    return None, all_search_results


def add_spotify_info(json_data: dict) -> dict:
    """JSONデータにSpotify情報を追加（composerの後）"""
    if sp is None:
        return json_data
    
    # 既にSpotify情報がある場合はスキップ
    if 'spotify_id' in json_data and json_data.get('spotify_id'):
        return json_data
    
    title = json_data.get('title', '')
    artist = json_data.get('artist', '')
    jtotal_path = json_data.get('jtotal_path', '')
    
    if not title or not artist:
        return json_data
    
    # jtotal_pathから英語のアーティスト名を抽出
    artist_en = extract_artist_en_from_path(jtotal_path) if jtotal_path else None
    
    try:
        spotify_info, search_results = get_spotify_track_info(title, artist, artist_en)
        if spotify_info:
            # Spotify情報を追加
            if 'composer' in json_data:
                # composerの後にSpotify情報を挿入
                # 新しい辞書を作成して順序を制御
                new_json_data = {}
                for key, value in json_data.items():
                    new_json_data[key] = value
                    # composerの後にSpotify情報を挿入
                    if key == 'composer':
                        new_json_data['album'] = spotify_info.get('album')
                        new_json_data['release_date'] = spotify_info.get('release_date')
                        new_json_data['duration_ms'] = spotify_info.get('duration_ms')
                        new_json_data['spotify_id'] = spotify_info.get('id')
                        new_json_data['spotify_artist_id'] = spotify_info.get('artist_id')
                        new_json_data['spotify_artist_en'] = spotify_info.get('artist_en')
                        new_json_data['spotify_popularity'] = spotify_info.get('popularity')
                json_data = new_json_data
            else:
                # composerがない場合は、そのまま追加
                json_data['album'] = spotify_info.get('album')
                json_data['release_date'] = spotify_info.get('release_date')
                json_data['duration_ms'] = spotify_info.get('duration_ms')
                json_data['spotify_id'] = spotify_info.get('id')
                json_data['spotify_artist_id'] = spotify_info.get('artist_id')
                json_data['spotify_artist_en'] = spotify_info.get('artist_en')
                json_data['spotify_popularity'] = spotify_info.get('popularity')
            print(f"Spotify情報を取得しました: {title} - {artist}")
        else:
            print(f"Spotify情報が見つかりませんでした: {title} - {artist}")
            # 一致しなかった場合でも、検索結果の最初の2件をログに保存
            if search_results:
                print("\nspotify_log:")
                print(json.dumps(search_results, indent=2, ensure_ascii=False))
            else:
                print(f"  [DEBUG] 検索結果がありませんでした（search_results is None）", file=sys.stderr)
    except Exception as e:
        print(f"Spotify情報の取得に失敗: {e}")
    
    return json_data
