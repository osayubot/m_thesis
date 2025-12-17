#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HTMLからデータを抽出して、移調前のJSONを保存する関数
"""

import os
import json
from playwright.sync_api import sync_playwright

from .html_extraction import (
    extract_key_info_from_page,
    clean_html,
    extract_jtotal_chord_progressions_and_lyrics
)


def extract_and_save_html_data(url: str, jtotal_path: str, existing_spotify_info: dict = None) -> dict:
    """
    URLからHTMLを取得し、データを抽出して移調前のJSONを保存する
    
    Args:
        url: J-Total MusicのURL
        jtotal_path: J-Total Musicのパス
        existing_spotify_info: 既存のSpotify情報
    
    Returns:
        抽出されたデータの辞書。キー情報があれば含まれる。
        {
            'success': True/False,
            'key_info': {...},  # キー情報がある場合
            'data': {...}  # 抽出されたデータ
        }
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        page = context.new_page()
        
        try:
            # ページを読み込む
            page.goto(url, wait_until="domcontentloaded", timeout=20000)
            
            # JavaScriptの実行を待つ
            page.wait_for_timeout(3000)
            
            # 初期HTMLを取得
            initial_html = page.content()
            
            # HTMLを保存
            html_path = f"data/jtotal/html/{jtotal_path}.html"
            os.makedirs(os.path.dirname(html_path), exist_ok=True)
            with open(html_path, 'w', encoding='utf-8', errors='replace') as f:
                f.write(initial_html)
            print(f"HTMLを保存しました: {html_path}")
            
            # box2の存在をチェック
            has_box2 = page.evaluate("""
                () => {
                    const box2 = document.querySelector('div.box2');
                    return box2 !== null;
                }
            """)
            
            # HTMLからデータを抽出（box2がなくても可能な限り抽出）
            extracted_data = page.evaluate("""
                () => {
                    let box2_data = {
                        title: '',
                        artist: '',
                        lyrics: '',
                        melody: '',
                        originalKey: '',
                        capo: '',
                        play: ''
                    };
                    let tt_content = '';
                    
                    const box2 = document.querySelector('div.box2');
                    if (box2) {
                        const h1 = box2.querySelector('h1');
                        const h2 = box2.querySelector('h2');
                        const h3 = box2.querySelector('h3');
                        
                        if (h1) box2_data.title = h1.textContent.trim();
                        
                        if (h2) {
                            const h2_text = h2.textContent;
                            const artist_match = h2_text.match(/歌[：:]([^/]+)/);
                            const lyrics_match = h2_text.match(/詞[：:]([^/]+)/);
                            const melody_match = h2_text.match(/曲[：:](.+)/);
                            if (artist_match) box2_data.artist = artist_match[1].trim();
                            if (lyrics_match) box2_data.lyrics = lyrics_match[1].trim();
                            if (melody_match) box2_data.melody = melody_match[1].trim();
                        }
                        
                        if (h3) {
                            const h3_text = h3.textContent;
                            const original_key_match = h3_text.match(/Original\\s+Key[：:]([^/]+)/);
                            const capo_match = h3_text.match(/Capo[：:]([^/]+)/);
                            const play_match = h3_text.match(/Play[：:](.+)/);
                            if (original_key_match) box2_data.originalKey = original_key_match[1].trim();
                            if (capo_match) box2_data.capo = capo_match[1].trim();
                            if (play_match) box2_data.play = play_match[1].trim();
                        }
                    }
                    
                    const ttElements = document.querySelectorAll('tt');
                    ttElements.forEach(tt => {
                        tt_content += tt.outerHTML + '\\n';
                    });
                    
                    return {
                        box2: box2_data,
                        tt: tt_content,
                        has_box2: box2 !== null
                    };
                }
            """)
            
            # HTMLをクリーンアップ（保存はしない）
            cleaned_html = clean_html(extracted_data['tt'])
            
            # コード進行と歌詞を抽出
            jtotal_chord_progressions_and_lyrics = extract_jtotal_chord_progressions_and_lyrics(cleaned_html)
            
            # キー情報を抽出（あれば）
            key_info = None
            original_key = None
            original_play_key = None
            if has_box2:
                key_info = extract_key_info_from_page(page)
                if key_info:
                    original_play_key = key_info.get('original_play_key')
                    original_key = key_info.get('original_key')
            
            # 移調前のJSONデータを作成
            json_data = {}
            if jtotal_path:
                json_data['jtotal_path'] = jtotal_path
            
            # 基本情報を取得
            title = extracted_data['box2'].get('title', '').strip()
            artist = extracted_data['box2'].get('artist', '').strip()
            lyricist = extracted_data['box2'].get('lyrics', '').strip()
            composer = extracted_data['box2'].get('melody', '').strip()
            
            # title、artist、lyricist、composerがすべて空の場合はJSONを保存しない
            if not title and not artist and not lyricist and not composer:
                print(f"基本情報（title, artist, lyricist, composer）が取得できなかったため、JSONを保存しません")
                return {
                    'success': False,
                    'error': '基本情報が取得できませんでした'
                }
            
            json_data.update({
                'title': title,
                'artist': artist,
                'lyricist': lyricist,
                'composer': composer,
                'jtotal_chord_progressions_and_lyrics': jtotal_chord_progressions_and_lyrics
            })
            
            # キー情報がある場合は追加
            if original_key:
                json_data['original_key'] = original_key
            if original_play_key:
                json_data['original_play_key'] = original_play_key
            
            # Spotify情報を追加
            if existing_spotify_info:
                spotify_keys = ['album', 'spotify_id', 'release_date', 'duration_ms', 
                               'spotify_artist_id', 'spotify_artist_en', 'spotify_popularity']
                for key in spotify_keys:
                    if key in existing_spotify_info:
                        json_data[key] = existing_spotify_info[key]
            
            # 移調前のJSONを保存
            json_path = f"data/jtotal/json/raw/{jtotal_path}.json"
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, 'w', encoding='utf-8', errors='replace') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            print(f"移調前のJSONを保存しました: {json_path}")
            
            result = {
                'success': True,
                'key_info': key_info,
                'original_play_key': original_play_key,
                'has_box2': has_box2
            }
            
            return result
            
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            context.close()
            browser.close()

