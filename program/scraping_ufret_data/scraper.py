#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
U-FRETのスクレイピングクラス
"""

from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page
from bs4 import BeautifulSoup

from .html_extraction import (
    extract_ufret_chord_progressions_and_lyrics,
    extract_chord_and_word,
    parse_composer_info
)

try:
    from .spotify import get_spotify_track_info
except ImportError:
    # 直接実行される場合のフォールバック
    from spotify import get_spotify_track_info


class MusicScraper:
    def __init__(self):
        """Initialize Playwright browser"""
        self.playwright = None
        self.browser: Browser = None
        self.context: BrowserContext = None
        self.page: Page = None

    def start_driver(self):
        """Start Playwright browser if not already running"""
        if not self.playwright:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=True)
            self.context = self.browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
            self.page = self.context.new_page()

    def close_driver(self):
        """Safely close Playwright browser"""
        if self.page:
            self.page.close()
            self.page = None
        if self.context:
            self.context.close()
            self.context = None
        if self.browser:
            self.browser.close()
            self.browser = None
        if self.playwright:
            self.playwright.stop()
            self.playwright = None

    def get_music_data(self, url, chord_id):
        """Get music data from specified URL"""
        try:
            # ページを読み込む（DOMContentLoadedまで待つ）
            self.page.goto(url, wait_until="domcontentloaded", timeout=20000)
            
            # コード譜要素が表示されるまで待つ（最大3秒に短縮）
            try:
                self.page.wait_for_selector('#my-chord-data', timeout=3000, state='attached')
            except Exception:
                # 要素が見つからない場合は早期チェックして終了
                html_content = self.page.content()
                soup = BeautifulSoup(html_content, 'html.parser')
                chord_element = soup.find(id='my-chord-data')
                if not chord_element:
                    print("コード譜要素が見つかりませんでした。")
                    return None
            
            # 少し待機してJavaScriptの実行を待つ（必要に応じて）
            self.page.wait_for_timeout(500)  # 1秒から0.5秒に短縮
            
            # HTMLを取得
            html_content = self.page.content()
            soup = BeautifulSoup(html_content, 'html.parser')
            
            chord_element = soup.find(id='my-chord-data')
            if not chord_element:
                print("コード譜要素が見つかりませんでした。")
                return None
            
            ufret_chord_progressions_and_lyrics = extract_ufret_chord_progressions_and_lyrics(str(chord_element))
            # chord_and_wordは各要素内に含まれるため、別途取得しない
            
            info_container = soup.find('div', class_='card card-body bg-light p-2')
            if not info_container:
                print("情報コンテナが見つかりませんでした。")
                return None
            
            song_name_element = info_container.find(class_='show_name')
            artist_element = info_container.find(class_='show_artist')
            composer_element = info_container.find(class_='show_lyrics')
            badge_info_element = info_container.find(class_='badge-info')
            if badge_info_element is not None:
                # "初心者向け簡単コード" または "ピアノソロ初級" の場合は原曲がある
                print("これは別に原曲があります")
                return None
            if not all([song_name_element, artist_element, composer_element]):
                print("必要な要素が見つかりませんでした。")
                return None
            
            song_name = song_name_element.text.strip()
            artist_name = artist_element.text.strip()
            composer_text = composer_element.text.strip()
            
            lyricist, composer = parse_composer_info(composer_text)
            
            # キー選択情報を取得
            ufret_original_key = None
            ufret_capo = None
            
            # <select name="key_capo"> の選択値を取得
            key_capo_select = soup.find('select', {'name': 'key_capo'})
            if key_capo_select:
                selected_option = key_capo_select.find('option', selected=True)
                if selected_option:
                    ufret_original_key = int(selected_option.get('value'))
            
            # <select name="keyselect"> の選択値を取得
            keyselect_select = soup.find('select', {'name': 'keyselect'})
            if keyselect_select:
                selected_option = keyselect_select.find('option', selected=True)
                if selected_option:
                    ufret_capo = int(selected_option.get('value'))


            print(f"Spotifyからメタデータを取得中: {song_name} - {artist_name}")
            import time
            spotify_start_time = time.time()
            spotify_data = get_spotify_track_info(song_name, artist_name)
            spotify_time = time.time() - spotify_start_time
            print(f"  Spotify API処理時間: {spotify_time:.2f}秒")

            # Spotify情報があるかチェック
            has_spotify_info = spotify_data is not None
            
            if has_spotify_info:
                print("  Spotify情報が見つかりました。")
            else:
                print("  Spotify情報が見つかりませんでしたが、データを保存します...")

            # フィールドの順序を指定通りに設定
            data = {
                'ufret_id': chord_id,
                'title': song_name,
                'artist': artist_name,
                'lyricist': lyricist,
                'composer': composer,
                'ufret_chord_progressions_and_lyrics': ufret_chord_progressions_and_lyrics,
            }
            
            # キー情報を追加
            if ufret_original_key is not None:
                data['ufret_original_key'] = ufret_original_key
            if ufret_capo is not None:
                data['ufret_capo'] = ufret_capo 

            # Spotify情報がある場合のみ追加
            if spotify_data:
                data['album'] = spotify_data.get('album')
                data['spotify_id'] = spotify_data.get('spotify_id')
                data['release_date'] = spotify_data.get('release_date')
                data['spotify_artist_id'] = spotify_data.get('artist_id')
                data['spotify_artist_en'] = spotify_data.get('artist')
                data['spotify_popularity'] = spotify_data.get('popularity')

            return data
                
        except Exception as e:
            print(f"データ取得中にエラーが発生しました: {str(e)}")
            return None

