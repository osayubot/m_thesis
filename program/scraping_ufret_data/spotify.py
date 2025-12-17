import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
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

def get_spotify_track_info(track_name, artist_name):
    # 認証情報なし
    if sp is None:
        print("Warning: Spotify credentials not configured")
        return None

    # 曲を検索
    query = f"track:{track_name} artist:{artist_name}"
    results = sp.search(q=query, type='track', limit=1)

    # 検索結果がある場合、メタ情報を取得
    if results['tracks']['items']:
        track = results['tracks']['items'][0]
        
        # メタ情報を辞書に格納
        track_info = {
            'spotify_id': track['id'],  # track IDを追加
            'artist_id': track['artists'][0]['id'],
            'name': track['name'],
            'artist': track['artists'][0]['name'],
            'album': track['album']['name'],
            'release_date': track['album']['release_date'],
            'duration_ms': track['duration_ms'],
            'popularity': track['popularity'],
            'explicit': track['explicit'],
            'preview_url': track['preview_url'],
            'external_url': track['external_urls']['spotify'],
            'uri': track['uri']
        }

        return track_info
    else:
        return None