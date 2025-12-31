"""
LINE BERTを用いた感情分析モジュール
"""
from __future__ import annotations
from typing import Dict, Optional, List, Union
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# 感情ラベル
EMOTION_LABELS = ["JOY", "SADNESS", "ANTICIPATION", "SURPRISE", "ANGER", "FEAR", "DISGUST", "TRUST"]

# モデル名
PRETRAINED_MODEL_NAME = "koshin2001/Japanese-to-emotions"
SENTIMENT_MODEL_NAME = "koheiduck/bert-japanese-finetuned-sentiment"

# グローバル変数でモデルとトークナイザーをキャッシュ
_model = None
_tokenizer = None
_device = None
_sentiment_classifier = None

def _get_model_and_tokenizer():
    """モデルとトークナイザーを取得（シングルトン）"""
    global _model, _tokenizer, _device
    
    if _model is None or _tokenizer is None:
        print(f"Loading emotion analysis model: {PRETRAINED_MODEL_NAME}")
        _tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, trust_remote_code=True)
        
        # 感情分類用にファインチューニングされたモデルを読み込む
        _model = AutoModelForSequenceClassification.from_pretrained(
            PRETRAINED_MODEL_NAME, trust_remote_code=True
        )
        
        # デバイス設定
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(_device)
        _model.eval()
        print(f"Model loaded on device: {_device}")
    
    return _model, _tokenizer, _device

def analyze_emotion(lyric: str) -> Optional[Dict[str, float]]:
    """
    歌詞から感情を分析する
    
    Args:
        lyric: 分析対象の歌詞
    
    Returns:
        感情スコアの辞書 {JOY: 0.5, SADNESS: 0.3, ...}
        歌詞が空の場合はNone
    """
    if not lyric or not lyric.strip():
        return None
    
    try:
        model, tokenizer, device = _get_model_and_tokenizer()
        
        # トークナイズ
        inputs = tokenizer(
            lyric,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 推論
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # multi_label_classificationなので、シグモイドで確率に変換
        # 各感情は独立に検出される（複数の感情が同時に高くなる可能性がある）
        probs = torch.sigmoid(logits)
        
        # ラベルに対応する確率を取得
        # モデルのラベル順序を確認（LABEL_0からLABEL_7）
        # 実際の感情ラベルとのマッピングが必要
        emotion_scores = {}
        
        if probs.shape[1] == len(EMOTION_LABELS):
            # モデルの出力が8次元の場合
            # 注意: モデルのラベル順序（LABEL_0～LABEL_7）とEMOTION_LABELSの順序が一致していることを前提
            # 実際のモデルのラベルマッピングを確認する必要がある
            for i, label in enumerate(EMOTION_LABELS):
                emotion_scores[label] = round(float(probs[0][i].item()), 3)
        else:
            print(f"Warning: Model output dimension ({probs.shape[1]}) doesn't match emotion labels ({len(EMOTION_LABELS)})")
            # とりあえず、最初の8次元を使用
            num_labels = min(probs.shape[1], len(EMOTION_LABELS))
            for i in range(num_labels):
                emotion_scores[EMOTION_LABELS[i]] = round(float(probs[0][i].item()), 3)
            # 残りは0で埋める
            for i in range(num_labels, len(EMOTION_LABELS)):
                emotion_scores[EMOTION_LABELS[i]] = 0.0
        
        return emotion_scores
    
    except Exception as e:
        print(f"Error in emotion analysis: {e}")
        # エラー時はデフォルト値（すべて0）を返す
        return {label: 0.0 for label in EMOTION_LABELS}

def analyze_emotion_batch(lyrics: List[str]) -> List[Optional[Dict[str, float]]]:
    """
    複数の歌詞をバッチ処理で感情分析する（処理時間短縮のため）
    
    Args:
        lyrics: 分析対象の歌詞のリスト
    
    Returns:
        感情スコアの辞書のリスト（空の歌詞はNone）
    """
    if not lyrics:
        return []
    
    start_time = time.time()
    
    # 空でない歌詞のインデックスを記録
    non_empty_indices = []
    non_empty_lyrics = []
    for i, lyric in enumerate(lyrics):
        if lyric and lyric.strip():
            non_empty_indices.append(i)
            non_empty_lyrics.append(lyric.strip())
    
    if not non_empty_lyrics:
        return [None] * len(lyrics)
    
    try:
        model, tokenizer, device = _get_model_and_tokenizer()
        
        # バッチでトークナイズ
        tokenize_start = time.time()
        inputs = tokenizer(
            non_empty_lyrics,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        tokenize_time = time.time() - tokenize_start
        
        # バッチで推論
        inference_start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # シグモイドで確率に変換
        probs = torch.sigmoid(logits)
        inference_time = time.time() - inference_start
        
        # 結果をリストに変換
        postprocess_start = time.time()
        results = [None] * len(lyrics)
        for idx, orig_idx in enumerate(non_empty_indices):
            emotion_scores = {}
            if probs.shape[1] == len(EMOTION_LABELS):
                for i, label in enumerate(EMOTION_LABELS):
                    emotion_scores[label] = round(float(probs[idx][i].item()), 3)
            else:
                num_labels = min(probs.shape[1], len(EMOTION_LABELS))
                for i in range(num_labels):
                    emotion_scores[EMOTION_LABELS[i]] = round(float(probs[idx][i].item()), 3)
                for i in range(num_labels, len(EMOTION_LABELS)):
                    emotion_scores[EMOTION_LABELS[i]] = 0.0
            results[orig_idx] = emotion_scores
        postprocess_time = time.time() - postprocess_start
        
        total_time = time.time() - start_time
        print(f"[感情分析] 処理時間: {total_time:.2f}秒 (トークナイズ: {tokenize_time:.2f}秒, 推論: {inference_time:.2f}秒, 後処理: {postprocess_time:.2f}秒) - {len(non_empty_lyrics)}件の歌詞をバッチ処理")
        
        return results
    
    except Exception as e:
        total_time = time.time() - start_time
        print(f"Error in batch emotion analysis: {e} (処理時間: {total_time:.2f}秒)")
        # エラー時はデフォルト値（すべて0）を返す
        default_emotion = {label: 0.0 for label in EMOTION_LABELS}
        return [default_emotion if (lyrics[i] and lyrics[i].strip()) else None for i in range(len(lyrics))]

def add_emotion_to_sections(sections: list) -> list:
    """
    セクションリストの各セクションに感情分析を追加（バッチ処理で高速化）
    
    Args:
        sections: セクションのリスト（各セクションに'lyric'フィールドが必要）
    
    Returns:
        感情分析を追加したセクションのリスト
    """
    if not sections:
        return sections
    
    start_time = time.time()
    
    # すべての歌詞をまとめて取得
    lyrics_list = [section.get("lyric", "") for section in sections]
    
    # バッチ処理で分析
    emotion_results = analyze_emotion_batch(lyrics_list)
    
    # 結果を各セクションに追加
    default_emotion = {label: 0.0 for label in EMOTION_LABELS}
    
    for i, section in enumerate(sections):
        if emotion_results[i] is not None:
            section["emotion"] = emotion_results[i]
        else:
            # 空の歌詞にはデフォルト値
            section["emotion"] = default_emotion
    
    total_time = time.time() - start_time
    print(f"[感情分析] セクション処理完了: {len(sections)}セクション, 合計時間: {total_time:.2f}秒")
    
    return sections

def _get_sentiment_classifier():
    """Sentiment分類器を取得（シングルトン）"""
    global _sentiment_classifier
    
    if _sentiment_classifier is None:
        print(f"Loading sentiment analysis model: {SENTIMENT_MODEL_NAME}")
        _sentiment_classifier = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL_NAME,
            tokenizer=SENTIMENT_MODEL_NAME,
            device=0 if torch.cuda.is_available() else -1
        )
        print(f"Sentiment model loaded")
    
    return _sentiment_classifier

def analyze_sentiment(lyrics: Union[str, List[str]]) -> Optional[Union[Dict[str, Union[str, float]], List[Dict[str, Union[str, float]]]]]:
    """
    歌詞からセンチメントを分析する
    
    Args:
        lyrics: 分析対象の歌詞（文字列または文字列のリスト）
    
    Returns:
        単一の歌詞の場合: {'label': 'positive' or 'negative', 'score': 0.0-1.0}
        複数の歌詞の場合: [{'label': 'positive', 'score': 0.82}, ...]
        歌詞が空の場合はNone
    """
    if not lyrics:
        return None
    
    # 文字列の場合はリストに変換
    is_single = isinstance(lyrics, str)
    if is_single:
        if not lyrics.strip():
            return None
        lyrics_list = [lyrics]
    else:
        # リストの場合、空の要素を除外
        lyrics_list = [lyr for lyr in lyrics if lyr and lyr.strip()]
        if not lyrics_list:
            return None
    
    try:
        classifier = _get_sentiment_classifier()
        
        # バッチ処理で分析（まとめて投げる方が早い）
        results = classifier(lyrics_list)
        
        # 結果を正規化（labelを小文字に統一、scoreをfloatに）
        normalized_results = []
        for result in results:
            if isinstance(result, dict):
                label = result.get('label', '').lower()
                score = float(result.get('score', 0.0))
                normalized_results.append({
                    'label': label,
                    'score': round(score, 3)
                })
            else:
                # フォールバック
                normalized_results.append({
                    'label': 'neutral',
                    'score': 0.5
                })
        
        # 単一の場合は辞書を返す、複数の場合はリストを返す
        if is_single:
            return normalized_results[0] if normalized_results else None
        else:
            return normalized_results
    
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        # エラー時はデフォルト値
        default_result = {'label': 'neutral', 'score': 0.5}
        if is_single:
            return default_result
        else:
            return [default_result] * len(lyrics_list)

def add_sentiment_to_sections(sections: list) -> list:
    """
    セクションリストの各セクションにセンチメント分析を追加
    
    Args:
        sections: セクションのリスト（各セクションに'lyric'フィールドが必要）
    
    Returns:
        センチメント分析を追加したセクションのリスト
    """
    # すべての歌詞をまとめて取得
    lyrics_list = [section.get("lyric", "") for section in sections]
    
    # 空でない歌詞のインデックスを記録
    non_empty_indices = []
    non_empty_lyrics = []
    for i, lyric in enumerate(lyrics_list):
        if lyric and lyric.strip():
            non_empty_indices.append(i)
            non_empty_lyrics.append(lyric)
    
    # バッチ処理で分析（まとめて投げる方が早い）
    if non_empty_lyrics:
        sentiment_results = analyze_sentiment(non_empty_lyrics)
    else:
        sentiment_results = None
    
    # 結果を各セクションに追加
    default_sentiment = {'label': 'neutral', 'score': 0.5}
    
    if sentiment_results and len(sentiment_results) == len(non_empty_lyrics):
        # 非空の歌詞の結果をマッピング
        sentiment_map = {non_empty_indices[i]: sentiment_results[i] 
                        for i in range(len(non_empty_indices))}
        
        for i, section in enumerate(sections):
            if i in sentiment_map:
                section["sentiment"] = sentiment_map[i]
            else:
                # 空の歌詞にはデフォルト値
                section["sentiment"] = default_sentiment
    else:
        # エラー時はデフォルト値
        for section in sections:
            section["sentiment"] = default_sentiment
    
    return sections

