"""
LINE BERTを用いた感情分析モジュール
"""
from __future__ import annotations
from typing import Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 感情ラベル
EMOTION_LABELS = ["JOY", "SADNESS", "ANTICIPATION", "SURPRISE", "ANGER", "FEAR", "DISGUST", "TRUST"]

# モデル名
PRETRAINED_MODEL_NAME = "koshin2001/Japanese-to-emotions"

# グローバル変数でモデルとトークナイザーをキャッシュ
_model = None
_tokenizer = None
_device = None

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

def add_emotion_to_sections(sections: list) -> list:
    """
    セクションリストの各セクションに感情分析を追加
    
    Args:
        sections: セクションのリスト（各セクションに'lyric'フィールドが必要）
    
    Returns:
        感情分析を追加したセクションのリスト
    """
    for section in sections:
        lyric = section.get("lyric", "")
        if lyric:
            emotion_scores = analyze_emotion(lyric)
            if emotion_scores:
                section["emotion"] = emotion_scores
        else:
            # 歌詞が空の場合はデフォルト値
            section["emotion"] = {label: 0.0 for label in EMOTION_LABELS}
    
    return sections

