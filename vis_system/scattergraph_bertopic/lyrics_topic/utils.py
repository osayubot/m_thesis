"""
ユーティリティ関数
"""
from __future__ import annotations
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """ロガーをセットアップ"""
    logger = logging.getLogger("lyrics_topic")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def detect_device() -> str:
    """利用可能なデバイスを自動検出"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def softmax(x: np.ndarray, temperature: float = 1.0, axis: int = -1) -> np.ndarray:
    """Softmax関数"""
    x = x / temperature
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def calculate_entropy(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """確率分布のエントロピーを計算"""
    # ゼロ除算を避ける
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log(probs), axis=axis)


def ensure_dir(path: str | Path) -> Path:
    """ディレクトリが存在しない場合は作成"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

