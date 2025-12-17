"""
評価関数
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from .chord_normalization import IDX, ROOTS_12

def split_key(k: str):
    """Split key into root and minor flag. 'Am' -> ('A', True), 'C' -> ('C', False)"""
    if k.endswith("m"):
        return k[:-1], True
    return k, False

def fifth_neighbors(root: str):
    """Return fifth circle neighbors (dominant and subdominant)."""
    i = IDX[root]
    return {ROOTS_12[(i+7)%12], ROOTS_12[(i-7)%12]}  # +5th, -5th

def is_musically_close(true_k: str, pred_k: str) -> bool:
    """
    Check if predicted key is musically close to true key.
    Considers:
    - Exact match
    - Parallel keys (C <-> Am, G <-> Em, etc.)
    - Relative keys / 同主調 (C <-> Cm)
    - Fifth neighbors (C <-> G / F)
    """
    tr, tmin = split_key(true_k)
    pr, pmin = split_key(pred_k)

    if tr == pr and tmin == pmin:
        return True  # exact match

    # Parallel keys: C <-> Am (same key signature)
    tr_i, pr_i = IDX[tr], IDX[pr]
    if (not tmin) and pmin and pr_i == (tr_i - 3) % 12:
        return True  # Major -> parallel minor
    if tmin and (not pmin) and pr_i == (tr_i + 3) % 12:
        return True  # Minor -> parallel major

    # Relative / 同主調: same root different mode (C <-> Cm)
    if tr == pr and tmin != pmin:
        return True

    # Fifth neighbors (same mode)
    if tmin == pmin and pr in fifth_neighbors(tr):
        return True

    return False

def top_k_accuracy(y_true, y_pred_proba, clf_classes, k=3):
    """Calculate top-k accuracy from probability predictions."""
    top_k_indices = np.argsort(y_pred_proba, axis=1)[:, -k:][:, ::-1]
    label_to_idx = {label: idx for idx, label in enumerate(clf_classes)}
    correct = 0
    valid_count = 0
    for i, true_label in enumerate(y_true):
        if true_label not in label_to_idx:
            continue
        valid_count += 1
        true_idx = label_to_idx[true_label]
        if true_idx in top_k_indices[i]:
            correct += 1
    return correct / valid_count if valid_count > 0 else 0.0

def evaluate_with_cv(texts_all, texts_last, X_root, y, n_splits=3, random_state=42):
    """Evaluate model with cross-validation and return metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    unique_classes = sorted(list(set(y)))
    
    top1_scores = []
    top3_scores = []
    close1_scores = []
    close_top3_scores = []
    all_y_test = []
    all_y_pred = []
    
    print(f"Running {n_splits}-fold cross-validation...")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_root, y), 1):
        vec_all = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        vec_last = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        
        texts_all_train = [texts_all[i] for i in train_idx]
        texts_all_test = [texts_all[i] for i in test_idx]
        texts_last_train = [texts_last[i] for i in train_idx]
        texts_last_test = [texts_last[i] for i in test_idx]
        
        X_all_train = vec_all.fit_transform(texts_all_train)
        X_all_test = vec_all.transform(texts_all_test)
        X_last_train = vec_last.fit_transform(texts_last_train)
        X_last_test = vec_last.transform(texts_last_test)
        X_root_train = X_root[train_idx]
        X_root_test = X_root[test_idx]
        
        X_train_fold = hstack([X_all_train, X_last_train, csr_matrix(X_root_train)]).tocsr()
        X_test_fold = hstack([X_all_test, X_last_test, csr_matrix(X_root_test)]).tocsr()
        
        y_train_fold = [y[i] for i in train_idx]
        y_test_fold = [y[i] for i in test_idx]
        
        clf = LogisticRegression(
            solver="saga",
            max_iter=2000,
            n_jobs=-1,
            C=4.0,
        )
        clf.fit(X_train_fold, y_train_fold)
        
        pred_fold = clf.predict(X_test_fold)
        pred_proba_fold = clf.predict_proba(X_test_fold)
        
        top1 = accuracy_score(y_test_fold, pred_fold)
        top3 = top_k_accuracy(y_test_fold, pred_proba_fold, clf.classes_, k=3)
        
        close1 = np.mean([is_musically_close(t, p) for t, p in zip(y_test_fold, pred_fold)])
        
        top3_indices = np.argsort(-pred_proba_fold, axis=1)[:, :3]
        classes_array = np.array(clf.classes_)
        close_top3_list = []
        for i, true_label in enumerate(y_test_fold):
            top3_preds = [classes_array[idx] for idx in top3_indices[i]]
            close_top3_list.append(any(is_musically_close(true_label, pk) for pk in top3_preds))
        close_top3 = np.mean(close_top3_list)
        
        top1_scores.append(top1)
        top3_scores.append(top3)
        close1_scores.append(close1)
        close_top3_scores.append(close_top3)
        all_y_test.extend(y_test_fold)
        all_y_pred.extend(pred_fold)
        
        print(f"Fold {fold}: Top-1={top1:.4f}, Top-3={top3:.4f}, Close-1={close1:.4f}, Close-Top3={close_top3:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Cross-Validation Results ({n_splits}-fold):")
    print(f"Top-1 Accuracy (exact):     {np.mean(top1_scores):.4f} ± {np.std(top1_scores):.4f}")
    print(f"Top-3 Accuracy (exact):     {np.mean(top3_scores):.4f} ± {np.std(top3_scores):.4f}")
    print(f"Close-1 Accuracy (musical): {np.mean(close1_scores):.4f} ± {np.std(close1_scores):.4f}")
    print(f"Close-Top3 Accuracy:        {np.mean(close_top3_scores):.4f} ± {np.std(close_top3_scores):.4f}")
    print(f"{'='*60}\n")
    
    print("Classification Report (aggregated over all folds):")
    print(classification_report(all_y_test, all_y_pred, digits=3, zero_division=0))
    
    cm = confusion_matrix(all_y_test, all_y_pred, labels=unique_classes)
    class_counts = {cls: sum(1 for label in all_y_test if label == cls) for cls in unique_classes}
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_classes = [cls for cls, _ in sorted_classes]
    top_indices = [unique_classes.index(cls) for cls in top_classes]
    
    print(f"\nConfusion Matrix (top {len(top_classes)} most common classes):")
    print("Classes:", ", ".join(top_classes))
    cm_top = cm[np.ix_(top_indices, top_indices)]
    print(cm_top)
    
    return top1_scores, top3_scores

