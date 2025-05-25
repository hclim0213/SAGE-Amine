"""
Copyright (c) 2024 Hocheol Lim.
"""
import numpy as np

def calculate_metrics(preds, targets, scoring):
    metrics = {}
    for name, scorer in scoring.items():
        metrics[name] = scorer._score_func(targets, preds)
    
    return metrics

def get_scoring(keyword: str):
    from sklearn.metrics import make_scorer, get_scorer
    from sklearn.metrics import accuracy_score, auc, f1_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    if keyword == 'classification':
        scoring = {
            'acc': get_scorer("accuracy"),
            'roc_auc': get_scorer("roc_auc"),
            'precision': get_scorer("precision"),
            'recall': get_scorer("recall"),
            'f1': get_scorer("f1"),
        }
    
    if keyword == 'regression':
        scoring = {
            'r2': make_scorer(r2_score),
            'mse': make_scorer(mean_squared_error, greater_is_better=False),
            'mae': make_scorer(mean_absolute_error, greater_is_better=False),
        }
    
    return scoring