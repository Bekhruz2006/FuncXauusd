"""
Model validation utilities
Ensures model quality before deployment
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def validate_class_balance(
    labels: pd.Series,
    min_ratio: float = 0.2,
    min_samples_per_class: int = 100
) -> Tuple[bool, dict]:
    """
    Validate that dataset has adequate class balance
    
    Args:
        labels: Target labels
        min_ratio: Minimum ratio for minority class
        min_samples_per_class: Minimum samples per class
        
    Returns:
        Tuple of (is_valid, statistics)
    """
    value_counts = labels.value_counts()
    total = len(labels)
    
    stats = {
        'total_samples': total,
        'class_counts': value_counts.to_dict(),
        'class_ratios': (value_counts / total).to_dict(),
        'minority_ratio': value_counts.min() / total,
        'min_samples': value_counts.min()
    }
    
    is_valid = (
        stats['minority_ratio'] >= min_ratio and
        stats['min_samples'] >= min_samples_per_class
    )
    
    return is_valid, stats


def validate_sample_size(
    data: pd.DataFrame,
    min_samples: int = 1000,
    recommended_samples: int = 5000
) -> Tuple[bool, dict]:
    """
    Validate dataset size
    
    Args:
        data: Dataset to validate
        min_samples: Minimum acceptable samples
        recommended_samples: Recommended sample count
        
    Returns:
        Tuple of (meets_minimum, statistics)
    """
    n_samples = len(data)
    
    stats = {
        'total_samples': n_samples,
        'meets_minimum': n_samples >= min_samples,
        'meets_recommended': n_samples >= recommended_samples,
        'ratio_to_minimum': n_samples / min_samples if min_samples > 0 else 0
    }
    
    return stats['meets_minimum'], stats


def validate_feature_quality(
    features: pd.DataFrame,
    max_missing_ratio: float = 0.1,
    max_constant_features: int = 0
) -> Tuple[bool, dict]:
    """
    Validate feature quality
    
    Args:
        features: Feature DataFrame
        max_missing_ratio: Maximum allowed missing value ratio
        max_constant_features: Maximum allowed constant features
        
    Returns:
        Tuple of (is_valid, statistics)
    """
    n_samples, n_features = features.shape
    
    # Missing values
    missing_counts = features.isnull().sum()
    missing_ratios = missing_counts / n_samples
    max_missing = missing_ratios.max()
    
    # Constant features
    constant_mask = features.std() == 0
    n_constant = constant_mask.sum()
    
    # Infinite values
    n_infinite = np.isinf(features.values).sum()
    
    stats = {
        'n_features': n_features,
        'n_samples': n_samples,
        'max_missing_ratio': max_missing,
        'features_with_missing': (missing_ratios > 0).sum(),
        'constant_features': n_constant,
        'infinite_values': n_infinite,
        'problematic_features': []
    }
    
    # Identify problematic features
    for col in features.columns:
        if missing_ratios[col] > max_missing_ratio:
            stats['problematic_features'].append(
                f"{col}: {missing_ratios[col]:.1%} missing"
            )
        if constant_mask[col]:
            stats['problematic_features'].append(f"{col}: constant")
    
    is_valid = (
        max_missing <= max_missing_ratio and
        n_constant <= max_constant_features and
        n_infinite == 0
    )
    
    return is_valid, stats


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None
) -> dict:
    """
    Calculate comprehensive classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Specificity
        if (tn + fp) > 0:
            metrics['specificity'] = tn / (tn + fp)
    
    # ROC AUC (if probabilities provided)
    if y_pred_proba is not None:
        try:
            from sklearn.metrics import roc_auc_score
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except Exception:
            pass
    
    return metrics


def validate_model_performance(
    metrics: dict,
    min_accuracy: float = 0.70,
    min_f1: float = 0.65
) -> Tuple[bool, dict]:
    """
    Validate that model meets performance thresholds
    
    Args:
        metrics: Dictionary of model metrics
        min_accuracy: Minimum acceptable accuracy
        min_f1: Minimum acceptable F1 score
        
    Returns:
        Tuple of (meets_threshold, validation_results)
    """
    results = {
        'accuracy_valid': metrics.get('accuracy', 0) >= min_accuracy,
        'f1_valid': metrics.get('f1', 0) >= min_f1,
        'accuracy': metrics.get('accuracy', 0),
        'f1': metrics.get('f1', 0),
        'thresholds': {
            'min_accuracy': min_accuracy,
            'min_f1': min_f1
        }
    }
    
    meets_threshold = results['accuracy_valid'] and results['f1_valid']
    
    return meets_threshold, results


def check_overfitting(
    train_metrics: dict,
    val_metrics: dict,
    max_gap: float = 0.10
) -> Tuple[bool, dict]:
    """
    Check for overfitting by comparing train and validation metrics
    
    Args:
        train_metrics: Metrics on training set
        val_metrics: Metrics on validation set
        max_gap: Maximum allowed gap between train and val
        
    Returns:
        Tuple of (no_overfitting, analysis)
    """
    analysis = {
        'accuracy_gap': train_metrics.get('accuracy', 0) - val_metrics.get('accuracy', 0),
        'f1_gap': train_metrics.get('f1', 0) - val_metrics.get('f1', 0),
        'train_accuracy': train_metrics.get('accuracy', 0),
        'val_accuracy': val_metrics.get('accuracy', 0),
        'max_allowed_gap': max_gap
    }
    
    # No overfitting if gap is within acceptable range
    no_overfitting = (
        analysis['accuracy_gap'] <= max_gap and
        analysis['f1_gap'] <= max_gap
    )
    
    analysis['verdict'] = 'OK' if no_overfitting else 'OVERFITTING DETECTED'
    
    return no_overfitting, analysis


def print_validation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str = 'Validation'
) -> None:
    """
    Print detailed validation report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        dataset_name: Name of dataset (for display)
    """
    print(f"\n{'='*60}")
    print(f"  {dataset_name} Set Performance")
    print(f"{'='*60}")
    
    metrics = calculate_classification_metrics(y_true, y_pred)
    
    print(f"\n📊 Metrics:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1 Score:  {metrics['f1']:.4f}")
    
    if 'specificity' in metrics:
        print(f"   Specificity: {metrics['specificity']:.4f}")
    
    if 'roc_auc' in metrics:
        print(f"   ROC AUC:   {metrics['roc_auc']:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n🔢 Confusion Matrix:")
    print(f"   True Neg:  {metrics.get('true_negatives', 'N/A')}")
    print(f"   False Pos: {metrics.get('false_positives', 'N/A')}")
    print(f"   False Neg: {metrics.get('false_negatives', 'N/A')}")
    print(f"   True Pos:  {metrics.get('true_positives', 'N/A')}")
    
    # Classification Report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))


def validate_prediction_distribution(
    predictions: np.ndarray,
    min_predictions: int = 10
) -> Tuple[bool, dict]:
    """
    Validate that model makes reasonable number of predictions
    
    Args:
        predictions: Array of binary predictions
        min_predictions: Minimum number of positive predictions
        
    Returns:
        Tuple of (is_valid, statistics)
    """
    n_total = len(predictions)
    n_positive = np.sum(predictions == 1)
    n_negative = np.sum(predictions == 0)
    
    stats = {
        'total_predictions': n_total,
        'positive_predictions': n_positive,
        'negative_predictions': n_negative,
        'positive_ratio': n_positive / n_total if n_total > 0 else 0
    }
    
    is_valid = (
        n_positive >= min_predictions and
        0.05 <= stats['positive_ratio'] <= 0.95  # Not too extreme
    )
    
    return is_valid, stats


def comprehensive_model_validation(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict
) -> Tuple[bool, dict]:
    """
    Comprehensive validation pipeline
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Configuration dictionary
        
    Returns:
        Tuple of (passes_validation, detailed_report)
    """
    report = {
        'timestamp': pd.Timestamp.now(),
        'checks': {},
        'metrics': {},
        'warnings': [],
        'errors': []
    }
    
    # 1. Class balance check
    balance_valid, balance_stats = validate_class_balance(
        y_train,
        min_ratio=config.get('validation', {}).get('criteria', {}).get('min_class_balance', 0.2)
    )
    report['checks']['class_balance'] = balance_valid
    report['metrics']['class_balance'] = balance_stats
    
    if not balance_valid:
        report['warnings'].append(f"Class imbalance: {balance_stats['minority_ratio']:.2%}")
    
    # 2. Sample size check
    size_valid, size_stats = validate_sample_size(
        X_train,
        min_samples=config.get('search', {}).get('space', {}).get('min_samples', [1000])[0]
    )
    report['checks']['sample_size'] = size_valid
    report['metrics']['sample_size'] = size_stats
    
    # 3. Feature quality check
    feature_valid, feature_stats = validate_feature_quality(X_train)
    report['checks']['feature_quality'] = feature_valid
    report['metrics']['feature_quality'] = feature_stats
    
    if not feature_valid:
        report['errors'].extend(feature_stats['problematic_features'])
    
    # 4. Model performance check
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_metrics = calculate_classification_metrics(y_train, y_train_pred)
    val_metrics = calculate_classification_metrics(y_val, y_val_pred)
    
    report['metrics']['train'] = train_metrics
    report['metrics']['validation'] = val_metrics
    
    perf_valid, perf_results = validate_model_performance(
        val_metrics,
        min_accuracy=config.get('search', {}).get('targets', {}).get('val_accuracy', 0.75)
    )
    report['checks']['performance'] = perf_valid
    
    # 5. Overfitting check
    no_overfit, overfit_analysis = check_overfitting(train_metrics, val_metrics)
    report['checks']['no_overfitting'] = no_overfit
    report['metrics']['overfitting'] = overfit_analysis
    
    if not no_overfit:
        report['warnings'].append(f"Overfitting detected: {overfit_analysis['accuracy_gap']:.2%} gap")
    
    # 6. Prediction distribution check
    pred_valid, pred_stats = validate_prediction_distribution(y_val_pred)
    report['checks']['prediction_distribution'] = pred_valid
    report['metrics']['predictions'] = pred_stats
    
    # Overall validation
    all_checks_passed = all(report['checks'].values())
    report['overall_valid'] = all_checks_passed
    
    return all_checks_passed, report