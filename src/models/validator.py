"""
Валидация моделей и данных

Критические проверки перед обучением:
    - Баланс классов
    - Размер выборки
    - Качество кластеризации
    - Валидность признаков
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.metrics import silhouette_score


def validate_class_balance(labels: pd.Series,
                          min_balance: float = 0.2) -> Tuple[bool, float, str]:
    """
    Проверка баланса классов
    
    Args:
        labels: Серия с метками [0, 1]
        min_balance: Минимальный допустимый баланс
    
    Returns:
        (valid, balance, message):
            valid: True если баланс OK
            balance: Фактический баланс минорного класса
            message: Описание
    
    Example:
        >>> labels = pd.Series([1, 1, 0, 1, 0, 1])
        >>> valid, balance, msg = validate_class_balance(labels, 0.2)
        >>> print(f"Valid: {valid}, Balance: {balance:.2f}")
        Valid: True, Balance: 0.33
    """
    if len(labels) == 0:
        return False, 0.0, "Пустая выборка"
    
    # Подсчет классов
    unique_labels = labels.unique()
    
    if len(unique_labels) < 2:
        return False, 0.0, f"Только один класс: {unique_labels[0]}"
    
    # Баланс = доля минорного класса
    counts = labels.value_counts()
    minority_class_count = counts.min()
    balance = minority_class_count / len(labels)
    
    if balance < min_balance:
        return False, balance, (
            f"Дисбаланс классов: {balance:.3f} < {min_balance} "
            f"(классов: {counts.to_dict()})"
        )
    
    return True, balance, f"OK (баланс: {balance:.3f})"


def validate_sample_size(data: pd.DataFrame,
                        min_samples: int = 100) -> Tuple[bool, str]:
    """
    Проверка достаточности размера выборки
    
    Args:
        data: DataFrame с данными
        min_samples: Минимальное количество примеров
    
    Returns:
        (valid, message)
    """
    n_samples = len(data)
    
    if n_samples < min_samples:
        return False, (
            f"Недостаточно данных: {n_samples} < {min_samples}"
        )
    
    return True, f"OK ({n_samples} примеров)"


def validate_features(features: pd.DataFrame) -> Tuple[bool, str]:
    """
    Валидация признаков
    
    Проверки:
        - Отсутствие NaN
        - Отсутствие inf
        - Отсутствие константных признаков
        - Достаточная вариативность
    
    Args:
        features: DataFrame с признаками
    
    Returns:
        (valid, message)
    """
    # NaN проверка
    if features.isna().any().any():
        nan_cols = features.columns[features.isna().any()].tolist()
        return False, f"NaN в признаках: {nan_cols}"
    
    # Inf проверка
    numeric_cols = features.select_dtypes(include=[np.number])
    if np.isinf(numeric_cols).any().any():
        inf_cols = numeric_cols.columns[np.isinf(numeric_cols).any()].tolist()
        return False, f"Inf в признаках: {inf_cols}"
    
    # Константные признаки
    constant_cols = [
        col for col in numeric_cols.columns
        if numeric_cols[col].nunique() == 1
    ]
    
    if constant_cols:
        return False, f"Константные признаки: {constant_cols}"
    
    # Низкая вариативность (std < 0.01)
    low_variance_cols = [
        col for col in numeric_cols.columns
        if numeric_cols[col].std() < 0.01
    ]
    
    if low_variance_cols:
        return False, f"Низкая вариативность: {low_variance_cols}"
    
    return True, f"OK ({len(features.columns)} признаков)"


def validate_cluster_quality(features: np.ndarray,
                            cluster_labels: np.ndarray,
                            min_score: float = 0.3) -> Tuple[bool, float, str]:
    """
    Оценка качества кластеризации
    
    Использует Silhouette Score:
        - Score ≈ 1: Отличная кластеризация
        - Score ≈ 0.5: Приемлемая кластеризация
        - Score ≈ 0: Случайная кластеризация
        - Score < 0: Плохая кластеризация
    
    Args:
        features: Признаки для кластеризации
        cluster_labels: Метки кластеров
        min_score: Минимальный допустимый score
    
    Returns:
        (valid, score, message)
    """
    if len(np.unique(cluster_labels)) < 2:
        return False, 0.0, "Менее 2 кластеров"
    
    try:
        score = silhouette_score(features, cluster_labels)
        
        if score < min_score:
            return False, score, (
                f"Низкое качество кластеризации: {score:.3f} < {min_score}"
            )
        
        return True, score, f"OK (Silhouette: {score:.3f})"
        
    except Exception as e:
        return False, 0.0, f"Ошибка расчета: {e}"


def validate_cluster_sizes(cluster_labels: np.ndarray,
                          min_cluster_size: int = 100) -> Tuple[bool, dict, str]:
    """
    Проверка размеров кластеров
    
    Args:
        cluster_labels: Метки кластеров
        min_cluster_size: Минимальный размер кластера
    
    Returns:
        (valid, sizes, message):
            valid: True если все кластеры >= min_size
            sizes: {cluster_id: size}
            message: Описание
    """
    unique_clusters = np.unique(cluster_labels)
    sizes = {
        int(cluster): int((cluster_labels == cluster).sum())
        for cluster in unique_clusters
    }
    
    small_clusters = {
        k: v for k, v in sizes.items()
        if v < min_cluster_size
    }
    
    if small_clusters:
        return False, sizes, (
            f"Маленькие кластеры: {small_clusters} "
            f"(минимум {min_cluster_size})"
        )
    
    return True, sizes, f"OK (кластеров: {len(sizes)})"


def validate_model(model,
                  X_test: pd.DataFrame,
                  y_test: pd.Series,
                  min_accuracy: float = 0.6) -> Tuple[bool, float, str]:
    """
    Валидация обученной модели на тестовых данных
    
    Args:
        model: Обученная модель с методом score()
        X_test: Тестовые признаки
        y_test: Тестовые метки
        min_accuracy: Минимальная точность
    
    Returns:
        (valid, accuracy, message)
    """
    try:
        accuracy = model.score(X_test, y_test)
        
        if accuracy < min_accuracy:
            return False, accuracy, (
                f"Низкая точность: {accuracy:.3f} < {min_accuracy}"
            )
        
        return True, accuracy, f"OK (accuracy: {accuracy:.3f})"
        
    except Exception as e:
        return False, 0.0, f"Ошибка валидации: {e}"


def validate_predictions(predictions: np.ndarray,
                        y_true: np.ndarray) -> Tuple[bool, dict, str]:
    """
    Валидация предсказаний модели
    
    Проверки:
        - Корректный формат
        - Наличие обоих классов
        - Согласованность размеров
    
    Args:
        predictions: Предсказания модели
        y_true: Истинные метки
    
    Returns:
        (valid, metrics, message)
    """
    # Проверка размеров
    if len(predictions) != len(y_true):
        return False, {}, (
            f"Несоответствие размеров: "
            f"{len(predictions)} != {len(y_true)}"
        )
    
    # Проверка формата
    unique_preds = np.unique(predictions)
    if len(unique_preds) > 2 or not all(p in [0, 1] for p in unique_preds):
        return False, {}, f"Некорректные предсказания: {unique_preds}"
    
    # Базовые метрики
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    
    metrics = {
        'accuracy': accuracy_score(y_true, predictions),
        'f1_score': f1_score(y_true, predictions),
        'confusion_matrix': confusion_matrix(y_true, predictions).tolist()
    }
    
    return True, metrics, f"OK (acc: {metrics['accuracy']:.3f})"


# === КОМПЛЕКСНАЯ ВАЛИДАЦИЯ ===

def validate_training_data(data: pd.DataFrame,
                          config: dict) -> Tuple[bool, list]:
    """
    Комплексная валидация данных перед обучением
    
    Проверяет все критические аспекты:
        - Размер выборки
        - Баланс классов
        - Качество признаков
        - Наличие требуемых колонок
    
    Args:
        data: Данные для обучения
        config: Конфигурация с минимальными требованиями
    
    Returns:
        (valid, errors): (True если все OK, список ошибок)
    """
    errors = []
    
    # Проверка колонок
    required_cols = ['close', 'labels']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        errors.append(f"Отсутствуют колонки: {missing_cols}")
        return False, errors
    
    # Размер выборки
    min_samples = config.get('validation', {}).get('min_samples_per_class', 100)
    valid, msg = validate_sample_size(data, min_samples)
    if not valid:
        errors.append(f"Размер выборки: {msg}")
    
    # Баланс классов
    min_balance = config.get('validation', {}).get('min_class_balance', 0.2)
    valid, balance, msg = validate_class_balance(data['labels'], min_balance)
    if not valid:
        errors.append(f"Баланс классов: {msg}")
    
    # Признаки (если есть)
    feat_cols = [col for col in data.columns 
                 if col.startswith('feat_') or col.startswith('meta_')]
    if feat_cols:
        valid, msg = validate_features(data[feat_cols])
        if not valid:
            errors.append(f"Признаки: {msg}")
    
    return len(errors) == 0, errors


def print_validation_report(data: pd.DataFrame,
                           cluster_labels: Optional[np.ndarray] = None,
                           config: Optional[dict] = None) -> None:
    """
    Вывод детального отчета валидации
    
    Args:
        data: Данные для проверки
        cluster_labels: Метки кластеров (опционально)
        config: Конфигурация (опционально)
    """
    print(f"\n{'='*60}")
    print(f"  VALIDATION REPORT")
    print(f"{'='*60}")
    
    # Размер данных
    print(f"\nРазмер данных:")
    print(f"  • Строк: {len(data)}")
    print(f"  • Колонок: {len(data.columns)}")
    
    # Баланс классов
    if 'labels' in data.columns:
        valid, balance, msg = validate_class_balance(data['labels'])
        status = "✓" if valid else "✗"
        print(f"\nБаланс классов: {status}")
        print(f"  {msg}")
    
    # Признаки
    feat_cols = [col for col in data.columns 
                 if col.startswith('feat_') or col.startswith('meta_')]
    if feat_cols:
        valid, msg = validate_features(data[feat_cols])
        status = "✓" if valid else "✗"
        print(f"\nПризнаки: {status}")
        print(f"  {msg}")
    
    # Кластеры
    if cluster_labels is not None:
        valid, sizes, msg = validate_cluster_sizes(cluster_labels)
        status = "✓" if valid else "✗"
        print(f"\nКластеры: {status}")
        print(f"  {msg}")
        print(f"  Размеры: {sizes}")
    
    print(f"{'='*60}\n")