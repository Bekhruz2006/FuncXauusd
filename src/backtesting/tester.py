"""
Быстрый бэктестер для торговых моделей

Реализован на Numba JIT для максимальной производительности.
Симулирует реальные торговые условия с:
    - Stop-Loss и Take-Profit
    - Маркапом (спред/комиссия)
    - Закрытием по обратным сигналам
    - Фильтрацией через meta-модель
"""

import numpy as np
import pandas as pd
from numba import jit
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, List


@jit(nopython=True)
def process_data_one_direction(
    close: np.ndarray,
    labels: np.ndarray,
    metalabels: np.ndarray,
    stop: float,
    take: float,
    markup: float,
    forward: int,
    backward: int,
    direction: str
) -> tuple:
    """
    Обработка торговых сигналов (Numba-оптимизирован)
    
    Логика:
        1. Открытие позиции по сигналу (labels > 0.5) если meta разрешает
        2. Закрытие по Stop-Loss или Take-Profit
        3. Закрытие по обратному сигналу
        4. Учет маркапа (спред/комиссия) на каждую сделку
    
    Args:
        close: Цены закрытия
        labels: Основные торговые сигналы [0, 1]
        metalabels: Фильтр мета-модели [0, 1]
        stop: Stop-Loss в пунктах
        take: Take-Profit в пунктах
        markup: Маркап (спред + комиссия) в пунктах
        forward: Индекс границы forward периода
        backward: Индекс границы backward периода
        direction: 'buy' или 'sell'
    
    Returns:
        (report, chart, line_f, line_b):
            report: Кумулятивная прибыль по сделкам
            chart: Кумулятивная прибыль без учета обратных позиций
            line_f: Индекс forward границы
            line_b: Индекс backward границы
    """
    last_deal = 2  # 2 = нет позиции, 1 = в позиции
    last_price = 0.0
    report = [0.0]
    chart = [0.0]
    line_f = 0
    line_b = 0
    
    for i in range(len(close)):
        # Запись индексов границ
        if i <= forward:
            line_f = len(report)
        if i <= backward:
            line_b = len(report)
        
        pred = labels[i]
        pr = close[i]
        pred_meta = metalabels[i]  # 1 = разрешить торговлю
        
        # === ОТКРЫТИЕ ПОЗИЦИИ ===
        if last_deal == 2 and pred_meta == 1:
            last_price = pr
            # Открываем только если сигнал > 0.5
            last_deal = 2 if pred < 0.5 else 1
            continue
        
        # === ЗАКРЫТИЕ ПО SL/TP (BUY) ===
        if last_deal == 1 and direction == 'buy':
            profit_pips = pr - last_price
            loss_pips = last_price - pr
            
            # Take-Profit достигнут
            if (-markup + profit_pips >= take):
                last_deal = 2
                profit = -markup + profit_pips
                report.append(report[-1] + profit)
                chart.append(chart[-1] + profit)
                continue
            
            # Stop-Loss достигнут
            if (-markup + loss_pips >= stop):
                last_deal = 2
                profit = -markup + profit_pips
                report.append(report[-1] + profit)
                chart.append(chart[-1] + profit)
                continue
        
        # === ЗАКРЫТИЕ ПО SL/TP (SELL) ===
        if last_deal == 1 and direction == 'sell':
            profit_pips = last_price - pr
            loss_pips = pr - last_price
            
            # Take-Profit достигнут
            if (-markup + profit_pips >= take):
                last_deal = 2
                profit = -markup + profit_pips
                report.append(report[-1] + profit)
                chart.append(chart[-1] + (pr - last_price))
                continue
            
            # Stop-Loss достигнут
            if (-markup + loss_pips >= stop):
                last_deal = 2
                profit = -markup + profit_pips
                report.append(report[-1] + profit)
                chart.append(chart[-1] + (pr - last_price))
                continue
        
        # === ЗАКРЫТИЕ ПО ОБРАТНОМУ СИГНАЛУ ===
        if last_deal == 1 and pred < 0.5:
            last_deal = 2
            
            if direction == 'buy':
                profit = -markup + (pr - last_price)
                report.append(report[-1] + profit)
                chart.append(chart[-1] + profit)
            else:  # sell
                profit = -markup + (last_price - pr)
                report.append(report[-1] + profit)
                chart.append(chart[-1] + (pr - last_price))
            
            continue
    
    return (
        np.array(report),
        np.array(chart),
        line_f,
        line_b
    )


def tester_one_direction(
    dataset: pd.DataFrame,
    stop: float,
    take: float,
    forward: datetime,
    backward: datetime,
    markup: float,
    direction: str,
    plt_show: bool = False
) -> float:
    """
    Тестирование модели на исторических данных
    
    Args:
        dataset: DataFrame с 'close', 'labels', 'meta_labels'
        stop: Stop-Loss в пунктах
        take: Take-Profit в пунктах
        forward: Граница In-Sample/Out-of-Sample
        backward: Граница начала данных
        markup: Маркап (спред + комиссия)
        direction: 'buy' или 'sell'
        plt_show: Показать график результатов
    
    Returns:
        float: R² score (качество линейной аппроксимации equity)
    
    Метрика R²:
        - R² ≈ 1.0: Идеальная линейная прибыль
        - R² ≈ 0.5: Умеренная прибыль с просадками
        - R² ≤ 0: Отрицательный тренд или хаотичная торговля
    """
    # Поиск индексов границ
    forw = dataset.index.get_indexer([forward], method='nearest')[0]
    backw = dataset.index.get_indexer([backward], method='nearest')[0]
    
    # Извлечение данных
    close = dataset['close'].to_numpy()
    labels = dataset['labels'].to_numpy()
    metalabels = dataset['meta_labels'].to_numpy()
    
    # Запуск симуляции
    report, chart, line_f, line_b = process_data_one_direction(
        close, labels, metalabels,
        stop, take, markup,
        forw, backw, direction
    )
    
    # Расчет R² через линейную регрессию
    y = report.reshape(-1, 1)
    X = np.arange(len(report)).reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(X, y)
    
    # Определение знака (прибыль или убыток)
    l = 1 if lr.coef_[0][0] >= 0 else -1
    r2_score = lr.score(X, y) * l
    
    # Визуализация
    if plt_show:
        plt.figure(figsize=(14, 7))
        
        plt.plot(report, label='Equity Curve', linewidth=2, color='#2E86AB')
        plt.plot(chart, label='Buy & Hold', linewidth=1.5, 
                alpha=0.7, linestyle='--', color='#A23B72')
        
        plt.axvline(x=line_f, color='purple', ls=':', lw=2, 
                   label='OOS Start')
        plt.axvline(x=line_b, color='red', ls=':', lw=2, 
                   label='Training Start')
        
        plt.plot(lr.predict(X), label='Linear Fit', 
                linewidth=2, linestyle='--', color='#F18F01')
        
        plt.title(
            f"Strategy Performance | R² = {r2_score:.4f} | "
            f"Direction: {direction.upper()}\n"
            f"Stop: {stop} | Take: {take} | Markup: {markup}",
            fontsize=14, fontweight='bold'
        )
        plt.xlabel("Number of Trades", fontsize=12)
        plt.ylabel("Cumulative Profit (pips)", fontsize=12)
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return r2_score


def test_model_one_direction(
    dataset: pd.DataFrame,
    result: List,
    config: dict,
    plt: bool = False
) -> float:
    """
    Тестирование обученной модели
    
    Args:
        dataset: DataFrame с признаками
        result: [main_model, meta_model] - обученные CatBoost модели
        config: Конфигурация с параметрами торговли
        plt: Показать визуализацию
    
    Returns:
        float: R² score
    """
    ext_dataset = dataset.copy()
    
    # Извлечение признаков (все кроме 'close')
    X = ext_dataset[ext_dataset.columns[1:]]
    
    # Прогнозы моделей
    ext_dataset['labels'] = result[0].predict_proba(X)[:, 1]
    ext_dataset['meta_labels'] = result[1].predict_proba(X)[:, 1]
    
    # Бинаризация (порог 0.5)
    ext_dataset['labels'] = ext_dataset['labels'].apply(
        lambda x: 0.0 if x < 0.5 else 1.0
    )
    ext_dataset['meta_labels'] = ext_dataset['meta_labels'].apply(
        lambda x: 0.0 if x < 0.5 else 1.0
    )
    
    # Запуск тестера
    return tester_one_direction(
        ext_dataset,
        stop=config['trading']['risk']['stop_loss'],
        take=config['trading']['risk']['take_profit'],
        forward=datetime.fromisoformat(config['data']['forward']),
        backward=datetime.fromisoformat(config['data']['backward']),
        markup=config['trading']['labeling']['markup'],
        direction=config['trading']['direction'],
        plt_show=plt
    )


# === ДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ ===

def calculate_advanced_metrics(report: np.ndarray,
                               chart: np.ndarray) -> dict:
    """
    Расчет расширенных торговых метрик
    
    Args:
        report: Equity curve
        chart: Buy & Hold curve
    
    Returns:
        dict: Метрики (Sharpe, Max DD, Win Rate и т.д.)
    """
    # Прибыль по сделкам
    returns = np.diff(report)
    
    # Фильтруем нулевые (нет сделки)
    actual_trades = returns[returns != 0]
    
    if len(actual_trades) == 0:
        return {
            'total_trades': 0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0
        }
    
    # Sharpe Ratio
    sharpe = (
        np.mean(actual_trades) / np.std(actual_trades) 
        if np.std(actual_trades) > 0 else 0.0
    )
    
    # Maximum Drawdown
    cummax = np.maximum.accumulate(report)
    drawdown = (report - cummax) / np.maximum(cummax, 1)
    max_dd = abs(drawdown.min())
    
    # Win Rate
    winning_trades = actual_trades[actual_trades > 0]
    win_rate = len(winning_trades) / len(actual_trades)
    
    # Profit Factor
    gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
    losing_trades = actual_trades[actual_trades < 0]
    gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    return {
        'total_trades': len(actual_trades),
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': np.mean(winning_trades) if len(winning_trades) > 0 else 0,
        'avg_loss': np.mean(losing_trades) if len(losing_trades) > 0 else 0
    }


def print_backtest_report(report: np.ndarray,
                         chart: np.ndarray,
                         r2_score: float) -> None:
    """
    Вывод детального отчета по бэктесту
    """
    metrics = calculate_advanced_metrics(report, chart)
    
    print(f"\n{'='*60}")
    print(f"  BACKTEST REPORT")
    print(f"{'='*60}")
    print(f"\nОсновные метрики:")
    print(f"  • R² Score:         {r2_score:>8.4f}")
    print(f"  • Total Trades:     {metrics['total_trades']:>8}")
    print(f"  • Win Rate:         {metrics['win_rate']:>7.1%}")
    print(f"  • Profit Factor:    {metrics['profit_factor']:>8.2f}")
    
    print(f"\nРиски:")
    print(f"  • Max Drawdown:     {metrics['max_drawdown']:>7.1%}")
    print(f"  • Sharpe Ratio:     {metrics['sharpe_ratio']:>8.2f}")
    
    print(f"\nСделки:")
    print(f"  • Avg Win:          {metrics['avg_win']:>8.2f} pips")
    print(f"  • Avg Loss:         {metrics['avg_loss']:>8.2f} pips")
    
    print(f"\nEquity:")
    print(f"  • Final Profit:     {report[-1]:>8.2f} pips")
    print(f"  • Buy & Hold:       {chart[-1]:>8.2f} pips")
    print(f"{'='*60}\n")