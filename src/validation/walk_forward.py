import numpy as np
import pandas as pd
import random
from typing import List, Dict, Callable, Tuple, Optional
from dataclasses import dataclass

@dataclass
class WalkForwardConfig:
    """Конфигурация для Walk-Forward валидации"""
    n_is_blocks: int = 10       # Количество блоков для обучения (In-Sample)
    n_oos_blocks: int = 5       # Количество блоков для теста (Out-of-Sample)
    min_r2: float = 0.01        # Минимальный R2 для прохождения этапа
    max_drawdown: float = 0.05  # Максимальная просадка (если используется)
    noise_level: float = 0.002  # Уровень шума при рестарте (0.2%)
    l2_increment: float = 1.0   # Шаг увеличения регуляризации при провале
    max_retries: int = 3        # Максимальное количество попыток рестарта

def create_walk_forward_splits(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Разделение данных на IS (60%), OOS (20%) и OOT (20%).
    Строгое разделение по времени без перемешивания.
    """
    n = len(data)
    is_end = int(n * 0.6)
    oos_end = int(n * 0.8)
    
    return data.iloc[:is_end].copy(), data.iloc[is_end:oos_end].copy(), data.iloc[oos_end:].copy()

class WalkForwardValidator:
    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.is_blocks = []
        self.oos_blocks = []
        self.retries = 0
        
    def split_data(self, is_data: pd.DataFrame, oos_data: pd.DataFrame):
        """Разбиение данных на временные блоки"""
        # In-Sample блоки (последовательные)
        # Если данных мало, берем минимум 100 баров на блок
        is_len = len(is_data)
        is_chunk = max(100, is_len // self.config.n_is_blocks)
        
        self.is_blocks = []
        for i in range(self.config.n_is_blocks):
            start = i * is_chunk
            # Последний блок забирает остаток
            end = (i + 1) * is_chunk if i < self.config.n_is_blocks - 1 else is_len
            self.is_blocks.append(is_data.iloc[start:end].copy())
        
        # Out-of-Sample блоки (для случайной валидации)
        oos_len = len(oos_data)
        oos_chunk = max(50, oos_len // self.config.n_oos_blocks)
        
        self.oos_blocks = []
        for i in range(self.config.n_oos_blocks):
            start = i * oos_chunk
            end = (i + 1) * oos_chunk if i < self.config.n_oos_blocks - 1 else oos_len
            self.oos_blocks.append(oos_data.iloc[start:end].copy())
        
    def validate_sequential(self, train_fn: Callable, eval_fn: Callable, params: Dict) -> Tuple[bool, object]:
        """
        Основной цикл валидации.
        Возвращает: (Success, Model)
        """
        # Случайный порядок OOS (Этап 2 - проверка робастности)
        # Мы работаем с копией списка, чтобы при рестарте снова перемешать
        current_oos_blocks = self.oos_blocks[:]
        random.shuffle(current_oos_blocks)
        
        # Начинаем с накопленных IS данных
        current_train_data = pd.concat(self.is_blocks)
        
        model = None
        
        print(f"\n🚀 Запуск Walk-Forward (Попытка {self.retries + 1}/{self.config.max_retries + 1})")
        
        for i, oos_block in enumerate(current_oos_blocks):
            print(f"  📍 Checkpoint {i+1}/{len(current_oos_blocks)} (Size: {len(oos_block)})")
            
            # 1. Обучение
            try:
                model = train_fn(current_train_data, params)
            except Exception as e:
                print(f"    ⚠️ Ошибка обучения: {e}")
                return self._restart(train_fn, eval_fn, params)

            if model is None:
                print("    ⚠️ Модель не обучилась (None)")
                return self._restart(train_fn, eval_fn, params)
            
            # 2. Валидация на текущем OOS
            try:
                metrics = eval_fn(model, oos_block)
                r2 = metrics.get('r2', -999)
            except Exception as e:
                print(f"    ⚠️ Ошибка валидации: {e}")
                r2 = -999
            
            # 3. Проверка критериев
            if r2 < self.config.min_r2:
                print(f"    ❌ FAIL: R2 {r2:.4f} < {self.config.min_r2}")
                # ПРОВАЛ -> РЕСТАРТ
                return self._restart(train_fn, eval_fn, params)
            
            print(f"    ✅ PASS: R2 {r2:.4f}")
            
            # 4. Успех -> Добавляем этот OOS в обучение для следующего шага
            current_train_data = pd.concat([current_train_data, oos_block])
            
        print("🎉 Все чекпоинты пройдены!")
        return True, model
        
    def _restart(self, train_fn, eval_fn, params):
        """Механизм сброса и усложнения задачи"""
        if self.retries >= self.config.max_retries:
            print("💀 Превышен лимит попыток. Система не нашла решение.")
            return False, None
            
        self.retries += 1
        print(f"\n🔄 РЕСТАРТ СИСТЕМЫ (Попытка {self.retries + 1})")
        print("   -> Добавление шума в In-Sample данные")
        print("   -> Усиление L2 регуляризации")
        
        # 1. Data Augmentation (Шум)
        # Мы модифицируем IS блоки "на месте" для следующих попыток
        for block in self.is_blocks:
            noise = np.random.uniform(
                -self.config.noise_level, 
                self.config.noise_level, 
                len(block)
            )
            # Шум накладываем на Close
            block['close'] = block['close'] * (1 + noise)
            
        # 2. Усиление регуляризации
        current_l2 = params.get('l2_leaf_reg', 3)
        params['l2_leaf_reg'] = current_l2 + self.config.l2_increment
        
        # Рекурсивный перезапуск
        return self.validate_sequential(train_fn, eval_fn, params)