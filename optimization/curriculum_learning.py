"""
Curriculum Learning с контрастными парами прототип-контрпример.
Постепенное усложнение обучающих задач от Stage 0 до Stage 4.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy.stats import mannwhitneyu
import random

from config.hyperparameters import HYPERPARAMS
from config.market_regimes import RegimeType, RegimeDefinitions
from utils.logger import LOGGER


class StageType(Enum):
    """Типы этапов обучения"""
    META_CALIBRATION = 0
    SIMPLE_PATTERNS = 1
    MEDIUM_COMPLEXITY = 2
    HIGH_COMPLEXITY = 3
    REGIME_ABSTRACTION = 4


@dataclass
class StageConfig:
    """Конфигурация этапа обучения"""
    stage_id: int
    stage_type: StageType
    duration_bars: int
    num_checkpoints: int
    min_required_checkpoints: int
    
    complexity_level: float
    contrastive_pairs: bool
    mann_whitney_pvalue: float
    
    allowed_regimes: List[str] = field(default_factory=list)
    focus_patterns: List[str] = field(default_factory=list)
    
    success_threshold: float = 0.6
    failure_penalty: float = 0.8


@dataclass
class Checkpoint:
    """Контрольная точка с данными для оценки"""
    checkpoint_id: str
    stage_id: int
    regime_type: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    data_segment: pd.DataFrame
    
    is_prototype: bool = True
    is_contrastive: bool = False
    paired_checkpoint_id: Optional[str] = None
    
    difficulty_score: float = 0.0
    metadata: Dict = field(default_factory=dict)


class CurriculumLearningScheduler:
    """Планировщик curriculum learning с прогрессивным усложнением"""
    
    def __init__(self,
                 data_dict: Dict[str, pd.DataFrame],
                 regime_labels: Optional[Dict[str, pd.DataFrame]] = None,
                 use_synthetic_stage0: bool = True):
        
        self.data_dict = data_dict
        self.regime_labels = regime_labels
        self.use_synthetic_stage0 = use_synthetic_stage0
        
        self.stages: List[StageConfig] = []
        self.checkpoints: Dict[int, List[Checkpoint]] = {}
        self.checkpoint_results: Dict[str, Dict] = {}
        
        self.current_stage = 0
        self.completed_stages = set()
        
        self._initialize_stages()
        
        LOGGER.info(f"Инициализирован CurriculumLearner: {len(self.stages)} этапов")
    
    def _initialize_stages(self) -> None:
        """Инициализация всех этапов обучения"""
        try:
            # Stage 0: Meta-calibration на патологических сценариях
            stage0 = StageConfig(
                stage_id=0,
                stage_type=StageType.META_CALIBRATION,
                duration_bars=HYPERPARAMS.curriculum.stage_0_duration,
                num_checkpoints=10,
                min_required_checkpoints=8,
                complexity_level=0.3,
                contrastive_pairs=False,
                mann_whitney_pvalue=0.1,
                focus_patterns=['sharp_gaps', 'false_breakouts', 'whipsaw']
            )
            
            # Stage 1: Простые паттерны
            stage1 = StageConfig(
                stage_id=1,
                stage_type=StageType.SIMPLE_PATTERNS,
                duration_bars=HYPERPARAMS.curriculum.stage_1_duration,
                num_checkpoints=HYPERPARAMS.curriculum.checkpoints_per_stage,
                min_required_checkpoints=12,
                complexity_level=0.5,
                contrastive_pairs=True,
                mann_whitney_pvalue=HYPERPARAMS.curriculum.mann_whitney_pvalue,
                allowed_regimes=[RegimeType.TREND_UP.value, RegimeType.RANGING.value],
                focus_patterns=['clear_trends', 'support_resistance']
            )
            
            # Stage 2: Средняя сложность
            stage2 = StageConfig(
                stage_id=2,
                stage_type=StageType.MEDIUM_COMPLEXITY,
                duration_bars=HYPERPARAMS.curriculum.stage_2_duration,
                num_checkpoints=HYPERPARAMS.curriculum.checkpoints_per_stage,
                min_required_checkpoints=12,
                complexity_level=0.7,
                contrastive_pairs=True,
                mann_whitney_pvalue=HYPERPARAMS.curriculum.mann_whitney_pvalue,
                allowed_regimes=[r.value for r in RegimeType],
                focus_patterns=['breakouts', 'pullbacks', 'consolidation']
            )
            
            # Stage 3: Высокая сложность
            stage3 = StageConfig(
                stage_id=3,
                stage_type=StageType.HIGH_COMPLEXITY,
                duration_bars=HYPERPARAMS.curriculum.stage_3_duration,
                num_checkpoints=HYPERPARAMS.curriculum.checkpoints_per_stage,
                min_required_checkpoints=10,
                complexity_level=0.9,
                contrastive_pairs=True,
                mann_whitney_pvalue=HYPERPARAMS.curriculum.mann_whitney_pvalue,
                allowed_regimes=[r.value for r in RegimeType],
                focus_patterns=['volatile_markets', 'regime_transitions', 'mixed_signals']
            )
            
            # Stage 4: Режимная абстракция
            stage4 = StageConfig(
                stage_id=4,
                stage_type=StageType.REGIME_ABSTRACTION,
                duration_bars=HYPERPARAMS.curriculum.stage_4_duration,
                num_checkpoints=HYPERPARAMS.curriculum.checkpoints_per_stage,
                min_required_checkpoints=10,
                complexity_level=1.0,
                contrastive_pairs=False,
                mann_whitney_pvalue=0.05,
                allowed_regimes=[r.value for r in RegimeType],
                focus_patterns=['all']
            )
            
            self.stages = [stage0, stage1, stage2, stage3, stage4]
            
            LOGGER.info(f"Инициализировано {len(self.stages)} этапов curriculum learning")
            
        except Exception as e:
            LOGGER.error(f"Ошибка инициализации этапов: {e}", exc_info=True)
            raise
    
    def generate_checkpoints_for_stage(self, 
                                       stage_id: int,
                                       base_timeframe: str = 'H1') -> List[Checkpoint]:
        """
        Генерация чекпоинтов для этапа.
        
        Args:
            stage_id: ID этапа
            base_timeframe: Базовый таймфрейм
        
        Returns:
            Список сгенерированных чекпоинтов
        """
        try:
            if stage_id >= len(self.stages):
                raise ValueError(f"Некорректный stage_id: {stage_id}")
            
            stage = self.stages[stage_id]
            
            LOGGER.info(f"Генерация {stage.num_checkpoints} чекпоинтов для Stage {stage_id}")
            
            if stage_id == 0 and self.use_synthetic_stage0:
                checkpoints = self._generate_synthetic_checkpoints(stage)
            else:
                checkpoints = self._generate_historical_checkpoints(stage, base_timeframe)
            
            self.checkpoints[stage_id] = checkpoints
            
            LOGGER.info(f"Stage {stage_id}: сгенерировано {len(checkpoints)} чекпоинтов")
            
            return checkpoints
            
        except Exception as e:
            LOGGER.error(f"Ошибка генерации чекпоинтов для Stage {stage_id}: {e}", exc_info=True)
            return []
    
    def _generate_synthetic_checkpoints(self, stage: StageConfig) -> List[Checkpoint]:
        """Генерация синтетических чекпоинтов для Stage 0"""
        try:
            from data.synthetic_generator import SyntheticDataGenerator
            
            generator = SyntheticDataGenerator()
            pathological_scenarios = generator.generate_pathological_scenarios(
                base_price=2000.0,
                num_bars=stage.duration_bars
            )
            
            checkpoints = []
            
            for i, (scenario_name, scenario_data) in enumerate(pathological_scenarios.items()):
                try:
                    checkpoint = Checkpoint(
                        checkpoint_id=f"stage0_synthetic_{i}_{scenario_name}",
                        stage_id=stage.stage_id,
                        regime_type="pathological",
                        start_date=scenario_data.index[0],
                        end_date=scenario_data.index[-1],
                        data_segment=scenario_data,
                        is_prototype=True,
                        is_contrastive=False,
                        difficulty_score=1.0,
                        metadata={'scenario_type': scenario_name}
                    )
                    
                    checkpoints.append(checkpoint)
                    
                except Exception as e:
                    LOGGER.error(f"Ошибка создания synthetic checkpoint {scenario_name}: {e}")
                    continue
            
            return checkpoints
            
        except Exception as e:
            LOGGER.error(f"Ошибка генерации synthetic checkpoints: {e}")
            return []
    
    def _generate_historical_checkpoints(self, 
                                         stage: StageConfig,
                                         base_timeframe: str) -> List[Checkpoint]:
        """Генерация чекпоинтов из исторических данных"""
        try:
            if base_timeframe not in self.data_dict:
                raise ValueError(f"Таймфрейм {base_timeframe} не найден")
            
            df = self.data_dict[base_timeframe]
            
            if len(df) < stage.duration_bars * stage.num_checkpoints:
                LOGGER.warning(f"Недостаточно данных для Stage {stage.stage_id}")
            
            # Фильтрация по режимам если есть разметка
            if self.regime_labels and base_timeframe in self.regime_labels:
                regime_df = self.regime_labels[base_timeframe]
                available_data = self._filter_by_regimes(df, regime_df, stage.allowed_regimes)
            else:
                available_data = df
            
            checkpoints = []
            
            if stage.contrastive_pairs:
                checkpoints = self._generate_contrastive_pairs(
                    available_data, stage, base_timeframe
                )
            else:
                checkpoints = self._generate_standard_checkpoints(
                    available_data, stage, base_timeframe
                )
            
            return checkpoints
            
        except Exception as e:
            LOGGER.error(f"Ошибка генерации historical checkpoints: {e}")
            return []
    
    def _filter_by_regimes(self,
                          df: pd.DataFrame,
                          regime_df: pd.DataFrame,
                          allowed_regimes: List[str]) -> pd.DataFrame:
        """Фильтрация данных по разрешенным режимам"""
        try:
            if not allowed_regimes:
                return df
            
            if 'regime_type' not in regime_df.columns:
                return df
            
            mask = regime_df['regime_type'].isin(allowed_regimes)
            filtered_indices = regime_df[mask].index
            
            filtered_df = df.loc[df.index.isin(filtered_indices)]
            
            LOGGER.debug(f"Отфильтровано {len(filtered_df)} баров по режимам {allowed_regimes}")
            
            return filtered_df
            
        except Exception as e:
            LOGGER.error(f"Ошибка фильтрации по режимам: {e}")
            return df
    
    def _generate_contrastive_pairs(self,
                                    df: pd.DataFrame,
                                    stage: StageConfig,
                                    base_timeframe: str) -> List[Checkpoint]:
        """Генерация контрастных пар прототип-контрпример"""
        try:
            checkpoints = []
            num_pairs = stage.num_checkpoints // 2
            
            for pair_idx in range(num_pairs):
                try:
                    # Прототип: явный паттерн
                    prototype = self._select_prototype_segment(
                        df, stage.duration_bars, stage.focus_patterns
                    )
                    
                    if prototype is None:
                        continue
                    
                    prototype_checkpoint = Checkpoint(
                        checkpoint_id=f"stage{stage.stage_id}_pair{pair_idx}_prototype",
                        stage_id=stage.stage_id,
                        regime_type=self._detect_segment_regime(prototype),
                        start_date=prototype.index[0],
                        end_date=prototype.index[-1],
                        data_segment=prototype,
                        is_prototype=True,
                        is_contrastive=True,
                        paired_checkpoint_id=f"stage{stage.stage_id}_pair{pair_idx}_contrast",
                        difficulty_score=stage.complexity_level * 0.8,
                        metadata={'pattern_type': 'clear'}
                    )
                    
                    # Контрпример: похожий но с ложными сигналами
                    contrast = self._select_contrast_segment(
                        df, prototype, stage.duration_bars
                    )
                    
                    if contrast is None:
                        continue
                    
                    contrast_checkpoint = Checkpoint(
                        checkpoint_id=f"stage{stage.stage_id}_pair{pair_idx}_contrast",
                        stage_id=stage.stage_id,
                        regime_type=self._detect_segment_regime(contrast),
                        start_date=contrast.index[0],
                        end_date=contrast.index[-1],
                        data_segment=contrast,
                        is_prototype=False,
                        is_contrastive=True,
                        paired_checkpoint_id=f"stage{stage.stage_id}_pair{pair_idx}_prototype",
                        difficulty_score=stage.complexity_level * 1.2,
                        metadata={'pattern_type': 'noisy'}
                    )
                    
                    checkpoints.extend([prototype_checkpoint, contrast_checkpoint])
                    
                except Exception as e:
                    LOGGER.error(f"Ошибка создания контрастной пары {pair_idx}: {e}")
                    continue
            
            return checkpoints
            
        except Exception as e:
            LOGGER.error(f"Ошибка генерации contrastive pairs: {e}")
            return []
    
    def _generate_standard_checkpoints(self,
                                       df: pd.DataFrame,
                                       stage: StageConfig,
                                       base_timeframe: str) -> List[Checkpoint]:
        """Генерация стандартных чекпоинтов без пар"""
        try:
            checkpoints = []
            
            available_length = len(df) - stage.duration_bars
            
            if available_length < stage.num_checkpoints:
                LOGGER.warning(f"Недостаточно данных для {stage.num_checkpoints} чекпоинтов")
            
            # Стратифицированная выборка по времени
            if available_length > 0:
                indices = np.linspace(0, available_length, 
                                    min(stage.num_checkpoints, available_length), 
                                    dtype=int)
            else:
                indices = [0]
            
            for i, start_idx in enumerate(indices):
                try:
                    end_idx = min(start_idx + stage.duration_bars, len(df))
                    segment = df.iloc[start_idx:end_idx]
                    
                    if len(segment) < stage.duration_bars * 0.8:
                        continue
                    
                    checkpoint = Checkpoint(
                        checkpoint_id=f"stage{stage.stage_id}_std_{i}",
                        stage_id=stage.stage_id,
                        regime_type=self._detect_segment_regime(segment),
                        start_date=segment.index[0],
                        end_date=segment.index[-1],
                        data_segment=segment,
                        is_prototype=True,
                        is_contrastive=False,
                        difficulty_score=stage.complexity_level,
                        metadata={'index': i}
                    )
                    
                    checkpoints.append(checkpoint)
                    
                except Exception as e:
                    LOGGER.error(f"Ошибка создания standard checkpoint {i}: {e}")
                    continue
            
            return checkpoints
            
        except Exception as e:
            LOGGER.error(f"Ошибка генерации standard checkpoints: {e}")
            return []
    
    def _select_prototype_segment(self,
                                  df: pd.DataFrame,
                                  duration: int,
                                  focus_patterns: List[str]) -> Optional[pd.DataFrame]:
        """Выбор сегмента-прототипа с явным паттерном"""
        try:
            # Вычисляем характеристики всех возможных сегментов
            candidates = []
            
            for i in range(0, len(df) - duration, duration // 4):
                segment = df.iloc[i:i+duration]
                
                if len(segment) < duration:
                    continue
                
                # Оценка "чистоты" паттерна
                clarity_score = self._evaluate_pattern_clarity(segment, focus_patterns)
                
                candidates.append({
                    'start_idx': i,
                    'segment': segment,
                    'clarity': clarity_score
                })
            
            if not candidates:
                return None
            
            # Выбираем сегмент с высокой clarity
            candidates.sort(key=lambda x: x['clarity'], reverse=True)
            
            # Берем из топ-30% случайно для разнообразия
            top_candidates = candidates[:max(1, len(candidates) // 3)]
            selected = random.choice(top_candidates)
            
            return selected['segment']
            
        except Exception as e:
            LOGGER.error(f"Ошибка выбора prototype segment: {e}")
            return None
    
    def _select_contrast_segment(self,
                                 df: pd.DataFrame,
                                 prototype: pd.DataFrame,
                                 duration: int) -> Optional[pd.DataFrame]:
        """Выбор контрастного сегмента (похожий но с шумом)"""
        try:
            prototype_characteristics = self._compute_segment_characteristics(prototype)
            
            candidates = []
            
            for i in range(0, len(df) - duration, duration // 4):
                segment = df.iloc[i:i+duration]
                
                if len(segment) < duration:
                    continue
                
                # Избегаем перекрытия с прототипом
                if segment.index[0] >= prototype.index[0] and segment.index[0] <= prototype.index[-1]:
                    continue
                
                segment_characteristics = self._compute_segment_characteristics(segment)
                
                # Оценка схожести и наличия шума
                similarity = self._compute_similarity(
                    prototype_characteristics, segment_characteristics
                )
                
                noise_level = self._evaluate_noise_level(segment)
                
                # Ищем сегмент с умеренной схожестью но высоким шумом
                contrast_score = similarity * noise_level
                
                candidates.append({
                    'start_idx': i,
                    'segment': segment,
                    'contrast_score': contrast_score
                })
            
            if not candidates:
                return None
            
            # Выбираем с высоким contrast_score
            candidates.sort(key=lambda x: x['contrast_score'], reverse=True)
            
            top_candidates = candidates[:max(1, len(candidates) // 3)]
            selected = random.choice(top_candidates)
            
            return selected['segment']
            
        except Exception as e:
            LOGGER.error(f"Ошибка выбора contrast segment: {e}")
            return None
    
    def _evaluate_pattern_clarity(self, 
                                  segment: pd.DataFrame,
                                  focus_patterns: List[str]) -> float:
        """Оценка чистоты паттерна в сегменте"""
        try:
            clarity_score = 0.0
            
            close = segment['Close']
            returns = close.pct_change()
            
            # Трендовость
            if 'clear_trends' in focus_patterns or 'breakouts' in focus_patterns:
                trend = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
                trend_consistency = 1.0 - returns.std() / (abs(returns.mean()) + 1e-8)
                clarity_score += abs(trend) * trend_consistency
            
            # Поддержка/сопротивление
            if 'support_resistance' in focus_patterns:
                low_touches = (segment['Low'] == segment['Low'].rolling(10).min()).sum()
                high_touches = (segment['High'] == segment['High'].rolling(10).max()).sum()
                clarity_score += (low_touches + high_touches) / len(segment)
            
            # Консолидация
            if 'consolidation' in focus_patterns:
                range_pct = (segment['High'].max() - segment['Low'].min()) / close.mean()
                clarity_score += 1.0 / (range_pct + 1e-8)
            
            return np.clip(clarity_score, 0, 1)
            
        except Exception as e:
            LOGGER.error(f"Ошибка оценки clarity: {e}")
            return 0.5
    
    def _evaluate_noise_level(self, segment: pd.DataFrame) -> float:
        """Оценка уровня шума в сегменте"""
        try:
            returns = segment['Close'].pct_change()
            
            # Волатильность
            volatility = returns.std()
            
            # Число разворотов
            sign_changes = (np.diff(np.sign(returns.dropna())) != 0).sum()
            reversal_rate = sign_changes / len(segment)
            
            # Гэпы
            gaps = abs(segment['Open'] - segment['Close'].shift(1))
            gap_rate = (gaps > returns.std() * 2).sum() / len(segment)
            
            noise_score = (volatility * 10 + reversal_rate + gap_rate) / 3
            
            return np.clip(noise_score, 0, 1)
            
        except Exception as e:
            LOGGER.error(f"Ошибка оценки noise: {e}")
            return 0.5
    
    def _compute_segment_characteristics(self, segment: pd.DataFrame) -> Dict:
        """Вычисление характеристик сегмента"""
        try:
            close = segment['Close']
            returns = close.pct_change()
            
            characteristics = {
                'total_return': (close.iloc[-1] - close.iloc[0]) / close.iloc[0],
                'volatility': returns.std(),
                'max_drawdown': (close.cummax() - close).max() / close.cummax().max(),
                'trend_strength': abs(np.polyfit(range(len(close)), close, 1)[0]),
                'mean_volume': segment['Volume'].mean()
            }
            
            return characteristics
            
        except Exception as e:
            LOGGER.error(f"Ошибка вычисления характеристик: {e}")
            return {}
    
    def _compute_similarity(self, char1: Dict, char2: Dict) -> float:
        """Вычисление схожести двух наборов характеристик"""
        try:
            if not char1 or not char2:
                return 0.0
            
            common_keys = set(char1.keys()) & set(char2.keys())
            
            if not common_keys:
                return 0.0
            
            similarities = []
            
            for key in common_keys:
                val1 = char1[key]
                val2 = char2[key]
                
                if abs(val1) < 1e-8 and abs(val2) < 1e-8:
                    similarities.append(1.0)
                else:
                    diff = abs(val1 - val2) / (abs(val1) + abs(val2) + 1e-8)
                    similarities.append(1.0 - diff)
            
            return np.mean(similarities)
            
        except Exception as e:
            LOGGER.error(f"Ошибка вычисления similarity: {e}")
            return 0.5
    
    def _detect_segment_regime(self, segment: pd.DataFrame) -> str:
        """Определение режима сегмента"""
        try:
            if 'regime_type' in segment.columns:
                regime_counts = segment['regime_type'].value_counts()
                return regime_counts.index[0] if len(regime_counts) > 0 else "unknown"
            
            # Простая эвристика
            close = segment['Close']
            returns = close.pct_change()
            
            trend = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
            volatility = returns.std()
            
            if abs(trend) > 0.05:
                return RegimeType.TREND_UP.value if trend > 0 else RegimeType.TREND_DOWN.value
            elif volatility > returns.rolling(20).std().mean() * 1.5:
                return RegimeType.VOLATILE.value
            else:
                return RegimeType.RANGING.value
                
        except Exception:
            return "unknown"
    
    def validate_agent_on_contrastive_pair(self,
                                          agent,
                                          prototype_checkpoint: Checkpoint,
                                          contrast_checkpoint: Checkpoint,
                                          backtest_fn: Callable) -> Tuple[bool, float, Dict]:
        """
        Валидация агента на контрастной паре.
        
        Returns:
            (passed, p_value, statistics)
        """
        try:
            # Бэктест на прототипе
            prototype_results = backtest_fn(agent, prototype_checkpoint.data_segment)
            prototype_metrics = prototype_results.get('metrics', {})
            
            # Бэктест на контрпримере
            contrast_results = backtest_fn(agent, contrast_checkpoint.data_segment)
            contrast_metrics = contrast_results.get('metrics', {})
            
            # Сравнение через Mann-Whitney U test
            prototype_returns = prototype_results.get('returns', pd.Series([0]))
            contrast_returns = contrast_results.get('returns', pd.Series([0]))
            
            if len(prototype_returns) < 2 or len(contrast_returns) < 2:
                return False, 1.0, {}
            
            statistic, p_value = mannwhitneyu(
                prototype_returns.dropna(),
                contrast_returns.dropna(),
                alternative='greater'
            )
            
            # Проверка значимого различия
            passed = p_value < HYPERPARAMS.curriculum.mann_whitney_pvalue
            
            statistics = {
                'prototype_sharpe': prototype_metrics.get('sharpe_ratio', 0),
                'contrast_sharpe': contrast_metrics.get('sharpe_ratio', 0),
                'prototype_trades': prototype_metrics.get('num_trades', 0),
                'contrast_trades': contrast_metrics.get('num_trades', 0),
                'mann_whitney_statistic': float(statistic),
                'p_value': float(p_value),
                'passed': passed
            }
            
            LOGGER.debug(f"Контрастная валидация: p_value={p_value:.4f}, passed={passed}")
            
            return passed, p_value, statistics
            
        except Exception as e:
            LOGGER.error(f"Ошибка валидации на контрастной паре: {e}", exc_info=True)
            return False, 1.0, {}
    
    def get_next_stage(self) -> Optional[StageConfig]:
        """Получение следующего этапа обучения"""
        try:
            if self.current_stage >= len(self.stages):
                return None
            
            stage = self.stages[self.current_stage]
            return stage
            
        except Exception as e:
            LOGGER.error(f"Ошибка получения следующего этапа: {e}")
            return None
    
    def advance_stage(self) -> bool:
        """Переход к следующему этапу"""
        try:
            self.completed_stages.add(self.current_stage)
            self.current_stage += 1
            
            LOGGER.info(f"Переход к Stage {self.current_stage}")
            
            return self.current_stage < len(self.stages)
            
        except Exception as e:
            LOGGER.error(f"Ошибка перехода к следующему этапу: {e}")
            return False
    
    def get_curriculum_summary(self) -> Dict:
        """Сводка по curriculum learning"""
        try:
            summary = {
                'total_stages': len(self.stages),
                'current_stage': self.current_stage,
                'completed_stages': list(self.completed_stages),
                'total_checkpoints': sum(len(cps) for cps in self.checkpoints.values()),
                'checkpoint_results': len(self.checkpoint_results),
                'stages_info': [
                    {
                        'stage_id': s.stage_id,
                        'type': s.stage_type.name,
                        'num_checkpoints': len(self.checkpoints.get(s.stage_id, [])),
                        'complexity': s.complexity_level
                    }
                    for s in self.stages
                ]
            }
            
            return summary
            
        except Exception as e:
            LOGGER.error(f"Ошибка создания сводки: {e}")
            return {}


def compute_checkpoint_difficulty(checkpoint: Checkpoint,
                                  historical_performance: Optional[Dict] = None) -> float:
    """
    Вычисление сложности чекпоинта на основе характеристик.
    
    Args:
        checkpoint: Чекпоинт для оценки
        historical_performance: История производительности агентов
    
    Returns:
        Оценка сложности от 0 (легко) до 1 (сложно)
    """
    try:
        segment = checkpoint.data_segment
        
        returns = segment['Close'].pct_change()
        
        # Факторы сложности
        volatility = returns.std()
        num_reversals = (np.diff(np.sign(returns.dropna())) != 0).sum() / len(segment)
        
        range_pct = (segment['High'].max() - segment['Low'].min()) / segment['Close'].mean()
        
        # Нормализация и комбинирование
        difficulty = (
            min(volatility * 20, 1.0) * 0.4 +
            num_reversals * 0.3 +
            min(range_pct * 5, 1.0) * 0.3
        )
        
        # Корректировка на основе исторической производительности
        if historical_performance:
            avg_success_rate = historical_performance.get('success_rate', 0.5)
            difficulty *= (1.0 + (0.5 - avg_success_rate))
        
        return np.clip(difficulty, 0, 1)
        
    except Exception as e:
        LOGGER.error(f"Ошибка вычисления сложности checkpoint: {e}")
        return 0.5