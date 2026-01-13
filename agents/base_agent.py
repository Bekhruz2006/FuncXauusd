"""
Базовый абстрактный класс агента и структуры данных для представления
торговых правил и генотипов.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import uuid

from utils.logger import LOGGER


class RuleType(Enum):
    """Типы торговых правил"""
    ENTRY = "entry"
    EXIT = "exit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    FILTER = "filter"


class LogicalOperator(Enum):
    """Логические операторы для комбинирования условий"""
    AND = "and"
    OR = "or"
    NOT = "not"
    XOR = "xor"


class ComparisonOperator(Enum):
    """Операторы сравнения"""
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    EQ = "=="
    NEQ = "!="


@dataclass
class Condition:
    """Единичное условие в торговом правиле"""
    feature_name: str
    operator: ComparisonOperator
    threshold: float
    weight: float = 1.0
    
    def evaluate(self, feature_value: float) -> bool:
        """Оценка условия"""
        try:
            if self.operator == ComparisonOperator.GT:
                return feature_value > self.threshold
            elif self.operator == ComparisonOperator.LT:
                return feature_value < self.threshold
            elif self.operator == ComparisonOperator.GTE:
                return feature_value >= self.threshold
            elif self.operator == ComparisonOperator.LTE:
                return feature_value <= self.threshold
            elif self.operator == ComparisonOperator.EQ:
                return abs(feature_value - self.threshold) < 1e-6
            elif self.operator == ComparisonOperator.NEQ:
                return abs(feature_value - self.threshold) >= 1e-6
            else:
                return False
        except Exception as e:
            LOGGER.error(f"Ошибка оценки условия: {e}")
            return False
    
    def __repr__(self) -> str:
        return f"{self.feature_name} {self.operator.value} {self.threshold:.3f}"


@dataclass
class TradingRule:
    """Торговое правило - комбинация условий"""
    rule_id: str
    rule_type: RuleType
    conditions: List[Condition]
    logical_operator: LogicalOperator = LogicalOperator.AND
    is_active: bool = True
    execution_priority: int = 0
    
    def evaluate(self, feature_dict: Dict[str, float]) -> Tuple[bool, float]:
        """
        Оценка правила на данных.
        
        Returns:
            (result, confidence)
        """
        try:
            if not self.is_active or not self.conditions:
                return False, 0.0
            
            evaluations = []
            weights = []
            
            for condition in self.conditions:
                if condition.feature_name not in feature_dict:
                    continue
                
                feature_value = feature_dict[condition.feature_name]
                result = condition.evaluate(feature_value)
                
                evaluations.append(result)
                weights.append(condition.weight)
            
            if not evaluations:
                return False, 0.0
            
            # Применяем логический оператор
            if self.logical_operator == LogicalOperator.AND:
                final_result = all(evaluations)
            elif self.logical_operator == LogicalOperator.OR:
                final_result = any(evaluations)
            elif self.logical_operator == LogicalOperator.NOT:
                final_result = not evaluations[0] if evaluations else False
            elif self.logical_operator == LogicalOperator.XOR:
                final_result = sum(evaluations) == 1
            else:
                final_result = False
            
            # Confidence на основе весов
            if final_result:
                satisfied_weights = [w for e, w in zip(evaluations, weights) if e]
                confidence = sum(satisfied_weights) / (sum(weights) + 1e-8)
            else:
                confidence = 0.0
            
            return final_result, confidence
            
        except Exception as e:
            LOGGER.error(f"Ошибка оценки правила {self.rule_id}: {e}")
            return False, 0.0
    
    def __repr__(self) -> str:
        conditions_str = f" {self.logical_operator.value} ".join([str(c) for c in self.conditions])
        return f"Rule[{self.rule_type.value}]: {conditions_str}"


@dataclass
class AgentGene:
    """Ген агента - набор параметров для эволюции"""
    entry_rules: List[TradingRule] = field(default_factory=list)
    exit_rules: List[TradingRule] = field(default_factory=list)
    filter_rules: List[TradingRule] = field(default_factory=list)
    
    risk_percent: float = 0.01
    atr_multiplier_sl: float = 2.0
    atr_multiplier_tp: List[float] = field(default_factory=lambda: [1.5, 2.5, 3.5])
    partial_exit_fractions: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    max_hold_bars: int = 48
    
    max_concurrent_positions: int = 1
    min_confidence_entry: float = 0.5
    
    def __post_init__(self):
        """Валидация после инициализации"""
        try:
            assert len(self.atr_multiplier_tp) == len(self.partial_exit_fractions), \
                "TP multipliers и exit fractions должны иметь одинаковую длину"
            
            assert abs(sum(self.partial_exit_fractions) - 1.0) < 0.01, \
                "Сумма partial_exit_fractions должна быть ~1.0"
            
            assert 0 < self.risk_percent < 0.1, \
                "risk_percent должен быть в диапазоне (0, 0.1)"
                
        except AssertionError as e:
            LOGGER.error(f"Ошибка валидации гена: {e}")
            raise
    
    def to_dict(self) -> Dict:
        """Сериализация гена"""
        return {
            'entry_rules': [str(r) for r in self.entry_rules],
            'exit_rules': [str(r) for r in self.exit_rules],
            'filter_rules': [str(r) for r in self.filter_rules],
            'risk_percent': self.risk_percent,
            'atr_multiplier_sl': self.atr_multiplier_sl,
            'atr_multiplier_tp': self.atr_multiplier_tp,
            'partial_exit_fractions': self.partial_exit_fractions,
            'max_hold_bars': self.max_hold_bars,
            'max_concurrent_positions': self.max_concurrent_positions,
            'min_confidence_entry': self.min_confidence_entry
        }


@dataclass
class TradeSignal:
    """Сигнал на вход/выход из позиции"""
    signal_type: str  # "ENTRY_LONG", "EXIT", etc.
    confidence: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profits: List[float] = field(default_factory=list)
    position_size: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[pd.Timestamp] = None


class BaseAgent(ABC):
    """Абстрактный базовый класс для всех агентов"""
    
    def __init__(self, 
                 agent_id: Optional[str] = None,
                 gene: Optional[AgentGene] = None):
        
        self.agent_id = agent_id or str(uuid.uuid4())
        self.gene = gene or AgentGene()
        
        self.fitness_score: float = 0.0
        self.metrics: Dict[str, float] = {}
        self.trade_history: List[Dict] = []
        self.generation: int = 0
        
        self.is_trained: bool = False
        self.training_stats: Dict = {}
        
        LOGGER.debug(f"Инициализирован агент {self.agent_id}")
    
    @abstractmethod
    def generate_signal(self, 
                       market_data: pd.DataFrame,
                       features: pd.DataFrame,
                       current_position: Optional[Dict] = None) -> Optional[TradeSignal]:
        """
        Генерация торгового сигнала.
        Должна быть реализована в подклассах.
        """
        raise NotImplementedError
    
    @abstractmethod
    def evaluate_fitness(self, 
                        backtest_results: Dict,
                        metrics: Dict[str, float]) -> float:
        """
        Оценка fitness агента на основе результатов бэктеста.
        Должна быть реализована в подклассах.
        """
        raise NotImplementedError
    
    def add_trade_to_history(self, trade: Dict) -> None:
        """Добавление сделки в историю"""
        try:
            self.trade_history.append(trade)
        except Exception as e:
            LOGGER.error(f"Ошибка добавления сделки в историю: {e}")
    
    def get_statistics(self) -> Dict:
        """Получение статистики агента"""
        try:
            stats = {
                'agent_id': self.agent_id,
                'generation': self.generation,
                'fitness_score': self.fitness_score,
                'num_trades': len(self.trade_history),
                'is_trained': self.is_trained,
                'metrics': self.metrics,
                'gene_summary': {
                    'num_entry_rules': len(self.gene.entry_rules),
                    'num_exit_rules': len(self.gene.exit_rules),
                    'risk_percent': self.gene.risk_percent
                }
            }
            return stats
            
        except Exception as e:
            LOGGER.error(f"Ошибка получения статистики: {e}")
            return {}
    
    def clone(self) -> 'BaseAgent':
        """Клонирование агента"""
        try:
            import copy
            cloned = copy.deepcopy(self)
            cloned.agent_id = str(uuid.uuid4())
            cloned.trade_history = []
            cloned.fitness_score = 0.0
            return cloned
            
        except Exception as e:
            LOGGER.error(f"Ошибка клонирования агента: {e}")
            raise
    
    def reset(self) -> None:
        """Сброс состояния агента"""
        try:
            self.trade_history = []
            self.fitness_score = 0.0
            self.metrics = {}
            
        except Exception as e:
            LOGGER.error(f"Ошибка сброса агента: {e}")
    
    def save_checkpoint(self, filepath: str) -> None:
        """Сохранение чекпоинта агента"""
        try:
            import pickle
            
            checkpoint = {
                'agent_id': self.agent_id,
                'gene': self.gene,
                'fitness_score': self.fitness_score,
                'metrics': self.metrics,
                'generation': self.generation,
                'training_stats': self.training_stats
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            LOGGER.info(f"Чекпоинт агента сохранен: {filepath}")
            
        except Exception as e:
            LOGGER.error(f"Ошибка сохранения чекпоинта: {e}", exc_info=True)
            raise
    
    @classmethod
    def load_checkpoint(cls, filepath: str) -> 'BaseAgent':
        """Загрузка агента из чекпоинта"""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                checkpoint = pickle.load(f)
            
            agent = cls(
                agent_id=checkpoint['agent_id'],
                gene=checkpoint['gene']
            )
            
            agent.fitness_score = checkpoint['fitness_score']
            agent.metrics = checkpoint['metrics']
            agent.generation = checkpoint['generation']
            agent.training_stats = checkpoint['training_stats']
            agent.is_trained = True
            
            LOGGER.info(f"Агент загружен из чекпоинта: {filepath}")
            return agent
            
        except Exception as e:
            LOGGER.error(f"Ошибка загрузки чекпоинта: {e}", exc_info=True)
            raise
    
    def __repr__(self) -> str:
        return (f"Agent(id={self.agent_id[:8]}, gen={self.generation}, "
                f"fitness={self.fitness_score:.4f}, trades={len(self.trade_history)})")


class AgentValidator:
    """Валидатор агентов для проверки корректности"""
    
    @staticmethod
    def validate_gene(gene: AgentGene) -> Tuple[bool, List[str]]:
        """
        Валидация гена агента.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            # Проверка правил
            if not gene.entry_rules:
                errors.append("Отсутствуют entry_rules")
            
            # Проверка риск-параметров
            if not (0 < gene.risk_percent < 0.1):
                errors.append(f"Некорректный risk_percent: {gene.risk_percent}")
            
            if gene.atr_multiplier_sl <= 0:
                errors.append(f"Некорректный atr_multiplier_sl: {gene.atr_multiplier_sl}")
            
            # Проверка TP levels
            if not all(tp > 0 for tp in gene.atr_multiplier_tp):
                errors.append("Некорректные значения atr_multiplier_tp")
            
            if not all(0 <= frac <= 1 for frac in gene.partial_exit_fractions):
                errors.append("Некорректные значения partial_exit_fractions")
            
            # Проверка логической консистентности
            total_fraction = sum(gene.partial_exit_fractions)
            if abs(total_fraction - 1.0) > 0.01:
                errors.append(f"Сумма partial_exit_fractions != 1.0: {total_fraction}")
            
            is_valid = len(errors) == 0
            return is_valid, errors
            
        except Exception as e:
            errors.append(f"Ошибка валидации: {e}")
            return False, errors
    
    @staticmethod
    def validate_rules(rules: List[TradingRule]) -> Tuple[bool, List[str]]:
        """Валидация списка правил"""
        errors = []
        
        try:
            for i, rule in enumerate(rules):
                if not rule.conditions:
                    errors.append(f"Правило {i} не имеет условий")
                
                for j, condition in enumerate(rule.conditions):
                    if condition.weight <= 0:
                        errors.append(f"Правило {i}, условие {j}: некорректный вес")
            
            is_valid = len(errors) == 0
            return is_valid, errors
            
        except Exception as e:
            errors.append(f"Ошибка валидации правил: {e}")
            return False, errors


def create_random_condition(available_features: List[str],
                           value_ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> Condition:
    """Создание случайного условия для генетического программирования"""
    try:
        feature_name = np.random.choice(available_features)
        operator = np.random.choice(list(ComparisonOperator))
        
        if value_ranges and feature_name in value_ranges:
            min_val, max_val = value_ranges[feature_name]
            threshold = np.random.uniform(min_val, max_val)
        else:
            threshold = np.random.uniform(-1.0, 1.0)
        
        weight = np.random.uniform(0.5, 1.5)
        
        return Condition(
            feature_name=feature_name,
            operator=operator,
            threshold=threshold,
            weight=weight
        )
        
    except Exception as e:
        LOGGER.error(f"Ошибка создания случайного условия: {e}")
        raise


def create_random_rule(rule_type: RuleType,
                       available_features: List[str],
                       num_conditions: Optional[int] = None,
                       value_ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> TradingRule:
    """Создание случайного торгового правила"""
    try:
        if num_conditions is None:
            num_conditions = np.random.randint(1, 4)
        
        conditions = [
            create_random_condition(available_features, value_ranges)
            for _ in range(num_conditions)
        ]
        
        logical_operator = np.random.choice([LogicalOperator.AND, LogicalOperator.OR])
        
        rule = TradingRule(
            rule_id=str(uuid.uuid4()),
            rule_type=rule_type,
            conditions=conditions,
            logical_operator=logical_operator,
            is_active=True,
            execution_priority=np.random.randint(0, 10)
        )
        
        return rule
        
    except Exception as e:
        LOGGER.error(f"Ошибка создания случайного правила: {e}")
        raise