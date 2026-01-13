"""
Модуль агентов: базовые классы, генетическое программирование,
управление риском и эволюционная оптимизация.
"""

from .base_agent import BaseAgent, AgentGene, TradingRule
from .genetic_agent import GeneticAgent, GeneticPopulation
from .risk_manager import RiskManager, PositionSizer

__all__ = [
    'BaseAgent',
    'AgentGene',
    'TradingRule',
    'GeneticAgent',
    'GeneticPopulation',
    'RiskManager',
    'PositionSizer'
]