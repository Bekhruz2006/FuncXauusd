"""
Модуль оптимизации: curriculum learning, fitness evaluation,
генетические операторы, мета-оптимизатор.
"""

from .curriculum_learning import CurriculumLearningScheduler, StageConfig
from .fitness_evaluator import FitnessEvaluator, MetricsCalculator
from .genetic_operators import GeneticOperators, SelectionStrategy
from .meta_optimizer import MetaOptimizer, OptimizationResults

__all__ = [
    'CurriculumLearningScheduler',
    'StageConfig',
    'FitnessEvaluator',
    'MetricsCalculator',
    'GeneticOperators',
    'SelectionStrategy',
    'MetaOptimizer',
    'OptimizationResults'
]