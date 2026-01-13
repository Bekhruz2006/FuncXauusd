"""
Генетический агент с эволюционной оптимизацией параметров и правил.
Реализует генетическое программирование для синтеза торговых стратегий.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import copy
import random

from agents.base_agent import (
    BaseAgent, AgentGene, TradingRule, TradeSignal, RuleType,
    create_random_rule, create_random_condition, AgentValidator
)
from agents.risk_manager import RiskManager
from config.hyperparameters import HYPERPARAMS
from utils.logger import LOGGER


class GeneticAgent(BaseAgent):
    """Агент с генетически эволюционируемыми правилами"""
    
    def __init__(self,
                 agent_id: Optional[str] = None,
                 gene: Optional[AgentGene] = None,
                 available_features: Optional[List[str]] = None):
        
        super().__init__(agent_id=agent_id, gene=gene)
        
        self.available_features = available_features or []
        self.risk_manager = RiskManager(
            initial_capital=HYPERPARAMS.risk.initial_deposit,
            max_drawdown=HYPERPARAMS.risk.max_drawdown_threshold
        )
        
        self.current_positions: List[Dict] = []
        self.evaluation_count = 0
        
    def generate_signal(self,
                       market_data: pd.DataFrame,
                       features: pd.DataFrame,
                       current_position: Optional[Dict] = None) -> Optional[TradeSignal]:
        """
        Генерация торгового сигнала на основе правил гена.
        
        Args:
            market_data: OHLCV данные
            features: Вычисленные признаки
            current_position: Текущая открытая позиция (если есть)
        
        Returns:
            TradeSignal или None
        """
        try:
            self.evaluation_count += 1
            
            if len(market_data) < 2 or len(features) < 2:
                return None
            
            # Текущий бар
            current_bar = len(market_data) - 1
            current_price = market_data['Close'].iloc[-1]
            current_time = market_data.index[-1]
            
            # Feature dict для оценки правил
            feature_dict = features.iloc[-1].to_dict()
            
            # Если есть позиция - проверяем exit
            if current_position is not None:
                exit_signal = self._evaluate_exit_conditions(
                    feature_dict, current_position, market_data
                )
                if exit_signal:
                    return exit_signal
                
                # Проверка max hold bars
                if current_position['bars_held'] >= self.gene.max_hold_bars:
                    return TradeSignal(
                        signal_type="EXIT",
                        confidence=1.0,
                        metadata={'reason': 'max_hold_bars_reached'},
                        timestamp=current_time
                    )
                
                return None
            
            # Если нет позиции - проверяем entry
            if len(self.current_positions) >= self.gene.max_concurrent_positions:
                return None
            
            entry_signal = self._evaluate_entry_conditions(
                feature_dict, market_data, features
            )
            
            return entry_signal
            
        except Exception as e:
            LOGGER.error(f"Ошибка генерации сигнала {self.agent_id}: {e}", exc_info=True)
            return None
    
    def _evaluate_entry_conditions(self,
                                   feature_dict: Dict[str, float],
                                   market_data: pd.DataFrame,
                                   features: pd.DataFrame) -> Optional[TradeSignal]:
        """Оценка условий входа"""
        try:
            # Проверяем filter rules
            for filter_rule in self.gene.filter_rules:
                result, _ = filter_rule.evaluate(feature_dict)
                if not result:
                    return None
            
            # Оцениваем entry rules
            entry_confidences = []
            
            for entry_rule in self.gene.entry_rules:
                result, confidence = entry_rule.evaluate(feature_dict)
                if result:
                    entry_confidences.append(confidence)
            
            if not entry_confidences:
                return None
            
            avg_confidence = np.mean(entry_confidences)
            
            if avg_confidence < self.gene.min_confidence_entry:
                return None
            
            # Вычисляем уровни входа/выхода
            current_price = market_data['Close'].iloc[-1]
            atr = self._compute_atr(market_data, period=14)
            
            stop_loss = current_price - (self.gene.atr_multiplier_sl * atr)
            
            take_profits = [
                current_price + (tp_mult * atr)
                for tp_mult in self.gene.atr_multiplier_tp
            ]
            
            # Размер позиции
            position_size = self.risk_manager.calculate_position_size(
                entry_price=current_price,
                stop_loss=stop_loss,
                risk_percent=self.gene.risk_percent,
                atr=atr
            )
            
            signal = TradeSignal(
                signal_type="ENTRY_LONG",
                confidence=avg_confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profits=take_profits,
                position_size=position_size,
                metadata={
                    'atr': atr,
                    'num_rules_fired': len(entry_confidences),
                    'partial_exit_fractions': self.gene.partial_exit_fractions
                },
                timestamp=market_data.index[-1]
            )
            
            return signal
            
        except Exception as e:
            LOGGER.error(f"Ошибка оценки entry условий: {e}")
            return None
    
    def _evaluate_exit_conditions(self,
                                  feature_dict: Dict[str, float],
                                  current_position: Dict,
                                  market_data: pd.DataFrame) -> Optional[TradeSignal]:
        """Оценка условий выхода"""
        try:
            # Проверяем exit rules
            for exit_rule in self.gene.exit_rules:
                result, confidence = exit_rule.evaluate(feature_dict)
                if result and confidence > 0.6:
                    return TradeSignal(
                        signal_type="EXIT",
                        confidence=confidence,
                        metadata={'reason': 'exit_rule_triggered', 'rule_id': exit_rule.rule_id},
                        timestamp=market_data.index[-1]
                    )
            
            # Проверка SL/TP
            current_price = market_data['Close'].iloc[-1]
            
            if current_price <= current_position['stop_loss']:
                return TradeSignal(
                    signal_type="EXIT",
                    confidence=1.0,
                    metadata={'reason': 'stop_loss_hit'},
                    timestamp=market_data.index[-1]
                )
            
            # Проверка частичных TP
            for i, tp_level in enumerate(current_position['take_profits']):
                if current_price >= tp_level and not current_position.get(f'tp_{i}_hit', False):
                    return TradeSignal(
                        signal_type="PARTIAL_EXIT",
                        confidence=1.0,
                        metadata={
                            'reason': 'take_profit_hit',
                            'tp_level': i,
                            'exit_fraction': self.gene.partial_exit_fractions[i]
                        },
                        timestamp=market_data.index[-1]
                    )
            
            return None
            
        except Exception as e:
            LOGGER.error(f"Ошибка оценки exit условий: {e}")
            return None
    
    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Вычисление ATR"""
        try:
            if len(df) < period:
                return df['Close'].iloc[-1] * 0.01
            
            high = df['High'].iloc[-period:]
            low = df['Low'].iloc[-period:]
            close = df['Close'].iloc[-period-1:-1]
            
            tr1 = high - low
            tr2 = (high - close.values).abs()
            tr3 = (low - close.values).abs()
            
            tr = pd.concat([tr1, pd.Series(tr2.values), pd.Series(tr3.values)], axis=1).max(axis=1)
            atr = tr.mean()
            
            return atr
            
        except Exception as e:
            LOGGER.error(f"Ошибка вычисления ATR: {e}")
            return df['Close'].iloc[-1] * 0.01
    
    def evaluate_fitness(self,
                        backtest_results: Dict,
                        metrics: Dict[str, float]) -> float:
        """
        Лексикографическая оценка fitness агента.
        
        Args:
            backtest_results: Результаты бэктеста
            metrics: Словарь метрик (sharpe, recovery, profit_factor, etc.)
        
        Returns:
            Fitness score
        """
        try:
            # Уровень 1: Выживание (MaxDD)
            max_dd = metrics.get('max_drawdown', 1.0)
            
            if max_dd > HYPERPARAMS.risk.max_drawdown_threshold:
                self.fitness_score = -1000.0
                return self.fitness_score
            
            # Уровень 2: Устойчивость (взвешенная комбинация метрик)
            sharpe = metrics.get('sharpe_ratio', 0.0)
            recovery = metrics.get('recovery_factor', 0.0)
            profit_factor = metrics.get('profit_factor', 1.0)
            
            # Нормализация метрик
            sharpe_norm = np.tanh(sharpe / 2.0)
            recovery_norm = np.tanh(recovery / 3.0)
            profit_norm = np.tanh((profit_factor - 1.0) / 2.0)
            
            # Взвешенная комбинация
            weights = HYPERPARAMS.optimization.fitness_weights
            
            fitness = (
                weights['sharpe_ratio'] * sharpe_norm +
                weights['recovery_factor'] * recovery_norm +
                weights['profit_factor'] * profit_norm
            )
            
            # Уровень 3: Стабильность (penalty за волатильность доходности)
            returns = backtest_results.get('returns', pd.Series())
            
            if len(returns) > 1:
                returns_volatility = returns.std()
                stability_penalty = np.clip(returns_volatility * 10, 0, 0.5)
                fitness -= stability_penalty
            
            # Penalty за малое количество сделок
            num_trades = metrics.get('num_trades', 0)
            if num_trades < HYPERPARAMS.validation.min_trades_per_checkpoint:
                fitness *= 0.5
            
            self.fitness_score = fitness
            self.metrics = metrics
            
            return fitness
            
        except Exception as e:
            LOGGER.error(f"Ошибка оценки fitness: {e}", exc_info=True)
            self.fitness_score = -999.0
            return self.fitness_score
    
    def mutate(self, mutation_rate: float = 0.15) -> None:
        """Мутация гена агента"""
        try:
            # Мутация численных параметров
            if random.random() < mutation_rate:
                self.gene.risk_percent *= np.random.uniform(0.8, 1.2)
                self.gene.risk_percent = np.clip(self.gene.risk_percent, 0.005, 0.025)
            
            if random.random() < mutation_rate:
                self.gene.atr_multiplier_sl *= np.random.uniform(0.8, 1.2)
                self.gene.atr_multiplier_sl = np.clip(self.gene.atr_multiplier_sl, 1.0, 5.0)
            
            # Мутация правил
            all_rules = self.gene.entry_rules + self.gene.exit_rules + self.gene.filter_rules
            
            for rule in all_rules:
                if random.random() < mutation_rate:
                    self._mutate_rule(rule)
            
            # Добавление/удаление правил
            if random.random() < mutation_rate * 0.5:
                self._add_random_rule()
            
            if random.random() < mutation_rate * 0.3:
                self._remove_random_rule()
            
            LOGGER.debug(f"Агент {self.agent_id[:8]} мутировал")
            
        except Exception as e:
            LOGGER.error(f"Ошибка мутации агента: {e}")
    
    def _mutate_rule(self, rule: TradingRule) -> None:
        """Мутация одного правила"""
        try:
            if not rule.conditions:
                return
            
            mutation_type = random.choice(['threshold', 'operator', 'weight', 'add_condition', 'remove_condition'])
            
            if mutation_type == 'threshold':
                condition = random.choice(rule.conditions)
                condition.threshold *= np.random.uniform(0.7, 1.3)
            
            elif mutation_type == 'operator':
                condition = random.choice(rule.conditions)
                from agents.base_agent import ComparisonOperator
                condition.operator = random.choice(list(ComparisonOperator))
            
            elif mutation_type == 'weight':
                condition = random.choice(rule.conditions)
                condition.weight *= np.random.uniform(0.8, 1.2)
                condition.weight = np.clip(condition.weight, 0.1, 2.0)
            
            elif mutation_type == 'add_condition' and len(rule.conditions) < 5:
                new_condition = create_random_condition(self.available_features)
                rule.conditions.append(new_condition)
            
            elif mutation_type == 'remove_condition' and len(rule.conditions) > 1:
                rule.conditions.pop(random.randint(0, len(rule.conditions) - 1))
            
        except Exception as e:
            LOGGER.error(f"Ошибка мутации правила: {e}")
    
    def _add_random_rule(self) -> None:
        """Добавление случайного правила"""
        try:
            rule_type = random.choice([RuleType.ENTRY, RuleType.EXIT, RuleType.FILTER])
            new_rule = create_random_rule(rule_type, self.available_features)
            
            if rule_type == RuleType.ENTRY:
                self.gene.entry_rules.append(new_rule)
            elif rule_type == RuleType.EXIT:
                self.gene.exit_rules.append(new_rule)
            else:
                self.gene.filter_rules.append(new_rule)
                
        except Exception as e:
            LOGGER.error(f"Ошибка добавления правила: {e}")
    
    def _remove_random_rule(self) -> None:
        """Удаление случайного правила"""
        try:
            all_rules = [
                (self.gene.entry_rules, 'entry'),
                (self.gene.exit_rules, 'exit'),
                (self.gene.filter_rules, 'filter')
            ]
            
            non_empty_rules = [(rules, name) for rules, name in all_rules if len(rules) > 1]
            
            if not non_empty_rules:
                return
            
            rules_list, _ = random.choice(non_empty_rules)
            rules_list.pop(random.randint(0, len(rules_list) - 1))
            
        except Exception as e:
            LOGGER.error(f"Ошибка удаления правила: {e}")


class GeneticPopulation:
    """Популяция генетических агентов"""
    
    def __init__(self,
                 population_size: Optional[int] = None,
                 available_features: Optional[List[str]] = None):
        
        self.population_size = population_size or HYPERPARAMS.agent.population_size
        self.available_features = available_features or []
        
        self.agents: List[GeneticAgent] = []
        self.generation = 0
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        
        LOGGER.info(f"Инициализация популяции: размер={self.population_size}")
    
    def initialize_random_population(self) -> None:
        """Инициализация случайной популяции"""
        try:
            self.agents = []
            
            for i in range(self.population_size):
                try:
                    gene = self._create_random_gene()
                    agent = GeneticAgent(
                        gene=gene,
                        available_features=self.available_features
                    )
                    agent.generation = self.generation
                    self.agents.append(agent)
                    
                except Exception as e:
                    LOGGER.error(f"Ошибка создания агента {i}: {e}")
                    continue
            
            LOGGER.info(f"Создано {len(self.agents)} агентов в поколении {self.generation}")
            
        except Exception as e:
            LOGGER.error(f"Ошибка инициализации популяции: {e}", exc_info=True)
            raise
    
    def _create_random_gene(self) -> AgentGene:
        """Создание случайного гена"""
        try:
            num_entry_rules = np.random.randint(1, HYPERPARAMS.agent.max_rules_per_agent + 1)
            num_exit_rules = np.random.randint(1, HYPERPARAMS.agent.max_rules_per_agent + 1)
            num_filter_rules = np.random.randint(0, 3)
            
            entry_rules = [
                create_random_rule(RuleType.ENTRY, self.available_features)
                for _ in range(num_entry_rules)
            ]
            
            exit_rules = [
                create_random_rule(RuleType.EXIT, self.available_features)
                for _ in range(num_exit_rules)
            ]
            
            filter_rules = [
                create_random_rule(RuleType.FILTER, self.available_features)
                for _ in range(num_filter_rules)
            ]
            
            risk_percent = np.random.uniform(*HYPERPARAMS.risk.risk_percent_range)
            atr_multiplier_sl = np.random.uniform(*HYPERPARAMS.risk.atr_multiplier_range)
            
            num_tp = HYPERPARAMS.agent.partial_exit_points
            atr_multiplier_tp = sorted([np.random.uniform(1.0, 5.0) for _ in range(num_tp)])
            
            partial_fractions = np.random.dirichlet(np.ones(num_tp))
            
            gene = AgentGene(
                entry_rules=entry_rules,
                exit_rules=exit_rules,
                filter_rules=filter_rules,
                risk_percent=risk_percent,
                atr_multiplier_sl=atr_multiplier_sl,
                atr_multiplier_tp=atr_multiplier_tp,
                partial_exit_fractions=list(partial_fractions),
                max_hold_bars=np.random.randint(24, 72)
            )
            
            return gene
            
        except Exception as e:
            LOGGER.error(f"Ошибка создания случайного гена: {e}")
            raise
    
    def evolve_generation(self,
                         tournament_size: Optional[int] = None,
                         crossover_prob: Optional[float] = None,
                         mutation_prob: Optional[float] = None,
                         elitism_count: Optional[int] = None) -> None:
        """Эволюция поколения"""
        try:
            tournament_size = tournament_size or HYPERPARAMS.agent.tournament_size
            crossover_prob = crossover_prob or HYPERPARAMS.agent.crossover_prob
            mutation_prob = mutation_prob or HYPERPARAMS.agent.mutation_prob
            elitism_count = elitism_count or HYPERPARAMS.agent.elitism_count
            
            # Сортируем по fitness
            self.agents.sort(key=lambda a: a.fitness_score, reverse=True)
            
            # Статистика
            best_fitness = self.agents[0].fitness_score
            avg_fitness = np.mean([a.fitness_score for a in self.agents])
            
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            
            LOGGER.info(f"Поколение {self.generation}: best={best_fitness:.4f}, avg={avg_fitness:.4f}")
            
            # Elitism: сохраняем лучших
            new_population = self.agents[:elitism_count]
            
            # Генерация новых агентов
            while len(new_population) < self.population_size:
                # Турнирная селекция
                parent1 = self._tournament_selection(tournament_size)
                parent2 = self._tournament_selection(tournament_size)
                
                # Crossover
                if random.random() < crossover_prob:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.clone()
                
                # Mutation
                if random.random() < mutation_prob:
                    child.mutate(mutation_rate=mutation_prob)
                
                child.generation = self.generation + 1
                new_population.append(child)
            
            self.agents = new_population[:self.population_size]
            self.generation += 1
            
        except Exception as e:
            LOGGER.error(f"Ошибка эволюции поколения: {e}", exc_info=True)
            raise
    
    def _tournament_selection(self, tournament_size: int) -> GeneticAgent:
        """Турнирная селекция"""
        try:
            tournament = random.sample(self.agents, min(tournament_size, len(self.agents)))
            winner = max(tournament, key=lambda a: a.fitness_score)
            return winner
            
        except Exception as e:
            LOGGER.error(f"Ошибка турнирной селекции: {e}")
            return random.choice(self.agents)
    
    def _crossover(self, parent1: GeneticAgent, parent2: GeneticAgent) -> GeneticAgent:
        """Скрещивание двух родителей"""
        try:
            child_gene = AgentGene()
            
            # Наследуем правила
            child_gene.entry_rules = self._crossover_rules(
                parent1.gene.entry_rules,
                parent2.gene.entry_rules
            )
            
            child_gene.exit_rules = self._crossover_rules(
                parent1.gene.exit_rules,
                parent2.gene.exit_rules
            )
            
            child_gene.filter_rules = self._crossover_rules(
                parent1.gene.filter_rules,
                parent2.gene.filter_rules
            )
            
            # Наследуем параметры
            child_gene.risk_percent = random.choice([parent1.gene.risk_percent, parent2.gene.risk_percent])
            child_gene.atr_multiplier_sl = random.choice([parent1.gene.atr_multiplier_sl, parent2.gene.atr_multiplier_sl])
            
            # Комбинируем TP параметры
            if random.random() < 0.5:
                child_gene.atr_multiplier_tp = parent1.gene.atr_multiplier_tp.copy()
                child_gene.partial_exit_fractions = parent1.gene.partial_exit_fractions.copy()
            else:
                child_gene.atr_multiplier_tp = parent2.gene.atr_multiplier_tp.copy()
                child_gene.partial_exit_fractions = parent2.gene.partial_exit_fractions.copy()
            
            child = GeneticAgent(gene=child_gene, available_features=self.available_features)
            
            return child
            
        except Exception as e:
            LOGGER.error(f"Ошибка скрещивания: {e}")
            return parent1.clone()
    
    def _crossover_rules(self, rules1: List[TradingRule], rules2: List[TradingRule]) -> List[TradingRule]:
        """Скрещивание списков правил"""
        try:
            if not rules1 and not rules2:
                return []
            
            if not rules1:
                return copy.deepcopy(random.sample(rules2, min(len(rules2), 3)))
            
            if not rules2:
                return copy.deepcopy(random.sample(rules1, min(len(rules1), 3)))
            
            # Случайная комбинация правил
            child_rules = []
            
            num_from_parent1 = random.randint(0, len(rules1))
            num_from_parent2 = random.randint(0, len(rules2))
            
            if num_from_parent1 > 0:
                child_rules.extend(copy.deepcopy(random.sample(rules1, num_from_parent1)))
            
            if num_from_parent2 > 0:
                child_rules.extend(copy.deepcopy(random.sample(rules2, num_from_parent2)))
            
            return child_rules[:HYPERPARAMS.agent.max_rules_per_agent]
            
        except Exception as e:
            LOGGER.error(f"Ошибка скрещивания правил: {e}")
            return copy.deepcopy(rules1[:3])
    
    def get_best_agents(self, n: int = 10) -> List[GeneticAgent]:
        """Получение лучших агентов"""
        try:
            sorted_agents = sorted(self.agents, key=lambda a: a.fitness_score, reverse=True)
            return sorted_agents[:n]
        except Exception as e:
            LOGGER.error(f"Ошибка получения лучших агентов: {e}")
            return []