"""
Централизованное хранилище всех гиперпараметров системы.
Разделено на секции для удобства настройки и масштабирования.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class DataConfig:
    """Параметры работы с данными"""
    base_path: str = "D:/MQ5/Пак/FuncXauusd/data/raw"
    timeframes: List[str] = field(default_factory=lambda: ["M1", "M5", "15M", "30M", "H1", "H4"])
    start_date: str = "2004-06-11"
    end_date: str = "2025-01-12"
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    zscore_window: int = 100
    low_liquidity_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 22, 23])
    fill_method: str = "ffill"


@dataclass
class RegimeConfig:
    """Параметры определения рыночных режимов"""
    num_regimes: int = 5
    regime_names: List[str] = field(default_factory=lambda: [
        "TREND_UP", "TREND_DOWN", "VOLATILE", "RANGING", "TRANSITION"
    ])
    pelt_penalty: float = 10.0
    min_segment_length: int = 168  # 1 неделя на H1
    regime_features: List[str] = field(default_factory=lambda: [
        "volatility_regime", "trend_strength", "mean_reversion_idx"
    ])


@dataclass
class AgentConfig:
    """Параметры агентов и генетического программирования"""
    population_size: int = 200
    num_generations: int = 100
    tournament_size: int = 7
    crossover_prob: float = 0.75
    mutation_prob: float = 0.15
    elitism_count: int = 10
    max_tree_depth: int = 8
    min_tree_depth: int = 3
    max_rules_per_agent: int = 5
    partial_exit_points: int = 3  # TP1, TP2, TP3


@dataclass
class RiskConfig:
    """Параметры управления риском"""
    initial_deposit: float = 350.0
    max_drawdown_threshold: float = 0.15
    risk_percent_range: Tuple[float, float] = (0.005, 0.025)  # 0.5% - 2.5%
    atr_multiplier_range: Tuple[float, float] = (1.5, 4.0)
    max_concurrent_positions: int = 3
    min_lot_size: float = 0.01
    max_lot_size: float = 0.5
    spread_points: float = 20.0
    swap_long: float = -5.0  # USD per lot per day


@dataclass
class CurriculumConfig:
    """Параметры curriculum learning"""
    num_stages: int = 5
    checkpoints_per_stage: int = 15
    stage_0_duration: int = 720  # 1 месяц H1 для синтетики
    stage_1_duration: int = 2160  # 3 месяца
    stage_2_duration: int = 4320  # 6 месяцев
    stage_3_duration: int = 8640  # 12 месяцев
    stage_4_duration: int = 17280  # 24 месяца
    contrastive_pairs: bool = True
    mann_whitney_pvalue: float = 0.1


@dataclass
class ValidationConfig:
    """Параметры валидации"""
    adversarial_auc_threshold: float = 0.65
    num_adversarial_models: int = 3
    structural_shift_multiplier: float = 2.0
    sensitivity_test_grid_size: int = 5
    economic_significance_pvalue: float = 0.05
    min_trades_per_checkpoint: int = 10


@dataclass
class OptimizationConfig:
    """Параметры оптимизации"""
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        "sharpe_ratio": 0.4,
        "recovery_factor": 0.35,
        "profit_factor": 0.25
    })
    convergence_patience: int = 15
    performance_degradation_threshold: float = 0.05
    meta_calibration_epochs: int = 50


@dataclass
class SyntheticConfig:
    """Параметры cGAN для синтетических данных"""
    latent_dim: int = 128
    generator_lr: float = 0.0002
    discriminator_lr: float = 0.0002
    batch_size: int = 64
    num_epochs: int = 200
    condition_dim: int = 16
    gradient_penalty_weight: float = 10.0


@dataclass
class ExportConfig:
    """Параметры экспорта"""
    onnx_opset_version: int = 13
    mql5_template_path: str = "templates/advisor_template.mq5"
    models_output_path: str = "D:/MQ5/Пак/FuncXauusd/models/exported"
    discriminator_refresh_hours: int = 4


class HyperParameters:
    """Главный класс-контейнер всех конфигураций"""
    
    def __init__(self):
        self.data = DataConfig()
        self.regime = RegimeConfig()
        self.agent = AgentConfig()
        self.risk = RiskConfig()
        self.curriculum = CurriculumConfig()
        self.validation = ValidationConfig()
        self.optimization = OptimizationConfig()
        self.synthetic = SyntheticConfig()
        self.export = ExportConfig()
        
        self._validate_params()
    
    def _validate_params(self) -> None:
        """Валидация взаимосвязей между параметрами"""
        try:
            assert self.data.train_ratio + self.data.validation_ratio + self.data.test_ratio == 1.0, \
                "Сумма разбиений данных должна быть 1.0"
            
            assert self.risk.max_drawdown_threshold > 0 and self.risk.max_drawdown_threshold < 1, \
                "MaxDD должен быть в интервале (0, 1)"
            
            assert self.curriculum.num_stages <= 5, \
                "Число этапов curriculum не должно превышать 5"
            
            assert self.agent.max_tree_depth >= self.agent.min_tree_depth, \
                "max_tree_depth должен быть >= min_tree_depth"
                
        except AssertionError as e:
            raise ValueError(f"Ошибка валидации гиперпараметров: {e}")
    
    def to_dict(self) -> Dict:
        """Сериализация всех параметров в словарь"""
        return {
            "data": self.data.__dict__,
            "regime": self.regime.__dict__,
            "agent": self.agent.__dict__,
            "risk": self.risk.__dict__,
            "curriculum": self.curriculum.__dict__,
            "validation": self.validation.__dict__,
            "optimization": self.optimization.__dict__,
            "synthetic": self.synthetic.__dict__,
            "export": self.export.__dict__
        }
    
    def update_from_dict(self, config_dict: Dict) -> None:
        """Обновление параметров из словаря (для экспериментов)"""
        for section, params in config_dict.items():
            if hasattr(self, section):
                config_obj = getattr(self, section)
                for key, value in params.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
        self._validate_params()


# Глобальный экземпляр параметров
HYPERPARAMS = HyperParameters()