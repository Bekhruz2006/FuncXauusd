"""
Управление путями к файлам и директориям проекта.
Обеспечивает кросс-платформенную совместимость.
"""

import os
from pathlib import Path
from typing import Dict


class ProjectPaths:
    """Централизованное управление путями проекта"""
    
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            self.BASE_DIR = Path(__file__).resolve().parent.parent
        else:
            self.BASE_DIR = Path(base_dir)
        
        self._init_directories()
    
    def _init_directories(self) -> None:
        """Инициализация всех путей проекта"""
        try:
            # Основные директории
            self.DATA_DIR = self.BASE_DIR / "data"
            self.RAW_DATA_DIR = self.DATA_DIR / "raw"
            self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"
            self.SYNTHETIC_DATA_DIR = self.DATA_DIR / "synthetic"
            
            self.MODELS_DIR = self.BASE_DIR / "models"
            self.CHECKPOINTS_DIR = self.MODELS_DIR / "checkpoints"
            self.EXPORTED_DIR = self.MODELS_DIR / "exported"
            self.ONNX_DIR = self.EXPORTED_DIR / "onnx"
            
            self.LOGS_DIR = self.BASE_DIR / "logs"
            self.RESULTS_DIR = self.BASE_DIR / "results"
            self.VISUALIZATIONS_DIR = self.RESULTS_DIR / "visualizations"
            self.REPORTS_DIR = self.RESULTS_DIR / "reports"
            
            self.TEMPLATES_DIR = self.BASE_DIR / "templates"
            self.MQL5_OUTPUT_DIR = self.EXPORTED_DIR / "mql5"
            
            # Создание директорий если не существуют
            self._create_directories()
            
        except Exception as e:
            raise RuntimeError(f"Ошибка инициализации директорий: {e}")
    
    def _create_directories(self) -> None:
        """Создание всех необходимых директорий"""
        dirs_to_create = [
            self.RAW_DATA_DIR, self.PROCESSED_DATA_DIR, self.SYNTHETIC_DATA_DIR,
            self.CHECKPOINTS_DIR, self.ONNX_DIR, self.MQL5_OUTPUT_DIR,
            self.LOGS_DIR, self.VISUALIZATIONS_DIR, self.REPORTS_DIR
        ]
        
        for directory in dirs_to_create:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise OSError(f"Не удалось создать директорию {directory}: {e}")
    
    def get_raw_data_path(self, timeframe: str) -> Path:
        """Путь к сырым данным для таймфрейма"""
        try:
            filename = f"XAUUSD_{timeframe}.csv"
            return self.RAW_DATA_DIR / filename
        except Exception as e:
            raise ValueError(f"Ошибка формирования пути для {timeframe}: {e}")
    
    def get_processed_data_path(self, timeframe: str, suffix: str = "") -> Path:
        """Путь к обработанным данным"""
        try:
            filename = f"XAUUSD_{timeframe}_processed{suffix}.pkl"
            return self.PROCESSED_DATA_DIR / filename
        except Exception as e:
            raise ValueError(f"Ошибка формирования пути: {e}")
    
    def get_checkpoint_path(self, stage: int, generation: int) -> Path:
        """Путь к чекпоинту агента"""
        try:
            filename = f"stage_{stage}_gen_{generation}.pkl"
            return self.CHECKPOINTS_DIR / filename
        except Exception as e:
            raise ValueError(f"Ошибка формирования пути чекпоинта: {e}")
    
    def get_onnx_model_path(self, model_name: str) -> Path:
        """Путь к ONNX модели"""
        try:
            filename = f"{model_name}.onnx"
            return self.ONNX_DIR / filename
        except Exception as e:
            raise ValueError(f"Ошибка формирования пути ONNX: {e}")
    
    def get_mql5_output_path(self, agent_id: str) -> Path:
        """Путь к сгенерированному MQL5 коду"""
        try:
            filename = f"Agent_{agent_id}.mq5"
            return self.MQL5_OUTPUT_DIR / filename
        except Exception as e:
            raise ValueError(f"Ошибка формирования пути MQL5: {e}")
    
    def get_log_path(self, log_type: str) -> Path:
        """Путь к файлу логов"""
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{log_type}_{timestamp}.log"
            return self.LOGS_DIR / filename
        except Exception as e:
            raise ValueError(f"Ошибка формирования пути логов: {e}")
    
    def validate_paths(self) -> Dict[str, bool]:
        """Проверка существования всех критических путей"""
        validation_results = {}
        
        critical_dirs = {
            "base": self.BASE_DIR,
            "data": self.DATA_DIR,
            "raw_data": self.RAW_DATA_DIR,
            "models": self.MODELS_DIR,
            "logs": self.LOGS_DIR
        }
        
        for name, path in critical_dirs.items():
            try:
                validation_results[name] = path.exists() and path.is_dir()
            except Exception as e:
                validation_results[name] = False
                print(f"Ошибка проверки {name}: {e}")
        
        return validation_results


# Глобальный экземпляр путей
PATHS = ProjectPaths()