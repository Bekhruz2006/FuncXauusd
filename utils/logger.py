"""
Централизованная система логирования с поддержкой различных уровней
и автоматической ротации файлов.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler
from datetime import datetime


class CustomFormatter(logging.Formatter):
    """Форматтер с цветовой разметкой для консоли"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m'   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        try:
            log_color = self.COLORS.get(record.levelname, self.RESET)
            record.levelname = f"{log_color}{record.levelname}{self.RESET}"
            return super().format(record)
        except Exception as e:
            return f"Ошибка форматирования лога: {e}"


class ProjectLogger:
    """Главный класс логирования проекта"""
    
    def __init__(self, name: str = "MetaOptimizer", log_dir: Optional[Path] = None):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        
        if log_dir is None:
            from config.paths import PATHS
            log_dir = PATHS.LOGS_DIR
        
        self.log_dir = log_dir
        self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Настройка обработчиков логов"""
        try:
            # Консольный обработчик с цветами
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_format = CustomFormatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_format)
            
            # Файловый обработчик для всех логов
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            all_logs_file = self.log_dir / f"all_{timestamp}.log"
            file_handler = RotatingFileHandler(
                all_logs_file,
                maxBytes=50*1024*1024,  # 50 MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            
            # Файловый обработчик только для ошибок
            error_logs_file = self.log_dir / f"errors_{timestamp}.log"
            error_handler = RotatingFileHandler(
                error_logs_file,
                maxBytes=20*1024*1024,  # 20 MB
                backupCount=3,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(file_format)
            
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(error_handler)
        except Exception as e:
                print(f"КРИТИЧЕСКАЯ ОШИБКА настройки логирования: {e}")
                sys.exit(1)

def debug(self, message: str, **kwargs) -> None:
    """Лог уровня DEBUG"""
    try:
        self.logger.debug(message, extra=kwargs)
    except Exception as e:
        print(f"Ошибка записи DEBUG лога: {e}")

def info(self, message: str, **kwargs) -> None:
    """Лог уровня INFO"""
    try:
        self.logger.info(message, extra=kwargs)
    except Exception as e:
        print(f"Ошибка записи INFO лога: {e}")

def warning(self, message: str, **kwargs) -> None:
    """Лог уровня WARNING"""
    try:
        self.logger.warning(message, extra=kwargs)
    except Exception as e:
        print(f"Ошибка записи WARNING лога: {e}")

def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
    """Лог уровня ERROR"""
    try:
        self.logger.error(message, exc_info=exc_info, extra=kwargs)
    except Exception as e:
        print(f"Ошибка записи ERROR лога: {e}")

def critical(self, message: str, exc_info: bool = True, **kwargs) -> None:
    """Лог уровня CRITICAL"""
    try:
        self.logger.critical(message, exc_info=exc_info, extra=kwargs)
    except Exception as e:
        print(f"Ошибка записи CRITICAL лога: {e}")

def log_experiment(self, stage: int, generation: int, metrics: dict) -> None:
    """Специализированный лог для экспериментов"""
    try:
        msg = f"Stage {stage} | Gen {generation} | Metrics: {metrics}"
        self.info(msg)
    except Exception as e:
        self.error(f"Ошибка логирования эксперимента: {e}")