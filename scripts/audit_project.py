#!/usr/bin/env python3

import sys
from pathlib import Path
import subprocess

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_imports():
    print("\n" + "="*70)
    print("  1. ПРОВЕРКА ИМПОРТОВ")
    print("="*70)
    
    modules = [
        ('src.data.loader', ['load_price_data', 'cache_prices']),
        ('src.features.engineering', ['create_features']),
        ('src.features.multiframe', ['create_multiframe_features']),
        ('src.labeling.strategies', ['get_labels_one_direction']),
        ('src.models.trainer', ['ClusterModelTrainer']),
        ('src.models.validator', ['validate_class_balance']),
        ('src.export.onnx_exporter', ['export_to_onnx']),
        ('src.backtesting.tester', ['test_model_one_direction']),
        ('src.risk.atr_manager', ['ATRRiskManager', 'calculate_atr']),
        ('src.validation.walk_forward', ['WalkForwardValidator']),
        ('src.monitoring.degradation', ['DegradationMonitor']),
    ]
    
    failed = []
    
    for module_name, functions in modules:
        try:
            module = __import__(module_name, fromlist=functions)
            for func in functions:
                if not hasattr(module, func):
                    failed.append(f"{module_name}.{func}")
                    print(f"  ✗ {module_name}.{func}")
                else:
                    print(f"  ✓ {module_name}.{func}")
        except ImportError as e:
            failed.append(module_name)
            print(f"  ✗ {module_name}: {e}")
    
    if failed:
        print(f"\n❌ Импорты не прошли: {len(failed)}")
        return False
    
    print(f"\n✅ Все импорты успешны")
    return True


def check_dependencies():
    print("\n" + "="*70)
    print("  2. ПРОВЕРКА ЗАВИСИМОСТЕЙ")
    print("="*70)
    
    required = [
        'numpy', 'pandas', 'scikit-learn', 'catboost',
        'numba', 'matplotlib', 'yaml', 'scipy'
    ]
    
    failed = []
    
    for package in required:
        try:
            if package == 'yaml':
                __import__('yaml')
            else:
                __import__(package.replace('-', '_'))
            print(f"  ✓ {package}")
        except ImportError:
            failed.append(package)
            print(f"  ✗ {package}")
    
    if failed:
        print(f"\n❌ Недостающие пакеты: {', '.join(failed)}")
        print(f"\nУстановите: pip install {' '.join(failed)}")
        return False
    
    print(f"\n✅ Все зависимости установлены")
    return True


def check_structure():
    print("\n" + "="*70)
    print("  3. ПРОВЕРКА СТРУКТУРЫ")
    print("="*70)
    
    required_dirs = [
        'src', 'src/data', 'src/features', 'src/labeling',
        'src/models', 'src/export', 'src/backtesting',
        'src/risk', 'src/validation', 'src/monitoring',
        'config', 'scripts', 'tests'
    ]
    
    required_files = [
        'config/training_config.yaml',
        'requirements.txt',
        'scripts/train_cluster_model.py',
        'tests/test_full_pipeline.py'
    ]
    
    missing = []
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            missing.append(str(dir_path))
            print(f"  ✗ {dir_path}/")
        else:
            print(f"  ✓ {dir_path}/")
    
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing.append(str(file_path))
            print(f"  ✗ {file_path}")
        else:
            print(f"  ✓ {file_path}")
    
    if missing:
        print(f"\n❌ Отсутствуют: {len(missing)}")
        return False
    
    print(f"\n✅ Структура проекта корректна")
    return True


def run_unit_tests():
    print("\n" + "="*70)
    print("  4. ЗАПУСК ЮНИТ-ТЕСТОВ")
    print("="*70)
    
    test_file = project_root / 'tests' / 'test_full_pipeline.py'
    
    if not test_file.exists():
        print("  ⚠️ Файл тестов не найден")
        return False
    
    try:
        result = subprocess.run(
            ['pytest', str(test_file), '-v', '--tb=short', '-x'],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        print(result.stdout)
        
        if result.returncode == 0:
            print(f"\n✅ Все тесты пройдены")
            return True
        else:
            print(f"\n❌ Тесты не прошли")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n⚠️ Тесты превысили таймаут (5 минут)")
        return False
    except Exception as e:
        print(f"\n❌ Ошибка запуска тестов: {e}")
        return False


def check_config():
    print("\n" + "="*70)
    print("  5. ПРОВЕРКА КОНФИГУРАЦИИ")
    print("="*70)
    
    import yaml
    
    config_path = project_root / 'config' / 'training_config.yaml'
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = [
            'symbol', 'trading', 'data', 'periods',
            'periods_meta', 'model', 'clustering',
            'validation', 'export'
        ]
        
        missing = []
        for key in required_keys:
            if key not in config:
                missing.append(key)
                print(f"  ✗ {key}")
            else:
                print(f"  ✓ {key}")
        
        if missing:
            print(f"\n❌ Отсутствуют ключи: {', '.join(missing)}")
            return False
        
        print(f"\n✅ Конфигурация валидна")
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка чтения конфигурации: {e}")
        return False


def generate_report(results: dict):
    print("\n" + "="*70)
    print("  ИТОГОВЫЙ ОТЧЁТ")
    print("="*70)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print(f"\nПройдено проверок: {passed}/{total}")
    print()
    
    for check, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {check}")
    
    print()
    
    if passed == total:
        print("🎉 ПРОЕКТ ГОТОВ К ОБУЧЕНИЮ!")
        print()
        print("Следующий шаг:")
        print("  python scripts/train_cluster_model.py")
        return True
    else:
        print("⚠️ ПРОЕКТ ТРЕБУЕТ ИСПРАВЛЕНИЙ")
        print()
        print("Исправьте ошибки выше и запустите аудит снова:")
        print("  python scripts/audit_project.py")
        return False


def main():
    print("\n" + "="*70)
    print(" "*15 + "🔍 ПОЛНЫЙ АУДИТ ПРОЕКТА 🔍")
    print("="*70)
    
    results = {
        'Импорты': check_imports(),
        'Зависимости': check_dependencies(),
        'Структура': check_structure(),
        'Конфигурация': check_config(),
        'Юнит-тесты': run_unit_tests()
    }
    
    success = generate_report(results)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())