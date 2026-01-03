#!/usr/bin/env python3
"""
Запуск всех проверок системы FuncXauusd

Выполняет:
    1. Аудит проекта (структура, зависимости, конфигурация)
    2. Юнит-тесты (test_full_pipeline.py, test_new_modules.py)
    3. Глубинную диагностику (логика всех компонентов)
"""

import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent


def run_command(cmd, description):
    """Запуск команды с обработкой ошибок"""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Ошибка выполнения: {e}")
        return False


def main():
    print("\n" + "="*70)
    print(" "*15 + "🚀 ПОЛНАЯ ПРОВЕРКА СИСТЕМЫ 🚀")
    print("="*70)
    
    results = {}
    
    print("\n[1/4] Проверка зависимостей...")
    results['dependencies'] = run_command(
        [sys.executable, '-m', 'pip', 'list'],
        "Установленные пакеты"
    )
    
    print("\n[2/4] Аудит проекта...")
    results['audit'] = run_command(
        [sys.executable, 'scripts/audit_project.py'],
        "Аудит структуры проекта"
    )
    
    print("\n[3/4] Юнит-тесты...")
    
    print("\n  3.1 Тестирование основного pipeline...")
    results['test_pipeline'] = run_command(
        [sys.executable, '-m', 'pytest', 'tests/test_full_pipeline.py', '-v', '--tb=short'],
        "tests/test_full_pipeline.py"
    )
    
    print("\n  3.2 Тестирование новых модулей...")
    results['test_new_modules'] = run_command(
        [sys.executable, '-m', 'pytest', 'tests/test_new_modules.py', '-v', '--tb=short'],
        "tests/test_new_modules.py"
    )
    
    print("\n[4/4] Глубинная диагностика...")
    results['diagnostics'] = run_command(
        [sys.executable, 'scripts/deep_diagnostics.py'],
        "Диагностика всех компонентов"
    )
    
    print("\n" + "="*70)
    print("  ИТОГОВЫЙ ОТЧЁТ")
    print("="*70)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print(f"\nПройдено проверок: {passed}/{total}\n")
    
    for check, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {check}")
    
    if passed == total:
        print("\n" + "="*70)
        print("  🎉 ВСЯ СИСТЕМА РАБОТАЕТ ОТЛИЧНО!")
        print("="*70)
        print("\nГотово к:")
        print("  • python scripts/train_cluster_model.py")
        print("  • python scripts/train_with_walk_forward.py --enable-walk-forward")
        return 0
    else:
        print("\n" + "="*70)
        print("  ⚠️ ОБНАРУЖЕНЫ ПРОБЛЕМЫ")
        print("="*70)
        print("\nПроверьте логи выше")
        return 1


if __name__ == "__main__":
    sys.exit(main())