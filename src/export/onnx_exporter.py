"""
Экспорт моделей в ONNX и генерация MQL5 кода

Генерируемые файлы:
    1. catmodel_SYMBOL_N.onnx - основная модель
    2. catmodel_m_SYMBOL_N.onnx - мета-модель
    3. SYMBOL_ONNX_include_N.mqh - include файл для MT5
"""

import re
from pathlib import Path
from typing import List
from catboost import CatBoostClassifier


def export_to_onnx(
    model_main: CatBoostClassifier,
    model_meta: CatBoostClassifier,
    config: dict,
    r2_score: float,
    model_number: int = 0
) -> None:
    """
    Полный экспорт системы для MetaTrader 5
    
    Args:
        model_main: Обученная основная модель
        model_meta: Обученная мета-модель
        config: Конфигурация с параметрами
        r2_score: R² score модели (для логирования)
        model_number: Номер модели (если несколько)
    
    Side Effects:
        Создает файлы в директории export_path:
            - 2 ONNX файла
            - 1 MQL include файл
    """
    # Параметры
    symbol = config['symbol']['name']
    periods = config['periods']
    periods_meta = config['periods_meta']
    export_path = Path(config['export']['paths']['onnx'])
    
    # Создание директории
    export_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Экспорт в MetaTrader 5...")
    print(f"  Symbol: {symbol}")
    print(f"  R² Score: {r2_score:.4f}")
    print(f"  Model Number: {model_number}")
    
    # 1. Экспорт ONNX моделей
    _export_onnx_models(
        model_main, model_meta,
        symbol, model_number,
        export_path
    )
    
    # 2. Генерация MQL include файла
    _generate_mql_include(
        symbol, model_number,
        periods, periods_meta,
        export_path
    )
    
    print(f"\n  ✅ Экспорт завершен!")
    print(f"  📁 Файлы: {export_path}")
    print(f"\n  📋 Следующие шаги:")
    print(f"    1. Скопировать *.onnx в: MQL5/Experts/Files/")
    print(f"    2. Скопировать *.mqh в: MQL5/Include/")
    print(f"    3. Перекомпилировать советник в MT5")


def _export_onnx_models(
    model_main: CatBoostClassifier,
    model_meta: CatBoostClassifier,
    symbol: str,
    model_number: int,
    export_path: Path
) -> None:
    """
    Экспорт двух моделей в формат ONNX
    
    ONNX параметры:
        - domain: ai.catboost
        - opset_version: 12 (совместим с MT5)
        - doc_string: описание модели
    """
    # Параметры ONNX экспорта
    onnx_params = {
        'onnx_domain': 'ai.catboost',
        'onnx_model_version': 1,
        'onnx_graph_name': 'CatBoostModel'
    }
    
    # === Main Model ===
    main_filename = f"catmodel {symbol} {model_number}.onnx"
    main_path = export_path / main_filename
    
    model_main.save_model(
        str(main_path),
        format="onnx",
        export_parameters={
            **onnx_params,
            'onnx_doc_string': 'Main trading model (std features)'
        },
        pool=None
    )
    print(f"  ✓ Main model: {main_filename}")
    
    # === Meta Model ===
    meta_filename = f"catmodel_m {symbol} {model_number}.onnx"
    meta_path = export_path / meta_filename
    
    model_meta.save_model(
        str(meta_path),
        format="onnx",
        export_parameters={
            **onnx_params,
            'onnx_doc_string': 'Meta filter model (skewness features)'
        },
        pool=None
    )
    print(f"  ✓ Meta model: {meta_filename}")


def _generate_mql_include(
    symbol: str,
    model_number: int,
    periods: List[int],
    periods_meta: List[int],
    export_path: Path
) -> None:
    """
    Генерация MQL5 include файла
    
    Файл содержит:
        1. #resource директивы для загрузки ONNX
        2. Массивы периодов для признаков
        3. Функции расчета признаков (fill_arays)
    """
    code_lines = []
    
    # === HEADER ===
    code_lines.extend([
        "// Auto-generated ONNX include file",
        f"// Symbol: {symbol}",
        f"// Model: {model_number}",
        "// DO NOT EDIT MANUALLY",
        "",
        "#include <Math\\Stat\\Math.mqh>",
        ""
    ])
    
    # === RESOURCE DIRECTIVES ===
    code_lines.extend([
        f"#resource \"catmodel {symbol} {model_number}.onnx\" as uchar ExtModel_{symbol}_{model_number}[]",
        f"#resource \"catmodel_m {symbol} {model_number}.onnx\" as uchar ExtModel2_{symbol}_{model_number}[]",
        ""
    ])
    
    # === PERIOD ARRAYS ===
    code_lines.extend([
        f"int Periods{symbol}_{model_number}[{len(periods)}] = {{{','.join(map(str, periods))}}};",
        f"int Periods_m{symbol}_{model_number}[{len(periods_meta)}] = {{{','.join(map(str, periods_meta))}}};",
        ""
    ])
    
    # === MAIN FEATURES FUNCTION (STD) ===
    code_lines.extend([
        f"void fill_arays{symbol}_{model_number}(double &features[]) {{",
        "   double pr[], ret[];",
        "   ArrayResize(ret, 1);",
        f"   for(int i=ArraySize(Periods{symbol}_{model_number})-1; i>=0; i--) {{",
        f"       CopyClose(NULL, PERIOD_H1, 1, Periods{symbol}_{model_number}[i], pr);",
        "       ret[0] = MathStandardDeviation(pr);",  # STD для main модели
        "       ArrayInsert(features, ret, ArraySize(features), 0, WHOLE_ARRAY);",
        "   }",
        "   ArraySetAsSeries(features, true);",
        "}",
        ""
    ])
    
    # === META FEATURES FUNCTION (SKEWNESS) ===
    code_lines.extend([
        f"void fill_arays_m{symbol}_{model_number}(double &features[]) {{",
        "   double pr[], ret[];",
        "   ArrayResize(ret, 1);",
        f"   for(int i=ArraySize(Periods_m{symbol}_{model_number})-1; i>=0; i--) {{",
        f"       CopyClose(NULL, PERIOD_H1, 1, Periods_m{symbol}_{model_number}[i], pr);",
        "       ret[0] = MathSkewness(pr);",  # Skewness для meta модели
        "       ArrayInsert(features, ret, ArraySize(features), 0, WHOLE_ARRAY);",
        "   }",
        "   ArraySetAsSeries(features, true);",
        "}",
        ""
    ])
    
    # === SAVE FILE ===
    filename = f"{symbol} ONNX include {model_number}.mqh"
    filepath = export_path / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(code_lines))
    
    print(f"  ✓ MQL include: {filename}")
    print(f"    • Main features: {len(periods)} (std)")
    print(f"    • Meta features: {len(periods_meta)} (skewness)")


# === АЛЬТЕРНАТИВНЫЕ ФОРМАТЫ ЭКСПОРТА ===

def export_to_cpp(
    model_main: CatBoostClassifier,
    model_meta: CatBoostClassifier,
    config: dict,
    model_number: int = 0
) -> None:
    """
    Экспорт в C++ код (для встраивания в MQL5)
    
    Note:
        Генерирует огромный файл. ONNX предпочтительнее.
        Используйте только если ONNX не работает.
    """
    symbol = config['symbol']['name']
    export_path = Path(config['export']['paths']['onnx'])
    export_path.mkdir(parents=True, exist_ok=True)
    
    # Main model
    main_cpp = export_path / f"catmodel_{symbol}_{model_number}.h"
    model_main.save_model(
        str(main_cpp),
        format="cpp",
        export_parameters=None,
        pool=None
    )
    
    # Meta model
    meta_cpp = export_path / f"catmodel_m_{symbol}_{model_number}.h"
    model_meta.save_model(
        str(meta_cpp),
        format="cpp",
        export_parameters=None,
        pool=None
    )
    
    print(f"  ✓ C++ export: {main_cpp.name}, {meta_cpp.name}")
    print(f"  ⚠️ Warning: Файлы могут быть очень большими")


def export_to_python(
    model_main: CatBoostClassifier,
    model_meta: CatBoostClassifier,
    config: dict,
    model_number: int = 0
) -> None:
    """
    Экспорт в нативный формат CatBoost (.cbm)
    
    Использование:
        >>> from catboost import CatBoostClassifier
        >>> model = CatBoostClassifier()
        >>> model.load_model('model.cbm')
    """
    symbol = config['symbol']['name']
    export_path = Path(config['export']['paths']['models'])
    export_path.mkdir(parents=True, exist_ok=True)
    
    # Main model
    main_cbm = export_path / f"main_{symbol}_{model_number}.cbm"
    model_main.save_model(str(main_cbm))
    
    # Meta model
    meta_cbm = export_path / f"meta_{symbol}_{model_number}.cbm"
    model_meta.save_model(str(meta_cbm))
    
    print(f"  ✓ CBM export: {main_cbm.name}, {meta_cbm.name}")


# === УТИЛИТЫ ===

def validate_onnx_export(onnx_path: Path) -> bool:
    """
    Проверка корректности ONNX файла
    
    Args:
        onnx_path: Путь к ONNX файлу
    
    Returns:
        bool: True если файл валиден
    """
    try:
        import onnx
        
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        
        print(f"  ✓ ONNX valid: {onnx_path.name}")
        return True
        
    except Exception as e:
        print(f"  ✗ ONNX invalid: {e}")
        return False


def get_onnx_input_shape(onnx_path: Path) -> tuple:
    """
    Извлечение размерности входа из ONNX модели
    
    Returns:
        tuple: (batch_size, n_features)
    """
    try:
        import onnx
        
        model = onnx.load(str(onnx_path))
        input_tensor = model.graph.input[0]
        
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        return tuple(shape)
        
    except Exception as e:
        print(f"  ⚠️ Cannot extract shape: {e}")
        return None


def create_export_readme(export_path: Path,
                        symbol: str,
                        model_number: int,
                        r2_score: float) -> None:
    """
    Создание README файла с инструкциями по установке
    """
    readme_text = f"""
# Model Export for {symbol}

## Files Generated
1. `catmodel {symbol} {model_number}.onnx` - Main trading model
2. `catmodel_m {symbol} {model_number}.onnx` - Meta filter model
3. `{symbol} ONNX include {model_number}.mqh` - MQL5 include file

## Performance
- R² Score: {r2_score:.4f}
- Model Number: {model_number}

## Installation Steps

### 1. Copy ONNX Models
```
MetaTrader 5/
└── MQL5/
    └── Experts/
        └── Files/
            ├── catmodel {symbol} {model_number}.onnx
            └── catmodel_m {symbol} {model_number}.onnx
```

### 2. Copy Include File
```
MetaTrader 5/
└── MQL5/
    └── Include/
        └── {symbol} ONNX include {model_number}.mqh
```

### 3. Update Expert Advisor
In your .mq5 file, add:
```cpp
#include <{symbol} ONNX include {model_number}.mqh>
```

### 4. Update Model Sizes
In `OnInit()`:
```cpp
int total_main_features = ArraySize(Periods{symbol}_{model_number});
int total_meta_features = ArraySize(Periods_m{symbol}_{model_number});

const ulong ExtInputShape[] = {{1, (ulong)total_main_features}};
const ulong ExtInputShape2[] = {{1, (ulong)total_meta_features}};
```

### 5. Recompile
Press F7 in MetaEditor to recompile the Expert Advisor.

## Verification
Check MetaTrader 5 Experts tab for:
- "ONNX models loaded successfully"
- No errors about missing files or wrong dimensions

## Troubleshooting
- **"File not found"**: Check paths in steps 1-2
- **"Wrong input shape"**: Verify ExtInputShape matches periods array size
- **"Model prediction failed"**: Ensure features are calculated correctly
"""
    
    readme_path = export_path / f"README_{symbol}_{model_number}.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_text)
    
    print(f"  ✓ README: {readme_path.name}")