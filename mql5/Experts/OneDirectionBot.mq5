//+------------------------------------------------------------------+
//|                                           OneDirectionBot.mq5     |
//|                     Copyright 2025, Trading Systems Engineering  |
//|                                    https://github.com/FuncXauusd |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Trading Systems Engineering"
#property link      "https://github.com/FuncXauusd"
#property version   "1.00"
#property strict

//+------------------------------------------------------------------+
//| INCLUDE FILES                                                     |
//+------------------------------------------------------------------+
// CRITICAL: Измените на правильный include файл после экспорта
#include <XAUUSD_H1 ONNX include 0.mqh>
#include <Trade\Trade.mqh>
#include <Trade\AccountInfo.mqh>

//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                  |
//+------------------------------------------------------------------+
CTrade mytrade;
CPositionInfo myposition;

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                  |
//+------------------------------------------------------------------+
input group "=== Trading Direction ==="
input bool direction = true;                    // True = Buy, False = Sell

input group "=== Model Thresholds ==="
input double main_threshold = 0.5;              // Main model signal threshold
input double meta_threshold = 0.5;              // Meta model filter threshold

input group "=== Risk Management ==="
sinput double MaximumRisk = 0.001;              // Progressive lot coefficient
sinput double ManualLot = 0.01;                 // Fixed lot (0 = auto)
input int stoploss = 2000;                      // Stop Loss (points)
input int takeprofit = 500;                     // Take Profit (points)

input group "=== Position Management ==="
input int max_orders = 3;                       // Max concurrent positions
input int orders_time_delay = 5;                // Hours between positions
input int max_spread = 25;                      // Maximum spread (points)

input group "=== System ==="
sinput ulong OrderMagic = 57633493;             // Order magic number
input string bot_comment = "FuncXauusd v1.0";   // Order comment

//+------------------------------------------------------------------+
//| GLOBAL STATE                                                      |
//+------------------------------------------------------------------+
static datetime last_time = 0;
long ExtHandle = INVALID_HANDLE;
long ExtHandle2 = INVALID_HANDLE;

// Макросы для удобства
#define Ask SymbolInfoDouble(_Symbol, SYMBOL_ASK)
#define Bid SymbolInfoDouble(_Symbol, SYMBOL_BID)

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
    Comment("FuncXauusd v1.0 | Telegram: @dmitrievskyai");
    mytrade.SetExpertMagicNumber(OrderMagic);
    
    // === КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ ===
    // Размеры массивов должны совпадать с фактическим количеством периодов
    int total_main_features = ArraySize(PeriodsXAUUSD_H1_0);   // ПРАВИЛЬНО
    int total_meta_features = ArraySize(Periods_mXAUUSD_H1_0); // ПРАВИЛЬНО
    
    Print("Initializing ONNX models...");
    Print("  Main features: ", total_main_features);
    Print("  Meta features: ", total_meta_features);
    
    // Создание input shape для моделей
    const ulong ExtInputShape[] = {1, (ulong)total_main_features};
    const ulong ExtInputShape2[] = {1, (ulong)total_meta_features};
    
    // Загрузка ONNX моделей из ресурсов
    ExtHandle = OnnxCreateFromBuffer(ExtModel_XAUUSD_H1_0, ONNX_DEFAULT);
    ExtHandle2 = OnnxCreateFromBuffer(ExtModel2_XAUUSD_H1_0, ONNX_DEFAULT);
    
    if(ExtHandle == INVALID_HANDLE || ExtHandle2 == INVALID_HANDLE)
    {
        Print("ERROR: ONNX models failed to load! Error: ", GetLastError());
        return(INIT_FAILED);
    }
    
    // Установка input shape для главной модели
    if(!OnnxSetInputShape(ExtHandle, 0, ExtInputShape))
    {
        Print("ERROR: Cannot set input shape for main model! Error: ", GetLastError());
        OnnxRelease(ExtHandle);
        return(INIT_FAILED);
    }
    
    // Установка input shape для мета-модели
    if(!OnnxSetInputShape(ExtHandle2, 0, ExtInputShape2))
    {
        Print("ERROR: Cannot set input shape for meta model! Error: ", GetLastError());
        OnnxRelease(ExtHandle2);
        return(INIT_FAILED);
    }
    
    // Установка output shape (одно значение - вероятность класса 1)
    const ulong output_shape[] = {1};
    
    if(!OnnxSetOutputShape(ExtHandle, 0, output_shape))
    {
        Print("ERROR: Cannot set output shape for main model! Error: ", GetLastError());
        return(INIT_FAILED);
    }
    
    if(!OnnxSetOutputShape(ExtHandle2, 0, output_shape))
    {
        Print("ERROR: Cannot set output shape for meta model! Error: ", GetLastError());
        return(INIT_FAILED);
    }
    
    Print("✓ ONNX models loaded successfully!");
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    OnnxRelease(ExtHandle);
    OnnxRelease(ExtHandle2);
    Comment("");
    
    Print("Bot stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
    // Работаем только на новых барах
    if(!isNewBar())
        return;
    
    // === РАСЧЕТ ПРИЗНАКОВ ===
    double features[], features_m[];
    ArrayResize(features, 0);
    ArrayResize(features_m, 0);
    
    // Вызов функций из include файла
    fill_araysXAUUSD_H1_0(features);
    fill_arays_mXAUUSD_H1_0(features_m);
    
    // === КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ ===
    // Создаем массивы правильного размера для передачи в ONNX
    int main_size = ArraySize(PeriodsXAUUSD_H1_0);
    int meta_size = ArraySize(Periods_mXAUUSD_H1_0);
    
    double f[];
    double f_m[];
    ArrayResize(f, main_size);
    ArrayResize(f_m, meta_size);
    
    // Копируем данные
    for(int i = 0; i < main_size; i++)
        f[i] = features[i];
    
    for(int i = 0; i < meta_size; i++)
        f_m[i] = features_m[i];
    
    // Проверка размеров (для отладки)
    if(ArraySize(features) != main_size)
    {
        Print("ERROR: Main features size mismatch! Expected: ", main_size, 
              ", Got: ", ArraySize(features));
        return;
    }
    
    if(ArraySize(features_m) != meta_size)
    {
        Print("ERROR: Meta features size mismatch! Expected: ", meta_size,
              ", Got: ", ArraySize(features_m));
        return;
    }
    
    // === ПРОГНОЗЫ МОДЕЛЕЙ ===
    static vector out(1), out_meta(1);
    
    struct output
    {
        long label[];
        float proba[];
    };
    
    output out2[], out2_meta[];
    
    // Запуск ONNX моделей
    if(!OnnxRun(ExtHandle, ONNX_NO_CONVERSION, f, out, out2))
    {
        Print("ERROR: Main model prediction failed!");
        return;
    }
    
    if(!OnnxRun(ExtHandle2, ONNX_NO_CONVERSION, f_m, out_meta, out2_meta))
    {
        Print("ERROR: Meta model prediction failed!");
        return;
    }
    
    // Извлечение вероятностей
    double sig = out2[0].proba[1];        // Вероятность класса 1 (торговать)
    double meta_sig = out2_meta[0].proba[1];  // Вероятность класса 1 (разрешить)
    
    // === ЗАКРЫТИЕ ПОЗИЦИЙ ПО СИГНАЛАМ ===
    if((Ask - Bid) < max_spread * _Point)
    {
        if(countOrders(OrderMagic) > 0)
        {
            for(int b = PositionsTotal() - 1; b >= 0; b--)
            {
                if(PositionGetSymbol(b) == _Symbol)
                {
                    // Закрытие BUY позиций
                    if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY &&
                       PositionGetInteger(POSITION_MAGIC) == OrderMagic &&
                       sig < main_threshold && direction)
                    {
                        double freeze_level = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL);
                        double price_diff = MathAbs(Bid - PositionGetDouble(POSITION_PRICE_OPEN)) / _Point;
                        
                        if(freeze_level < price_diff)
                        {
                            if(!mytrade.PositionClose(_Symbol))
                            {
                                Print("ERROR: Cannot close BUY position! Error: ", GetLastError());
                            }
                        }
                    }
                    
                    // Закрытие SELL позиций
                    if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL &&
                       PositionGetInteger(POSITION_MAGIC) == OrderMagic &&
                       sig < main_threshold && !direction)
                    {
                        double freeze_level = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL);
                        double price_diff = MathAbs(Bid - PositionGetDouble(POSITION_PRICE_OPEN)) / _Point;
                        
                        if(freeze_level < price_diff)
                        {
                            if(!mytrade.PositionClose(_Symbol))
                            {
                                Print("ERROR: Cannot close SELL position! Error: ", GetLastError());
                            }
                        }
                    }
                }
            }
        }
    }
    
    // === ОТКРЫТИЕ НОВЫХ ПОЗИЦИЙ ===
    if((Ask - Bid) < max_spread * _Point && 
       meta_sig > meta_threshold &&
       AllowTrade(OrderMagic))
    {
        if(countOrders(OrderMagic) < max_orders)
        {
            double l = LotsOptimized();
            
            if(CheckMoneyForTrade(_Symbol, l, ORDER_TYPE_BUY))
            {
                // BUY сигнал
                if(sig > main_threshold && direction)
                {
                    double stop = Bid - stoploss * _Point;
                    double take = Ask + takeprofit * _Point;
                    
                    if(!mytrade.PositionOpen(_Symbol, ORDER_TYPE_BUY, l, Ask, stop, take, bot_comment))
                    {
                        Print("ERROR: Cannot open BUY! Error: ", GetLastError());
                    }
                    else
                    {
                        Print("✓ BUY opened: ", l, " lots @ ", Ask, 
                              " | SL: ", stop, " | TP: ", take);
                    }
                }
                // SELL сигнал
                else if(sig > main_threshold && !direction)
                {
                    double stop = Ask + stoploss * _Point;
                    double take = Bid - takeprofit * _Point;
                    
                    if(!mytrade.PositionOpen(_Symbol, ORDER_TYPE_SELL, l, Bid, stop, take, bot_comment))
                    {
                        Print("ERROR: Cannot open SELL! Error: ", GetLastError());
                    }
                    else
                    {
                        Print("✓ SELL opened: ", l, " lots @ ", Bid,
                              " | SL: ", stop, " | TP: ", take);
                    }
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Подсчет открытых позиций                                         |
//+------------------------------------------------------------------+
int countOrders(ulong magic)
{
    int count = 0;
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(PositionGetSymbol(i) == _Symbol)
        {
            if(PositionGetInteger(POSITION_MAGIC) == magic)
            {
                count++;
            }
        }
    }
    return count;
}

//+------------------------------------------------------------------+
//| Проверка разрешения на торговлю                                  |
//+------------------------------------------------------------------+
bool AllowTrade(ulong magic)
{
    if(countOrders(OrderMagic) == 0)
        return true;
    
    datetime last_pos = 0;
    
    if(countOrders(OrderMagic) != 0)
    {
        for(int b = PositionsTotal() - 1; b >= 0; b--)
        {
            if(PositionGetSymbol(b) == _Symbol)
            {
                if(PositionGetInteger(POSITION_MAGIC) == magic)
                {
                    if(PositionGetInteger(POSITION_TIME) > last_pos)
                        last_pos = datetime(PositionGetInteger(POSITION_TIME));
                }
            }
        }
        
        datetime time[];
        CopyTime(_Symbol, PERIOD_H1, 0, 1, time);
        
        if(time[0] > last_pos + 3600 * orders_time_delay)
            return true;
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| Расчет оптимального лота                                         |
//+------------------------------------------------------------------+
double LotsOptimized()
{
    double lot;
    
    // В режиме оптимизации - минимальный лот
    if(MQLInfoInteger(MQL_OPTIMIZATION) == true)
    {
        lot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
        return lot;
    }
    
    CAccountInfo myaccount;
    
    // Прогрессивный лот на основе свободной маржи
    lot = NormalizeDouble(myaccount.FreeMargin() * MaximumRisk / 1000.0, 2);
    
    // Или фиксированный лот
    if(ManualLot != 0.0)
        lot = ManualLot;
    
    // Нормализация по шагу объема
    double volume_step = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP);
    int ratio = (int)MathRound(lot / volume_step);
    
    if(MathAbs(ratio * volume_step - lot) > 0.0000001)
        lot = ratio * volume_step;
    
    // Ограничения брокера
    if(lot < SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN))
        lot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
    
    if(lot > SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX))
        lot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX);
    
    return lot;
}

//+------------------------------------------------------------------+
//| Проверка нового бара                                             |
//+------------------------------------------------------------------+
bool isNewBar()
{
    datetime lastbar_time = datetime(SeriesInfoInteger(Symbol(), PERIOD_CURRENT, SERIES_LASTBAR_DATE));
    
    if(last_time == 0)
    {
        last_time = lastbar_time;
        return false;
    }
    
    if(last_time != lastbar_time)
    {
        last_time = lastbar_time;
        return true;
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| Проверка достаточности средств                                   |
//+------------------------------------------------------------------+
bool CheckMoneyForTrade(string symb, double lots, ENUM_ORDER_TYPE type)
{
    MqlTick mqltick;
    SymbolInfoTick(symb, mqltick);
    
    double price = mqltick.ask;
    if(type == ORDER_TYPE_SELL)
        price = mqltick.bid;
    
    double margin;
    double free_margin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
    
    if(!OrderCalcMargin(type, symb, lots, price, margin))
    {
        Print("ERROR: Cannot calculate margin! Error: ", GetLastError());
        return false;
    }
    
    if(margin > free_margin)
    {
        Print("ERROR: Not enough money! Required: ", margin, " | Available: ", free_margin);
        return false;
    }
    
    return true;
}
//+------------------------------------------------------------------+