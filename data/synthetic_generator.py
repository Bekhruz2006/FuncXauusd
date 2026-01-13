"""
Генератор синтетических данных с использованием условного GAN (cGAN).
Генерирует реалистичные траектории цен с сохранением статистических свойств.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from scipy import stats

from config.hyperparameters import HYPERPARAMS
from config.market_regimes import RegimeDefinitions
from utils.logger import LOGGER


class FinancialTimeSeriesDataset(Dataset):
    """Dataset для обучения GAN на финансовых данных"""
    
    def __init__(self, 
                 data: np.ndarray,
                 conditions: np.ndarray,
                 sequence_length: int = 100):
        
        self.data = torch.FloatTensor(data)
        self.conditions = torch.FloatTensor(conditions)
        self.sequence_length = sequence_length
        
        if len(self.data) < sequence_length:
            raise ValueError(f"Недостаточно данных: {len(self.data)} < {sequence_length}")
    
    def __len__(self) -> int:
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            sequence = self.data[idx:idx + self.sequence_length]
            condition = self.conditions[idx]
            return sequence, condition
        except Exception as e:
            LOGGER.error(f"Ошибка получения элемента датасета: {e}")
            return torch.zeros(self.sequence_length, self.data.shape[1]), torch.zeros(self.conditions.shape[1])


class Generator(nn.Module):
    """Генератор cGAN для временных рядов"""
    
    def __init__(self, 
                 latent_dim: int,
                 condition_dim: int,
                 output_dim: int,
                 sequence_length: int,
                 hidden_dims: List[int] = [256, 512, 256]):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        
        input_dim = latent_dim + condition_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim * sequence_length))
        layers.append(nn.Tanh())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        try:
            x = torch.cat([noise, condition], dim=1)
            output = self.model(x)
            output = output.view(-1, self.sequence_length, self.output_dim)
            return output
        except Exception as e:
            LOGGER.error(f"Ошибка forward pass Generator: {e}")
            return torch.zeros(noise.size(0), self.sequence_length, self.output_dim)


class Discriminator(nn.Module):
    """Дискриминатор cGAN для временных рядов"""
    
    def __init__(self,
                 input_dim: int,
                 condition_dim: int,
                 sequence_length: int,
                 hidden_dims: List[int] = [256, 512, 256]):
        super(Discriminator, self).__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.sequence_length = sequence_length
        
        # Обработка последовательности
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Полносвязные слои с условием
        fc_input_dim = 128 + condition_dim
        
        layers = []
        prev_dim = fc_input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.fc = nn.Sequential(*layers)
    
    def forward(self, sequence: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        try:
            # LSTM обработка
            lstm_out, (h_n, c_n) = self.lstm(sequence)
            lstm_features = h_n[-1]  # Берем выход последнего слоя
            
            # Конкатенация с условием
            x = torch.cat([lstm_features, condition], dim=1)
            
            # Классификация
            output = self.fc(x)
            return output
            
        except Exception as e:
            LOGGER.error(f"Ошибка forward pass Discriminator: {e}")
            return torch.zeros(sequence.size(0), 1)


class SyntheticDataGenerator:
    """Генератор синтетических финансовых данных с cGAN"""
    
    def __init__(self,
                 latent_dim: Optional[int] = None,
                 sequence_length: int = 100,
                 batch_size: Optional[int] = None,
                 num_epochs: Optional[int] = None,
                 learning_rate_g: Optional[float] = None,
                 learning_rate_d: Optional[float] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.latent_dim = latent_dim or HYPERPARAMS.synthetic.latent_dim
        self.sequence_length = sequence_length
        self.batch_size = batch_size or HYPERPARAMS.synthetic.batch_size
        self.num_epochs = num_epochs or HYPERPARAMS.synthetic.num_epochs
        self.learning_rate_g = learning_rate_g or HYPERPARAMS.synthetic.generator_lr
        self.learning_rate_d = learning_rate_d or HYPERPARAMS.synthetic.discriminator_lr
        self.device = device
        
        self.generator: Optional[Generator] = None
        self.discriminator: Optional[Discriminator] = None
        self.condition_dim = HYPERPARAMS.synthetic.condition_dim
        
        self.training_history: Dict[str, List[float]] = {
            'g_loss': [],
            'd_loss': [],
            'd_real_acc': [],
            'd_fake_acc': []
        }
        
        LOGGER.info(f"Инициализация cGAN: device={device}, latent_dim={self.latent_dim}, "
                   f"epochs={self.num_epochs}")
    
    def prepare_training_data(self, 
                              df: pd.DataFrame,
                              regime_features: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготовка данных для обучения GAN.
        
        Args:
            df: DataFrame с предобработанными данными
            regime_features: Список признаков для условного вектора
        
        Returns:
            (data_array, condition_array)
        """
        try:
            if regime_features is None:
                regime_features = ['volatility_percentile', 'trend_strength', 'mean_reversion_idx']
            
            # Нормализация OHLCV
            price_features = ['Open', 'High', 'Low', 'Close']
            volume_feature = 'Volume'
            
            data_list = []
            
            for feature in price_features:
                if feature in df.columns:
                    normalized = (df[feature] - df[feature].mean()) / (df[feature].std() + 1e-8)
                    data_list.append(normalized.values)
            
            if volume_feature in df.columns:
                vol_normalized = (df[volume_feature] - df[volume_feature].mean()) / (df[volume_feature].std() + 1e-8)
                data_list.append(vol_normalized.values)
            
            data_array = np.column_stack(data_list)
            
            # Подготовка условного вектора
            condition_list = []
            
            for feature in regime_features:
                if feature in df.columns:
                    condition_list.append(df[feature].values)
                else:
                    # Вычисляем если отсутствует
                    if feature == 'volatility_percentile':
                        from config.market_regimes import compute_volatility_percentile
                        returns = df['Close'].pct_change().dropna().values
                        vol_pct = compute_volatility_percentile(returns, window=50)
                        condition_list.append(np.full(len(df), vol_pct))
                    else:
                        condition_list.append(np.zeros(len(df)))
            
            # Добавляем синтетические условия до condition_dim
            while len(condition_list) < self.condition_dim:
                condition_list.append(np.random.randn(len(df)) * 0.1)
            
            condition_array = np.column_stack(condition_list[:self.condition_dim])
            
            LOGGER.info(f"Подготовлено данных для обучения: shape={data_array.shape}, "
                       f"condition_shape={condition_array.shape}")
            
            return data_array, condition_array
            
        except Exception as e:
            LOGGER.error(f"Ошибка подготовки данных для GAN: {e}", exc_info=True)
            raise
    
    def train(self, 
              data_array: np.ndarray,
              condition_array: np.ndarray,
              validation_split: float = 0.1) -> Dict[str, List[float]]:
        """
        Обучение cGAN.
        
        Args:
            data_array: Массив данных (N, features)
            condition_array: Массив условий (N, condition_dim)
            validation_split: Доля данных для валидации
        
        Returns:
            История обучения
        """
        try:
            # Разделение на train/val
            split_idx = int(len(data_array) * (1 - validation_split))
            train_data = data_array[:split_idx]
            train_conditions = condition_array[:split_idx]
            
            # Создание Dataset и DataLoader
            dataset = FinancialTimeSeriesDataset(
                train_data, train_conditions, self.sequence_length
            )
            dataloader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
            )
            
            # Инициализация моделей
            output_dim = data_array.shape[1]
            
            self.generator = Generator(
                latent_dim=self.latent_dim,
                condition_dim=self.condition_dim,
                output_dim=output_dim,
                sequence_length=self.sequence_length
            ).to(self.device)
            
            self.discriminator = Discriminator(
                input_dim=output_dim,
                condition_dim=self.condition_dim,
                sequence_length=self.sequence_length
            ).to(self.device)
            
            # Оптимизаторы
            optimizer_g = optim.Adam(
                self.generator.parameters(),
                lr=self.learning_rate_g,
                betas=(0.5, 0.999)
            )
            optimizer_d = optim.Adam(
                self.discriminator.parameters(),
                lr=self.learning_rate_d,
                betas=(0.5, 0.999)
            )
            
            # Функция потерь
            criterion = nn.BCELoss()
            
            LOGGER.info(f"Начало обучения cGAN: {self.num_epochs} эпох")
            
            # Обучение
            for epoch in range(self.num_epochs):
                epoch_g_loss = 0.0
                epoch_d_loss = 0.0
                epoch_d_real_acc = 0.0
                epoch_d_fake_acc = 0.0
                num_batches = 0
                
                for batch_idx, (real_sequences, conditions) in enumerate(dataloader):
                    try:
                        real_sequences = real_sequences.to(self.device)
                        conditions = conditions.to(self.device)
                        batch_size = real_sequences.size(0)
                        
                        # Метки
                        real_labels = torch.ones(batch_size, 1).to(self.device)
                        fake_labels = torch.zeros(batch_size, 1).to(self.device)
                        
                        # ========== Обучение Discriminator ==========
                        optimizer_d.zero_grad()
                        
                        # Реальные данные
                        d_real = self.discriminator(real_sequences, conditions)
                        d_real_loss = criterion(d_real, real_labels)
                        
                        # Фейковые данные
                        noise = torch.randn(batch_size, self.latent_dim).to(self.device)
                        fake_sequences = self.generator(noise, conditions)
                        d_fake = self.discriminator(fake_sequences.detach(), conditions)
                        d_fake_loss = criterion(d_fake, fake_labels)
                        
                        # Общая потеря discriminator
                        d_loss = d_real_loss + d_fake_loss
                        d_loss.backward()
                        optimizer_d.step()
                        
                        # ========== Обучение Generator ==========
                        optimizer_g.zero_grad()
                        
                        noise = torch.randn(batch_size, self.latent_dim).to(self.device)
                        fake_sequences = self.generator(noise, conditions)
                        d_fake = self.discriminator(fake_sequences, conditions)
                        g_loss = criterion(d_fake, real_labels)
                        
                        g_loss.backward()
                        optimizer_g.step()
                        
                        # Метрики
                        epoch_g_loss += g_loss.item()
                        epoch_d_loss += d_loss.item()
                        epoch_d_real_acc += (d_real > 0.5).float().mean().item()
                        epoch_d_fake_acc += (d_fake < 0.5).float().mean().item()
                        num_batches += 1
                        
                    except Exception as e:
                        LOGGER.error(f"Ошибка в батче {batch_idx}: {e}")
                        continue
                
                # Усреднение метрик
                if num_batches > 0:
                    avg_g_loss = epoch_g_loss / num_batches
                    avg_d_loss = epoch_d_loss / num_batches
                    avg_d_real_acc = epoch_d_real_acc / num_batches
                    avg_d_fake_acc = epoch_d_fake_acc / num_batches
                    
                    self.training_history['g_loss'].append(avg_g_loss)
                    self.training_history['d_loss'].append(avg_d_loss)
                    self.training_history['d_real_acc'].append(avg_d_real_acc)
                    self.training_history['d_fake_acc'].append(avg_d_fake_acc)
                    
                    if (epoch + 1) % 10 == 0:
                        LOGGER.info(f"Epoch [{epoch+1}/{self.num_epochs}] | "
                                  f"G_loss: {avg_g_loss:.4f} | D_loss: {avg_d_loss:.4f} | "
                                  f"D_real_acc: {avg_d_real_acc:.4f} | D_fake_acc: {avg_d_fake_acc:.4f}")
            
            LOGGER.info("Обучение cGAN завершено")
            return self.training_history
            
        except Exception as e:
            LOGGER.error(f"Критическая ошибка обучения cGAN: {e}", exc_info=True)
            raise
    
    def generate(self, 
                 condition: np.ndarray,
                 num_sequences: int = 1) -> np.ndarray:
        """
        Генерация синтетических последовательностей.
        
        Args:
            condition: Условный вектор (condition_dim,) или (num_sequences, condition_dim)
            num_sequences: Число последовательностей для генерации
        
        Returns:
            Сгенерированные последовательности (num_sequences, sequence_length, features)
        """
        try:
            if self.generator is None:
                raise ValueError("Generator не обучен")
            
            self.generator.eval()
            
            with torch.no_grad():
                # Подготовка условия
                if condition.ndim == 1:
                    condition = np.tile(condition, (num_sequences, 1))
                
                condition_tensor = torch.FloatTensor(condition).to(self.device)
                
                # Генерация
                noise = torch.randn(num_sequences, self.latent_dim).to(self.device)
                generated = self.generator(noise, condition_tensor)
                
                generated_np = generated.cpu().numpy()
            
            LOGGER.debug(f"Сгенерировано {num_sequences} последовательностей")
            return generated_np
            
        except Exception as e:
            LOGGER.error(f"Ошибка генерации данных: {e}", exc_info=True)
            raise
    
    def generate_pathological_scenarios(self, 
                                        base_price: float = 2000.0,
                                        num_bars: int = 720) -> Dict[str, pd.DataFrame]:
        """
        Генерация патологических сценариев для Stage 0.
        
        Args:
            base_price: Базовая цена актива
            num_bars: Число баров для генерации
        
        Returns:
            Словарь {scenario_name: DataFrame}
        """
        try:
            scenarios = {}
            pathological_specs = RegimeDefinitions.get_pathological_scenarios()
            
            for spec in pathological_specs:
                try:
                    scenario_name = spec['name']
                    LOGGER.info(f"Генерация патологического сценария: {scenario_name}")
                    
                    if scenario_name == 'sharp_gap':
                        df = self._generate_sharp_gaps(base_price, num_bars, spec)
                    elif scenario_name == 'prolonged_flat':
                        df = self._generate_prolonged_flat(base_price, num_bars, spec)
                    elif scenario_name == 'false_breakout':
                        df = self._generate_false_breakouts(base_price, num_bars, spec)
                    elif scenario_name == 'whipsaw':
                        df = self._generate_whipsaw(base_price, num_bars, spec)
                    elif scenario_name == 'liquidity_crisis':
                        df = self._generate_liquidity_crisis(base_price, num_bars, spec)
                    else:
                        continue
                    
                    scenarios[scenario_name] = df
                    
                except Exception as e:
                    LOGGER.error(f"Ошибка генерации сценария {spec.get('name', 'unknown')}: {e}")
                    continue
            
            LOGGER.info(f"Сгенерировано {len(scenarios)} патологических сценариев")
            return scenarios
            
        except Exception as e:
            LOGGER.error(f"Ошибка генерации патологических сценариев: {e}", exc_info=True)
            return {}
    
    def _generate_sharp_gaps(self, base_price: float, num_bars: int, spec: Dict) -> pd.DataFrame:
        """Генерация данных с резкими гэпами"""
        try:
            prices = [base_price]
            gap_frequency = spec.get('frequency', 0.05)
            gap_size = spec.get('gap_size', 50)
            
            for i in range(num_bars - 1):
                # Случайный гэп
                if np.random.random() < gap_frequency:
                    direction = np.random.choice([-1, 1])
                    gap = direction * gap_size * np.random.uniform(0.5, 1.5)
                    new_price = prices[-1] + gap
                else:
                    # Обычное движение
                    change = np.random.randn() * base_price * 0.001
                    new_price = prices[-1] + change
                
                prices.append(max(new_price, base_price * 0.5))
            
            df = self._create_ohlcv_from_close(prices)
            return df
            
        except Exception as e:
            LOGGER.error(f"Ошибка генерации sharp_gaps: {e}")
            raise
    
    def _generate_prolonged_flat(self, base_price: float, num_bars: int, spec: Dict) -> pd.DataFrame:
        """Генерация длительного флэта"""
        try:
            max_range = spec.get('max_range', 0.002)
            prices = []
            
            for i in range(num_bars):
                noise = np.random.randn() * base_price * max_range
                prices.append(base_price + noise)
            
            df = self._create_ohlcv_from_close(prices)
            df['Volume'] = df['Volume'] * spec.get('volume_multiplier', 0.3)
            
            return df
            
        except Exception as e:
            LOGGER.error(f"Ошибка генерации prolonged_flat: {e}")
            raise
    
    def _generate_false_breakouts(self, base_price: float, num_bars: int, spec: Dict) -> pd.DataFrame:
        """Генерация ложных пробоев"""
        try:
            prices = [base_price]
            frequency = spec.get('frequency', 0.1)
            breakout_size = spec.get('breakout_size', 30)
            reversal_size = spec.get('reversal_size', 50)
            
            in_breakout = False
            breakout_counter = 0
            
            for i in range(num_bars - 1):
                if not in_breakout and np.random.random() < frequency:
                    # Начало ложного пробоя
                    in_breakout = True
                    breakout_counter = 0
                    direction = np.random.choice([-1, 1])
                    change = direction * breakout_size
                else:
                    if in_breakout:
                        breakout_counter += 1
                        if breakout_counter < 5:
                            # Продолжение пробоя
                            change = direction * np.random.uniform(5, 15)
                        else:
                            # Разворот
                            change = -direction * reversal_size
                            in_breakout = False
                    else:
                        # Обычное движение
                        change = np.random.randn() * base_price * 0.001
                
                new_price = prices[-1] + change
                prices.append(max(new_price, base_price * 0.5))
            
            df = self._create_ohlcv_from_close(prices)
            return df
            
        except Exception as e:
            LOGGER.error(f"Ошибка генерации false_breakouts: {e}")
            raise
    
    def _generate_whipsaw(self, base_price: float, num_bars: int, spec: Dict) -> pd.DataFrame:
        """Генерация whipsaw движений"""
        try:
            amplitude = spec.get('amplitude', 40)
            frequency = spec.get('frequency', 0.15)
            
            prices = [base_price]
            direction = 1
            
            for i in range(num_bars - 1):
                if np.random.random() < frequency:
                    direction *= -1
                
                change = direction * amplitude * np.random.uniform(0.5, 1.5)
                new_price = prices[-1] + change
                prices.append(max(new_price, base_price * 0.5))
            
            df = self._create_ohlcv_from_close(prices)
            return df
            
        except Exception as e:
            LOGGER.error(f"Ошибка генерации whipsaw: {e}")
            raise
    
    def _generate_liquidity_crisis(self, base_price: float, num_bars: int, spec: Dict) -> pd.DataFrame:
        """Генерация кризиса ликвидности"""
        try:
            prices = []
            spread_multiplier = spec.get('spread_multiplier', 5.0)
            vol_multiplier = spec.get('volume_multiplier', 0.1)
            
            for i in range(num_bars):
                # Сильное расширение спреда
                change = np.random.randn() * base_price * 0.005
                prices.append(base_price + change)
            
            df = self._create_ohlcv_from_close(prices)
            
            # Расширяем спред
            df['High'] = df['High'] + (df['High'] - df['Low']) * (spread_multiplier - 1) / 2
            df['Low'] = df['Low'] - (df['High'] - df['Low']) * (spread_multiplier - 1) / 2
            
            # Снижаем объем
            df['Volume'] = df['Volume'] * vol_multiplier
            
            return df
            
        except Exception as e:
            LOGGER.error(f"Ошибка генерации liquidity_crisis: {e}")
            raise
    
    def _create_ohlcv_from_close(self, close_prices: List[float]) -> pd.DataFrame:
        """Создание OHLCV DataFrame из цен закрытия"""
        try:
            df = pd.DataFrame()
            df['Close'] = close_prices
            
            # Генерация OHLV с реалистичными значениями
            df['Open'] = df['Close'].shift(1).fillna(df['Close'].iloc[0])
            
            # High и Low с некоторым шумом
            noise = np.abs(np.random.randn(len(df))) * df['Close'] * 0.002
            df['High'] = df[['Open', 'Close']].max(axis=1) + noise
            df['Low'] = df[['Open', 'Close']].min(axis=1) - noise
            
            # Объем
            df['Volume'] = np.random.randint(10, 200, size=len(df))
            
            # Индекс времени
            dates = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
            df.index = dates
            
            return df
            
        except Exception as e:
            LOGGER.error(f"Ошибка создания OHLCV: {e}")
            raise