import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import yaml
import tempfile

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestImports:
    def test_all_imports(self):
        try:
            from src.data.loader import load_price_data, cache_prices
            from src.features.engineering import create_features
            from src.features.multiframe import create_multiframe_features
            from src.labeling.strategies import get_labels_one_direction
            from src.models.trainer import ClusterModelTrainer
            from src.models.validator import validate_class_balance
            from src.export.onnx_exporter import export_to_onnx
            from src.backtesting.tester import test_model_one_direction
            from src.risk.atr_manager import ATRRiskManager, calculate_atr
            from src.validation.walk_forward import WalkForwardValidator, WalkForwardConfig
            from src.monitoring.degradation import DegradationMonitor
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")


class TestDataPipeline:
    @pytest.fixture
    def mock_csv_data(self, tmp_path):
        csv_path = tmp_path / "XAUUSD_H1.csv"
        dates = pd.date_range('2020-01-01', periods=2000, freq='H')
        prices = 1800 + np.cumsum(np.random.randn(2000) * 5)
        
        df = pd.DataFrame({
            'Date': dates.strftime('%Y.%m.%d %H:%M'),
            'Open': prices * 0.999,
            'High': prices * 1.002,
            'Low': prices * 0.998,
            'Close': prices,
            'Volume': np.random.randint(100, 1000, 2000)
        })
        
        df.to_csv(csv_path, sep=';', index=False)
        return tmp_path, csv_path
    
    @pytest.fixture
    def mock_config(self, mock_csv_data):
        tmp_path, csv_path = mock_csv_data
        return {
            'symbol': {'name': 'XAUUSD_H1', 'timeframe': 'H1'},
            'trading': {
                'direction': 'buy',
                'risk': {'stop_loss': 10.0, 'take_profit': 5.0},
                'labeling': {'markup': 0.25, 'min_bars': 1, 'max_bars': 15}
            },
            'data': {
                'paths': {'raw': str(tmp_path)},
                'backward': '2020-01-01',
                'forward': '2020-06-01',
                'full_forward': '2021-01-01'
            },
            'periods': [5, 10, 15, 20, 30],
            'periods_meta': [5],
            'n_clusters': 3,
            'markup': 0.25,
            'model': {
                'main': {
                    'params': {
                        'iterations': 100,
                        'depth': 4,
                        'learning_rate': 0.03,
                        'l2_leaf_reg': 3,
                        'eval_metric': 'Accuracy',
                        'verbose': False,
                        'random_seed': 42
                    }
                },
                'meta': {
                    'params': {
                        'iterations': 50,
                        'depth': 3,
                        'learning_rate': 0.03,
                        'l2_leaf_reg': 3,
                        'eval_metric': 'F1',
                        'verbose': False,
                        'random_seed': 42
                    }
                }
            },
            'clustering': {'n_clusters': 3, 'random_state': 42, 'n_init': 10},
            'validation': {
                'train_size': 0.75,
                'shuffle': False,
                'stratify': False,
                'random_state': 42,
                'criteria': {
                    'min_class_balance': 0.15,
                    'min_samples_per_class': 50
                }
            },
            'export': {
                'paths': {
                    'models': str(tmp_path / 'models'),
                    'onnx': str(tmp_path / 'onnx')
                }
            }
        }
    
    def test_data_loading(self, mock_config):
        from src.data.loader import load_price_data
        
        data = load_price_data(mock_config)
        
        assert isinstance(data, pd.DataFrame)
        assert 'close' in data.columns
        assert len(data) > 0
        assert data.index.name == 'time' or isinstance(data.index, pd.DatetimeIndex)
    
    def test_feature_creation(self, mock_config):
        from src.data.loader import load_price_data
        from src.features.engineering import create_features
        
        data = load_price_data(mock_config)
        features = create_features(data, mock_config['periods'], mock_config['periods_meta'])
        
        assert 'feat_0' in features.columns
        assert 'meta_0' in features.columns
        assert len(features) > 0
        assert features.isna().sum().sum() == 0
    
    def test_atr_calculation(self, mock_config):
        from src.data.loader import load_price_data
        from src.risk.atr_manager import calculate_atr
        
        data = load_price_data(mock_config)
        
        data['high'] = data['close'] * 1.002
        data['low'] = data['close'] * 0.998
        
        atr = calculate_atr(data, period=14)
        
        assert isinstance(atr, pd.Series)
        assert (atr.dropna() >= 0).all()
    
    def test_labeling_with_atr(self, mock_config):
        from src.data.loader import load_price_data
        from src.features.engineering import create_features
        from src.labeling.strategies import get_labels_one_direction
        
        data = load_price_data(mock_config)
        
        data['high'] = data['close'] * 1.002
        data['low'] = data['close'] * 0.998
        
        features = create_features(data, mock_config['periods'], mock_config['periods_meta'])
        
        labeled = get_labels_one_direction(
            features,
            markup=mock_config['markup'],
            min_bars=mock_config['trading']['labeling']['min_bars'],
            max_bars=mock_config['trading']['labeling']['max_bars'],
            direction=mock_config['trading']['direction']
        )
        
        assert 'labels' in labeled.columns
        assert set(labeled['labels'].unique()).issubset({0.0, 0.2, 1.0})
        
        timeout_count = (labeled['labels'] == 0.2).sum()
        assert timeout_count > 0, "Should have timeout labels (0.2)"


class TestTrainingPipeline:
    @pytest.fixture
    def prepared_data(self, mock_config):
        from src.data.loader import load_price_data
        from src.features.engineering import create_features
        from src.labeling.strategies import get_labels_one_direction
        
        data = load_price_data(mock_config)
        data['high'] = data['close'] * 1.002
        data['low'] = data['close'] * 0.998
        
        features = create_features(data, mock_config['periods'], mock_config['periods_meta'])
        
        labeled = get_labels_one_direction(
            features,
            markup=mock_config['markup'],
            min_bars=1,
            max_bars=15,
            direction='buy'
        )
        
        return labeled
    
    def test_cluster_trainer_initialization(self, mock_config):
        from src.models.trainer import ClusterModelTrainer
        
        trainer = ClusterModelTrainer(mock_config)
        
        assert trainer.config == mock_config
        assert trainer.data is None
        assert trainer.clusters is None
    
    def test_full_training_cycle(self, mock_config, prepared_data):
        from src.models.trainer import ClusterModelTrainer
        
        mock_config['min_samples'] = 50
        
        trainer = ClusterModelTrainer(mock_config)
        trainer.data = prepared_data
        trainer._perform_clustering()
        
        assert trainer.clusters is not None
        assert len(trainer.clusters) == len(prepared_data)
        
        cluster_0_data = prepared_data[trainer.clusters == 0]
        
        if len(cluster_0_data) >= 100:
            result = trainer._train_single_cluster(0)
            
            if result is not None:
                assert 'model' in result
                assert 'meta_model' in result
                assert 'val_acc' in result
                assert 'r2' in result
                assert result['val_acc'] >= 0
                assert result['val_acc'] <= 1


class TestWalkForward:
    def test_walk_forward_splits(self):
        from src.validation.walk_forward import create_walk_forward_splits
        
        data = pd.DataFrame({
            'close': np.random.randn(1000),
            'labels': np.random.randint(0, 2, 1000).astype(float)
        })
        
        is_data, oos_data, oot_data = create_walk_forward_splits(data)
        
        assert len(is_data) == 600
        assert len(oos_data) == 200
        assert len(oot_data) == 200
    
    def test_walk_forward_validator(self):
        from src.validation.walk_forward import WalkForwardValidator, WalkForwardConfig
        
        config = WalkForwardConfig(n_is_blocks=3, n_oos_blocks=2)
        validator = WalkForwardValidator(config)
        
        data = pd.DataFrame({
            'close': np.random.randn(1000),
            'labels': np.random.randint(0, 2, 1000).astype(float)
        })
        
        is_data = data.iloc[:600]
        oos_data = data.iloc[600:800]
        
        validator.split_data(is_data, oos_data)
        
        assert len(validator.is_blocks) == 3
        assert len(validator.oos_blocks) == 2


class TestATRManager:
    def test_atr_manager_initialization(self):
        from src.risk.atr_manager import ATRRiskManager
        
        manager = ATRRiskManager()
        
        assert manager.sl_multiplier == 2.0
        assert manager.tp_multiplier == 2.5
    
    def test_level_calculation(self):
        from src.risk.atr_manager import ATRRiskManager
        
        manager = ATRRiskManager()
        levels = manager.calculate_levels(1800.0, 5.0, 'buy')
        
        assert levels['sl'] < 1800.0
        assert levels['tp'] > 1800.0
        assert levels['risk_reward_ratio'] > 1.0


class TestDegradationMonitor:
    def test_monitor_initialization(self):
        from src.monitoring.degradation import DegradationMonitor, DegradationStatus
        
        hist_metrics = {
            'max_drawdown': 0.08,
            'win_rate': 0.58,
            'avg_profit_per_trade': 12.5,
            'profit_factor': 1.5
        }
        
        monitor = DegradationMonitor(hist_metrics)
        
        assert monitor.status == DegradationStatus.HEALTHY
        assert len(monitor.triggers) > 0
    
    def test_monitor_update(self):
        from src.monitoring.degradation import DegradationMonitor
        from datetime import datetime
        
        hist_metrics = {
            'max_drawdown': 0.08,
            'win_rate': 0.58,
            'avg_profit_per_trade': 12.5,
            'profit_factor': 1.5
        }
        
        monitor = DegradationMonitor(hist_metrics)
        
        trade = {
            'profit': 10.0,
            'entry_price': 1800,
            'exit_price': 1810,
            'direction': 'buy',
            'timestamp': datetime.now()
        }
        
        status = monitor.update(trade)
        
        assert monitor.metrics.total_trades == 1
        assert monitor.metrics.total_profit == 10.0


class TestEndToEnd:
    def test_minimal_training_workflow(self, mock_config):
        from src.data.loader import load_price_data
        from src.features.engineering import create_features
        from src.labeling.strategies import get_labels_one_direction
        from src.models.trainer import ClusterModelTrainer
        
        data = load_price_data(mock_config)
        data['high'] = data['close'] * 1.002
        data['low'] = data['close'] * 0.998
        
        features = create_features(
            data,
            mock_config['periods'],
            mock_config['periods_meta']
        )
        
        labeled = get_labels_one_direction(
            features,
            markup=0.25,
            min_bars=1,
            max_bars=15,
            direction='buy'
        )
        
        assert len(labeled) > 500
        assert 'labels' in labeled.columns
        assert 'feat_0' in labeled.columns
        assert 'meta_0' in labeled.columns
        
        mock_config['min_samples'] = 50
        
        trainer = ClusterModelTrainer(mock_config)
        trainer.data = labeled
        trainer._perform_clustering()
        
        assert trainer.clusters is not None
        print(f"\n✅ End-to-end workflow completed successfully")
        print(f"   Data points: {len(labeled)}")
        print(f"   Clusters: {len(np.unique(trainer.clusters))}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])