"""
Model training module
Implements cluster-based model training with hyperparameter search
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from catboost import CatBoostClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from ..data.loader import load_price_data, split_data
from ..features.engineering import create_features
from ..labeling.strategies import get_labels_one_direction
from ..models.validator import (
    validate_class_balance,
    validate_sample_size,
    calculate_classification_metrics
)


class ClusterModelTrainer:
    """
    Trains separate CatBoost models for each market regime (cluster)
    """
    
    def __init__(self, config: dict):
        """
        Initialize trainer with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data = None
        self.features_main = None
        self.features_meta = None
        self.labels = None
        self.clusters = None
        self.scaler = StandardScaler()
        
        # Extract key parameters
        self.symbol = config['symbol']['name']
        self.direction = config['trading']['direction']
        self.markup = config.get('markup', config['trading']['labeling']['markup'])
        self.n_clusters = config.get('n_clusters', config['clustering']['n_clusters'])
        self.periods = config.get('periods', [5, 35, 65, 95, 125, 155, 185, 215, 245, 275])
        self.meta_periods = config.get('periods_meta', [5])
        self.min_samples = config.get('min_samples', 1000)
        
    def prepare_data(self, verbose: bool = True) -> bool:
        """
        Load and prepare all data for training
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load price data
            self.data = load_price_data(self.config, verbose=verbose)
            
            # Create features
            if verbose:
                print(f"🔧 Creating features...")
            
            self.features_main, self.features_meta = create_features(
                self.data,
                self.periods,
                self.meta_periods,
                verbose=False
            )
            
            # Create labels
            if verbose:
                print(f"🏷️  Creating labels (markup={self.markup})...")
            
            self.labels = get_labels_one_direction(
                self.data,
                markup=self.markup,
                direction=self.direction,
                min_bars=self.config['trading']['labeling']['min_bars'],
                max_bars=self.config['trading']['labeling']['max_bars'],
                verbose=False
            )
            
            # Align all datasets
            common_idx = (
                self.features_main.index
                .intersection(self.features_meta.index)
                .intersection(self.labels.index)
            )
            
            self.features_main = self.features_main.loc[common_idx]
            self.features_meta = self.features_meta.loc[common_idx]
            self.labels = self.labels.loc[common_idx]
            
            if verbose:
                print(f"✅ Prepared {len(self.labels):,} samples")
            
            return True
            
        except Exception as e:
            print(f"❌ Data preparation failed: {e}")
            return False
    
    def perform_clustering(self, verbose: bool = True) -> bool:
        """
        Cluster market regimes using meta-features
        
        Returns:
            True if successful
        """
        try:
            if verbose:
                print(f"🎯 Clustering into {self.n_clusters} regimes...")
            
            # Scale meta-features
            X_meta_scaled = self.scaler.fit_transform(self.features_meta)
            
            # K-Means clustering
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.config['clustering']['random_state'],
                n_init=self.config['clustering']['n_init']
            )
            
            self.clusters = kmeans.fit_predict(X_meta_scaled)
            
            # Analyze cluster distribution
            unique, counts = np.unique(self.clusters, return_counts=True)
            
            if verbose:
                print(f"\n📊 Cluster distribution:")
                for cluster_id, count in zip(unique, counts):
                    pct = count / len(self.clusters) * 100
                    print(f"   Cluster {cluster_id}: {count:,} samples ({pct:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"❌ Clustering failed: {e}")
            return False
    
    def train_single_cluster(
        self,
        cluster_id: int,
        verbose: bool = True
    ) -> Optional[Dict]:
        """
        Train model for a single cluster
        
        Args:
            cluster_id: ID of cluster to train
            verbose: Print training info
            
        Returns:
            Dictionary with model and metrics, or None if failed
        """
        # Filter data for this cluster
        cluster_mask = self.clusters == cluster_id
        
        X_cluster = self.features_main[cluster_mask]
        y_cluster = self.labels[cluster_mask]
        X_meta_cluster = self.features_meta[cluster_mask]
        
        n_samples = len(X_cluster)
        
        if verbose:
            print(f"\n{'─'*60}")
            print(f"  Training Cluster {cluster_id}")
            print(f"{'─'*60}")
            print(f"  Samples: {n_samples:,}")
        
        # Validate sample size
        if n_samples < self.min_samples:
            if verbose:
                print(f"  ⚠️ Insufficient samples (need {self.min_samples:,})")
            return None
        
        # Validate class balance
        balance_valid, balance_stats = validate_class_balance(
            y_cluster,
            min_ratio=self.config['validation']['criteria']['min_class_balance']
        )
        
        if not balance_valid:
            if verbose:
                print(f"  ⚠️ Poor class balance: {balance_stats['minority_ratio']:.2%}")
            return None
        
        # Split into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_cluster, y_cluster,
            test_size=self.config['validation']['test_size'],
            shuffle=self.config['validation']['shuffle'],
            stratify=y_cluster if self.config['validation']['stratify'] else None,
            random_state=self.config['validation']['random_state']
        )
        
        X_meta_train, X_meta_val = train_test_split(
            X_meta_cluster,
            test_size=self.config['validation']['test_size'],
            shuffle=self.config['validation']['shuffle'],
            random_state=self.config['validation']['random_state']
        )
        
        if verbose:
            print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")
            print(f"  Positive class: {(y_train == 1).mean():.2%}")
        
        # Train main model (trading signals)
        if verbose:
            print(f"\n  🚀 Training main model...")
        
        model_main = CatBoostClassifier(
            iterations=self.config.get('iterations', 
                                      self.config['model']['main']['params']['iterations']),
            depth=self.config.get('depth',
                                 self.config['model']['main']['params']['depth']),
            learning_rate=self.config['model']['main']['params']['learning_rate'],
            l2_leaf_reg=self.config['model']['main']['params']['l2_leaf_reg'],
            eval_metric=self.config['model']['main']['params']['eval_metric'],
            verbose=False,
            use_best_model=True,
            early_stopping_rounds=50,
            random_seed=self.config['model']['main']['params']['random_seed']
        )
        
        model_main.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )
        
        # Train meta model (cluster filtering)
        if verbose:
            print(f"  🎯 Training meta model...")
        
        # Create binary labels: 1 if in current cluster, 0 otherwise
        y_meta_train = (self.clusters[X_meta_train.index] == cluster_id).astype(int)
        y_meta_val = (self.clusters[X_meta_val.index] == cluster_id).astype(int)
        
        model_meta = CatBoostClassifier(
            iterations=self.config['model']['meta']['params']['iterations'],
            depth=self.config['model']['meta']['params']['depth'],
            learning_rate=self.config['model']['meta']['params']['learning_rate'],
            l2_leaf_reg=self.config['model']['meta']['params']['l2_leaf_reg'],
            eval_metric=self.config['model']['meta']['params']['eval_metric'],
            verbose=False,
            use_best_model=True,
            early_stopping_rounds=30,
            random_seed=self.config['model']['meta']['params']['random_seed']
        )
        
        model_meta.fit(
            X_meta_train, y_meta_train,
            eval_set=(X_meta_val, y_meta_val),
            verbose=False
        )
        
        # Calculate metrics
        y_pred_val = model_main.predict(X_val)
        metrics = calculate_classification_metrics(y_val, y_pred_val)
        
        if verbose:
            print(f"\n  📊 Results:")
            print(f"     Val Accuracy: {metrics['accuracy']:.4f}")
            print(f"     Val F1: {metrics['f1']:.4f}")
            print(f"     Precision: {metrics['precision']:.4f}")
            print(f"     Recall: {metrics['recall']:.4f}")
        
        # Prepare result
        result = {
            'cluster': cluster_id,
            'model': model_main,
            'meta_model': model_meta,
            'val_acc': metrics['accuracy'],
            'val_f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'samples': n_samples,
            'balance': balance_stats['minority_ratio'],
            'X_val': X_val,
            'y_val': y_val,
            'dataset': self.data  # For later testing
        }
        
        return result
    
    def train_all_clusters(self, verbose: bool = True) -> List[Dict]:
        """
        Train models for all clusters
        
        Returns:
            List of successfully trained models
        """
        if self.data is None:
            if not self.prepare_data(verbose=verbose):
                return []
        
        if self.clusters is None:
            if not self.perform_clustering(verbose=verbose):
                return []
        
        results = []
        
        for cluster_id in range(self.n_clusters):
            try:
                result = self.train_single_cluster(cluster_id, verbose=verbose)
                
                if result is not None:
                    results.append(result)
                    
                    if verbose:
                        print(f"  ✅ Cluster {cluster_id} trained successfully")
                
            except Exception as e:
                if verbose:
                    print(f"  ❌ Cluster {cluster_id} failed: {e}")
                continue
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"  ✅ Trained {len(results)}/{self.n_clusters} clusters")
            print(f"{'='*60}")
        
        return results
    
    def get_best_model(self, results: List[Dict]) -> Optional[Dict]:
        """
        Select best model from results
        
        Args:
            results: List of training results
            
        Returns:
            Best model result or None
        """
        if not results:
            return None
        
        # Sort by validation accuracy
        best = max(results, key=lambda x: x['val_acc'])
        
        return best


def fit_model(
    dataset: pd.DataFrame,
    result: List,
    config: dict,
    verbose: bool = False
) -> Tuple[CatBoostClassifier, CatBoostClassifier, dict]:
    """
    Legacy compatibility function
    Fits models and returns them with statistics
    
    Args:
        dataset: OHLCV DataFrame
        result: Empty list (will be populated)
        config: Configuration dictionary
        verbose: Print training info
        
    Returns:
        Tuple of (main_model, meta_model, statistics)
    """
    trainer = ClusterModelTrainer(config)
    
    if not trainer.prepare_data(verbose=verbose):
        raise ValueError("Data preparation failed")
    
    if not trainer.perform_clustering(verbose=verbose):
        raise ValueError("Clustering failed")
    
    results = trainer.train_all_clusters(verbose=verbose)
    
    if not results:
        raise ValueError("No models trained successfully")
    
    best = trainer.get_best_model(results)
    result.append(best['model'])
    result.append(best['meta_model'])
    
    stats = {
        'val_accuracy': best['val_acc'],
        'val_f1': best['val_f1'],
        'cluster': best['cluster'],
        'samples': best['samples']
    }
    
    return best['model'], best['meta_model'], stats