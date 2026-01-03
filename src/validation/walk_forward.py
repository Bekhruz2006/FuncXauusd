import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import random


@dataclass
class WalkForwardConfig:
    n_is_blocks: int = 10
    n_oos_blocks: int = 5
    min_ppt: float = 0.0
    max_drawdown: float = 0.05
    min_sharpe: float = 0.5
    max_retries: int = 3
    noise_level: float = 0.001
    l2_increment: float = 0.5
    depth_decrement: int = 1


class WalkForwardValidator:
    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.is_blocks: List[pd.DataFrame] = []
        self.oos_blocks: List[pd.DataFrame] = []
        self.checkpoint_history: List[Dict] = []
        self.current_retry = 0
        
    def split_data(self, is_data: pd.DataFrame, oos_data: pd.DataFrame) -> None:
        is_size = len(is_data) // self.config.n_is_blocks
        oos_size = len(oos_data) // self.config.n_oos_blocks
        
        self.is_blocks = [
            is_data.iloc[i*is_size:(i+1)*is_size].copy()
            for i in range(self.config.n_is_blocks)
        ]
        
        self.oos_blocks = [
            oos_data.iloc[i*oos_size:(i+1)*oos_size].copy()
            for i in range(self.config.n_oos_blocks)
        ]
        
        random.shuffle(self.oos_blocks)
        
    def validate_sequential(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        params: Dict
    ) -> Tuple[bool, List[Dict]]:
        accumulated_data = self.is_blocks[0].copy()
        checkpoint_results = []
        
        for checkpoint_idx, oos_block in enumerate(self.oos_blocks):
            model = train_fn(accumulated_data, params)
            metrics = eval_fn(model, oos_block)
            
            passed = self._check_checkpoint(metrics)
            
            checkpoint_results.append({
                'checkpoint': checkpoint_idx,
                'metrics': metrics,
                'passed': passed,
                'accumulated_samples': len(accumulated_data)
            })
            
            if not passed:
                if self.current_retry < self.config.max_retries:
                    return self._restart_with_enhancement(
                        train_fn, eval_fn, params
                    )
                return False, checkpoint_results
                
            accumulated_data = pd.concat([accumulated_data, oos_block], ignore_index=True)
            
        return True, checkpoint_results
    
    def _check_checkpoint(self, metrics: Dict) -> bool:
        return (
            metrics['ppt'] >= self.config.min_ppt and
            metrics['drawdown'] <= self.config.max_drawdown and
            metrics['sharpe'] >= self.config.min_sharpe
        )
    
    def _restart_with_enhancement(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        params: Dict
    ) -> Tuple[bool, List[Dict]]:
        self.current_retry += 1
        
        for block in self.is_blocks:
            noise = np.random.uniform(
                -self.config.noise_level,
                self.config.noise_level,
                len(block)
            )
            block['close'] = block['close'] * (1 + noise)
        
        params['l2_leaf_reg'] = params.get('l2_leaf_reg', 3) + self.config.l2_increment
        params['depth'] = max(3, params.get('depth', 5) - self.config.depth_decrement)
        
        random.shuffle(self.oos_blocks)
        
        return self.validate_sequential(train_fn, eval_fn, params)
    
    def print_detailed_report(self) -> None:
        print(f"\n{'='*70}")
        print(f"  WALK-FORWARD VALIDATION REPORT")
        print(f"{'='*70}\n")
        
        for result in self.checkpoint_history:
            status = "✓" if result['passed'] else "✗"
            print(f"Checkpoint {result['checkpoint']}: {status}")
            print(f"  PPT: {result['metrics']['ppt']:.2f}")
            print(f"  DD: {result['metrics']['drawdown']:.2%}")
            print(f"  Sharpe: {result['metrics']['sharpe']:.2f}\n")


def create_walk_forward_splits(
    data: pd.DataFrame,
    is_ratio: float = 0.6,
    oos_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = len(data)
    is_end = int(total * is_ratio)
    oos_end = int(total * (is_ratio + oos_ratio))
    
    return (
        data.iloc[:is_end].copy(),
        data.iloc[is_end:oos_end].copy(),
        data.iloc[oos_end:].copy()
    )