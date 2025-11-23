#!/usr/bin/env python
"""Test script for warmup_plateau_decay scheduler"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
import torch
from torch.optim import AdamW

# Import create_scheduler from train.py
import importlib.util
spec = importlib.util.spec_from_file_location('train', str(project_root / 'train.py'))
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

# Load config
config = load_config('configs/default.yaml')

# Create dummy model and optimizer
model = torch.nn.Linear(10, 1)
optimizer = AdamW(model.parameters(), lr=config.training.learning_rate)

# Test creating the new scheduler
config.training.scheduler_name = 'warmup_plateau_decay'
total_steps = 1000

scheduler = train_module.create_scheduler(optimizer, config, total_steps)
print('Scheduler created successfully!')
print(f'Scheduler type: {type(scheduler).__name__}')

# Test LR schedule at different steps
print('\nLearning rate schedule:')
print(f'Total steps: {total_steps}')
print(f'Warmup ratio: {config.training.warmup_plateau_decay.warmup_ratio}')
print(f'Plateau ratio: {config.training.warmup_plateau_decay.plateau_ratio}')
print(f'Warmup steps: {int(total_steps * config.training.warmup_plateau_decay.warmup_ratio)}')
print(f'Plateau ends at step: {int(total_steps * config.training.warmup_plateau_decay.plateau_ratio)}')
print()

test_steps = [0, 50, 100, 250, 500, 600, 750, 900, 999]
for step in test_steps:
    # Get current LR multiplier
    lr_mult = scheduler.lr_lambdas[0](step)
    actual_lr = config.training.learning_rate * lr_mult
    print(f'Step {step:4d}: LR multiplier = {lr_mult:.4f}, Actual LR = {actual_lr:.2e}')

print('\nTest passed!')
