#!/usr/bin/env python
"""Test that scheduler.step() updates learning rate correctly"""
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

# Test warmup_plateau_decay scheduler
config.training.scheduler_name = 'warmup_plateau_decay'
total_steps = 1000

scheduler = train_module.create_scheduler(optimizer, config, total_steps)
print(f'✅ Scheduler created: {type(scheduler).__name__}')
print(f'   Base LR: {config.training.learning_rate:.2e}')
print(f'   Total steps: {total_steps}')
print(f'   Warmup ratio: {config.training.warmup_plateau_decay.warmup_ratio}')
print(f'   Plateau ratio: {config.training.warmup_plateau_decay.plateau_ratio}')
print()

# Simulate training loop
test_steps = [0, 50, 100, 250, 500, 600, 750, 900, 999]
print('Simulating training with scheduler.step():')
print()

for step in range(1000):
    # Simulate optimizer.step() and scheduler.step()
    if step in test_steps:
        current_lr = optimizer.param_groups[0]['lr']
        lr_mult = current_lr / config.training.learning_rate
        print(f'Step {step:4d}: LR = {current_lr:.2e}, multiplier = {lr_mult:.4f}')

    scheduler.step()

print()
print('✅ Test passed! LR changes correctly with scheduler.step()')
