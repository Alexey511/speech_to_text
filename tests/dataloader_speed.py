#!/usr/bin/env python3
"""
Script to measure DataLoader speed for CommonVoiceDataset
"""

import sys
import logging
import pandas as pd
import torch
from pathlib import Path
import torchaudio
import time
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.data import DataManager, CommonVoiceDataset, AudioCollator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)

logger = logging.getLogger(__name__)

def load_subset(data_manager, max_rows, split="train"):
    """Load a subset of dataset (up to max_rows)"""
    split_mapping = {
        "train": ("cv22_train", data_manager.train_tsv),
        "validation": ("cv22_dev", data_manager.dev_tsv),
        "val": ("cv22_dev", data_manager.dev_tsv),
        "dev": ("cv22_dev", data_manager.dev_tsv),
        "test": ("cv22_test", data_manager.test_tsv)
    }
    cache_key, tsv_file = split_mapping[split]

    if cache_key in data_manager.dataset_cache:
        logger.info(f"Loading {split} subset from cache (key: {cache_key})...")
        return data_manager.dataset_cache[cache_key].iloc[:max_rows]

    if not tsv_file.exists():
        raise ValueError(f"TSV file not found: {tsv_file}")

    logger.info(f"Loading first {max_rows} rows from {tsv_file}...")
    df = pd.read_csv(tsv_file, sep='\t', encoding='utf-8', nrows=max_rows)

    # Calculate durations
    durations = []
    for idx, row in df.iterrows():
        audio_path = data_manager.clips_dir / row['path']
        try:
            if audio_path.exists():
                info = torchaudio.info(str(audio_path))
                duration = info.num_frames / info.sample_rate
            else:
                logger.warning(f"Audio file not found: {audio_path}")
                duration = 0.0
            durations.append(duration)
        except Exception as e:
            logger.warning(f"Could not get duration for {audio_path}: {e}")
            durations.append(0.0)

    df['duration'] = durations
    df = data_manager._apply_filters(df)
    data_manager.dataset_cache[cache_key] = df
    return df.iloc[:max_rows]

def measure_dataloader_speed(config, device, batch_size, num_workers, max_rows=1000):
    """Measure DataLoader speed for given device, batch_size, and num_workers"""
    logger.info(f"Starting speed test for {device} mode (batch_size={batch_size}, num_workers={num_workers})")
    
    # Initialize DataManager
    data_manager = DataManager(config)
    
    # Override load_dataset
    original_load_dataset = data_manager.load_dataset
    data_manager.load_dataset = lambda split="train": load_subset(data_manager, max_rows, split)
    
    # Set device
    config.data.processing_and_augmentation_device = device
    
    # Setup processor
    data_manager.setup_processor(config.model.model_name, "whisper")
    
    # Create dataset and dataloader
    dataset = data_manager.create_dataset("train")
    dataloader = data_manager.create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # Measure speed
    start_time = time.time()
    num_batches = 0
    total_samples = 0
    for batch in tqdm(dataloader, total=len(dataloader), desc=f"{device.upper()} mode (batch_size={batch_size}, num_workers={num_workers})"):
        num_batches += 1
        if batch is not None:
            batch_size_actual = batch['input_features'].shape[0] if device == "cpu" else batch['audio'].shape[0]
            total_samples += batch_size_actual
        else:
            continue
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate metrics
    avg_time_per_batch = total_time / num_batches if num_batches > 0 else 0
    avg_time_per_sample = total_time / total_samples if total_samples > 0 else 0
    
    logger.info(
        f"{device.upper()} mode: Processed {num_batches} batches, "
        f"{total_samples} samples in {total_time:.2f} seconds. "
        f"Avg time per batch: {avg_time_per_batch:.4f} seconds, "
        f"Avg time per sample: {avg_time_per_sample:.4f} seconds "
        f"(batch_size={batch_size}, num_workers={num_workers})"
    )
    
    # Restore original load_dataset
    data_manager.load_dataset = original_load_dataset
    
    return total_time, num_batches, total_samples, avg_time_per_batch, avg_time_per_sample

def main():
    """Run speed tests for DataLoader"""
    config = load_config("configs/default.yaml")
    
    # Parameters to test
    batch_sizes = [9]
    num_workers_list = [3]
    devices = ["cuda", "cpu"]
    
    for device in devices:
        for batch_size in batch_sizes:
            for num_workers in num_workers_list:
                try:
                    total_time, num_batches, total_samples, avg_time_per_batch, avg_time_per_sample = \
                        measure_dataloader_speed(config, device, batch_size, num_workers)
                except Exception as e:
                    logger.error(f"Error in {device} mode (batch_size={batch_size}, num_workers={num_workers}): {e}")

if __name__ == "__main__":
    main()