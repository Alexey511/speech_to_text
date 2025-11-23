"""Test that CrossEntropyLoss with label_smoothing=0.0 is equivalent to model's built-in loss"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

def test_loss_equivalence():
    """Test that manual CrossEntropyLoss(label_smoothing=0.0) equals built-in loss"""

    print("Testing loss equivalence...")
    print("=" * 60)

    # Load a small Whisper model for testing
    model_name = "openai/whisper-tiny"
    print(f"Loading {model_name}...")

    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)
    model.eval()  # Set to eval mode for consistent results

    # Create dummy inputs
    batch_size = 2
    seq_len = 10  # Short sequence for testing

    # Random input features (mel-spectrogram)
    input_features = torch.randn(batch_size, 80, 3000)  # (batch, feature_dim, time_steps)

    # Random labels (token IDs)
    vocab_size = model.config.vocab_size
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Add some padding tokens (-100)
    labels[0, -2:] = -100  # Last 2 tokens of first sample are padding

    print(f"Input shape: {input_features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Vocab size: {vocab_size}")
    print()

    # ===== Method 1: Built-in loss (model computes it) =====
    with torch.no_grad():
        outputs_with_loss = model(input_features=input_features, labels=labels)
        builtin_loss = outputs_with_loss.loss

    print(f"Built-in loss (model's internal): {builtin_loss.item():.6f}")

    # ===== Method 2: Manual CrossEntropyLoss with label_smoothing=0.0 =====
    # When we pass labels, the model returns logits automatically
    # So we can reuse the same forward pass to get logits
    with torch.no_grad():
        # We need to get logits - the model returns them even when labels are passed
        logits = outputs_with_loss.logits  # (batch_size, seq_len, vocab_size)

    # Flatten logits and labels for CrossEntropyLoss
    logits_flat = logits.view(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
    labels_flat = labels.view(-1)  # (batch_size * seq_len)

    # Create CrossEntropyLoss with label_smoothing=0.0
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=-100,
        label_smoothing=0.0
    )
    manual_loss = criterion(logits_flat, labels_flat)

    print(f"Manual loss (CrossEntropyLoss, smoothing=0.0): {manual_loss.item():.6f}")
    print()

    # ===== Compare losses =====
    absolute_diff = abs(builtin_loss.item() - manual_loss.item())
    relative_diff = absolute_diff / builtin_loss.item() * 100

    print(f"Absolute difference: {absolute_diff:.8f}")
    print(f"Relative difference: {relative_diff:.6f}%")
    print()

    # Check if losses are equivalent (allowing for small numerical differences)
    tolerance = 1e-5
    if absolute_diff < tolerance:
        print("✅ SUCCESS: Losses are equivalent!")
        print(f"   Difference ({absolute_diff:.8f}) is within tolerance ({tolerance})")
        return True
    else:
        print("❌ WARNING: Losses differ significantly!")
        print(f"   Difference ({absolute_diff:.8f}) exceeds tolerance ({tolerance})")
        return False

if __name__ == "__main__":
    print("Testing CrossEntropyLoss equivalence with model's built-in loss")
    print()

    success = test_loss_equivalence()

    print()
    if success:
        print("🎉 Test passed! Safe to use unified loss calculation.")
    else:
        print("⚠️  Test failed! Need to investigate differences.")
