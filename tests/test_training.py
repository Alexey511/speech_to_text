"""
Unit tests for training functions (train.py).
"""
import pytest
import logging
from pathlib import Path
import shutil

# Import functions from train module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from train import cleanup_old_checkpoints


class TestCleanupOldCheckpoints:
    """Tests for cleanup_old_checkpoints function"""

    def test_cleanup_keeps_last_n_checkpoints_numerical_order(self, temp_dir):
        """
        Test that cleanup correctly keeps the last N checkpoints based on NUMERICAL order,
        not lexicographical order.

        This is a regression test for the bug where epoch_7, epoch_8, epoch_9 were kept
        instead of epoch_12, epoch_13, epoch_14 when keep_last=3 and epochs 7-14 existed.
        """
        checkpoints_dir = temp_dir / "checkpoints"
        checkpoints_dir.mkdir()

        # Create checkpoint directories for epochs 7-14 (simulating real scenario)
        for epoch in [7, 8, 9, 10, 11, 12, 13, 14]:
            (checkpoints_dir / f"epoch_{epoch}").mkdir()

        # Setup logger
        logger = logging.getLogger("test_cleanup")
        logger.setLevel(logging.WARNING)  # Suppress INFO logs during test

        # Cleanup with keep_last=3
        cleanup_old_checkpoints(checkpoints_dir, keep_last=3, logger=logger)

        # Check remaining directories
        remaining_dirs = sorted([d.name for d in checkpoints_dir.glob("epoch_*")],
                               key=lambda x: int(x.split('_')[1]))

        # Should keep LAST 3 epochs (12, 13, 14), not lexicographically last (7, 8, 9)
        assert remaining_dirs == ['epoch_12', 'epoch_13', 'epoch_14'], \
            f"Expected ['epoch_12', 'epoch_13', 'epoch_14'], got {remaining_dirs}"

    def test_cleanup_with_single_digit_epochs(self, temp_dir):
        """Test cleanup with epochs 0-5"""
        checkpoints_dir = temp_dir / "checkpoints"
        checkpoints_dir.mkdir()

        for epoch in range(6):
            (checkpoints_dir / f"epoch_{epoch}").mkdir()

        logger = logging.getLogger("test_cleanup")
        logger.setLevel(logging.WARNING)

        cleanup_old_checkpoints(checkpoints_dir, keep_last=2, logger=logger)

        remaining_dirs = sorted([d.name for d in checkpoints_dir.glob("epoch_*")],
                               key=lambda x: int(x.split('_')[1]))

        assert remaining_dirs == ['epoch_4', 'epoch_5']

    def test_cleanup_with_fewer_checkpoints_than_limit(self, temp_dir):
        """Test that cleanup does nothing when there are fewer checkpoints than keep_last"""
        checkpoints_dir = temp_dir / "checkpoints"
        checkpoints_dir.mkdir()

        # Create only 2 checkpoints
        for epoch in [0, 1]:
            (checkpoints_dir / f"epoch_{epoch}").mkdir()

        logger = logging.getLogger("test_cleanup")
        logger.setLevel(logging.WARNING)

        # Try to keep 5 checkpoints (more than exist)
        cleanup_old_checkpoints(checkpoints_dir, keep_last=5, logger=logger)

        # Should keep all 2 checkpoints
        remaining_dirs = sorted([d.name for d in checkpoints_dir.glob("epoch_*")],
                               key=lambda x: int(x.split('_')[1]))

        assert remaining_dirs == ['epoch_0', 'epoch_1']

    def test_cleanup_mixed_epoch_numbers(self, temp_dir):
        """Test cleanup with non-consecutive epoch numbers"""
        checkpoints_dir = temp_dir / "checkpoints"
        checkpoints_dir.mkdir()

        # Create checkpoints for non-consecutive epochs
        for epoch in [0, 5, 10, 15, 20]:
            (checkpoints_dir / f"epoch_{epoch}").mkdir()

        logger = logging.getLogger("test_cleanup")
        logger.setLevel(logging.WARNING)

        cleanup_old_checkpoints(checkpoints_dir, keep_last=2, logger=logger)

        remaining_dirs = sorted([d.name for d in checkpoints_dir.glob("epoch_*")],
                               key=lambda x: int(x.split('_')[1]))

        # Should keep epochs 15 and 20 (highest epoch numbers)
        assert remaining_dirs == ['epoch_15', 'epoch_20']
