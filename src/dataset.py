"""
Dataset Module for LSTM Frequency Extraction

This module provides the PyTorch Dataset wrapper for loading and serving
the frequency extraction data generated in Phase 1.

Reference: prd/03_TRAINING_PIPELINE_PRD.md
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple


class FrequencyDataset(Dataset):
    """
    PyTorch Dataset wrapper for frequency extraction data.

    Data Structure:
        Each row in the dataset has 6 values:
        [S(t), C1, C2, C3, C4, Target_i(t)]
        - S(t): Noisy mixed signal (1 value)
        - C1-C4: One-hot frequency selector (4 values)
        - Target_i(t): Clean target sinusoid (1 value)

    This dataset is designed for L=1 training where each sample is processed
    individually with state preservation across the sequence.

    Args:
        data_path (str): Path to .npy file containing the dataset
            Expected shape: (40,000, 6) for full dataset
            Expected dtype: float32

    Example:
        >>> dataset = FrequencyDataset('data/train_data.npy')
        >>> print(len(dataset))  # 40000
        >>> input, target = dataset[0]
        >>> print(input.shape)  # torch.Size([5])
        >>> print(target.shape)  # torch.Size([1])
    """

    def __init__(self, data_path: str):
        """
        Initialize the dataset by loading from file.

        Args:
            data_path: Path to .npy file
        """
        # Load data from Phase 1
        self.data = np.load(data_path).astype(np.float32)

        print(f"Loaded dataset from {data_path}")
        print(f"  Shape: {self.data.shape}")
        print(f"  Size: {self.data.nbytes / 1024 / 1024:.2f} MB")

        # Validate shape
        if self.data.shape[1] != 6:
            raise ValueError(
                f"Expected 6 columns (S(t) + 4 one-hot + target), "
                f"got {self.data.shape[1]}"
            )

        # Split into inputs and targets
        # Input: [S(t), C1, C2, C3, C4] - columns 0-4
        # Target: [Target_i(t)] - column 5
        self.inputs = self.data[:, :5]    # Shape: (N, 5)
        self.targets = self.data[:, 5:6]  # Shape: (N, 1) - keep 2D for consistency

        print(f"  Inputs shape: {self.inputs.shape}")
        print(f"  Targets shape: {self.targets.shape}")

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples (40,000 for full dataset)
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            tuple containing:
                - input: Tensor of shape (5,) containing [S(t), C1, C2, C3, C4]
                - target: Tensor of shape (1,) containing [Target_i(t)]

        Example:
            >>> dataset = FrequencyDataset('data/train_data.npy')
            >>> input, target = dataset[0]
            >>> # input[0] is S(t) - noisy mixed signal
            >>> # input[1:5] is one-hot vector C
            >>> # target[0] is the clean target value
        """
        # Convert numpy arrays to tensors
        input_vec = torch.from_numpy(self.inputs[idx])    # Shape: (5,)
        target_val = torch.from_numpy(self.targets[idx])  # Shape: (1,)

        return input_vec, target_val

    def get_sample_info(self, idx: int) -> dict:
        """
        Get detailed information about a specific sample (for debugging).

        Args:
            idx: Index of the sample

        Returns:
            dict: Dictionary with sample details
        """
        input_vec, target_val = self[idx]

        # Determine which frequency is selected
        one_hot = input_vec[1:5]
        freq_idx = torch.argmax(one_hot).item()
        freq_map = {0: '1Hz', 1: '3Hz', 2: '5Hz', 3: '7Hz'}

        return {
            'index': idx,
            'noisy_signal': input_vec[0].item(),
            'selected_frequency': freq_map[freq_idx],
            'one_hot': one_hot.numpy(),
            'target': target_val.item()
        }


def main():
    """
    Test the FrequencyDataset with actual data from Phase 1.
    """
    print("=" * 70)
    print("FrequencyDataset - Test Run")
    print("=" * 70)
    print()

    # Test with training data
    print("Loading training dataset...")
    print("-" * 70)
    train_dataset = FrequencyDataset('data/train_data.npy')
    print()

    # Test __len__
    print(f"Dataset length: {len(train_dataset):,} samples")
    print()

    # Test __getitem__
    print("Sample 0 (first sample):")
    print("-" * 70)
    input_0, target_0 = train_dataset[0]
    print(f"Input shape: {input_0.shape}")
    print(f"Input values: {input_0.numpy()}")
    print(f"Target shape: {target_0.shape}")
    print(f"Target value: {target_0.item():.6f}")
    print()

    # Test get_sample_info
    info = train_dataset.get_sample_info(0)
    print("Sample info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()

    # Test samples from different frequencies
    print("Testing samples from different frequencies:")
    print("-" * 70)
    test_indices = [0, 10000, 20000, 30000]  # One from each frequency
    for idx in test_indices:
        info = train_dataset.get_sample_info(idx)
        print(f"Sample {idx:5d}: {info['selected_frequency']} - "
              f"Noisy={info['noisy_signal']:7.4f}, Target={info['target']:7.4f}")
    print()

    # Test with DataLoader (L=1 configuration)
    print("Testing with DataLoader (L=1 configuration):")
    print("-" * 70)
    from torch.utils.data import DataLoader

    loader = DataLoader(
        train_dataset,
        batch_size=1,      # CRITICAL: L=1 constraint
        shuffle=False,     # CRITICAL: preserve temporal order
        num_workers=0      # Avoid multiprocessing issues
    )

    print(f"DataLoader created with {len(loader):,} batches")
    print("First 3 batches:")

    for i, (inputs, targets) in enumerate(loader):
        if i >= 3:
            break
        print(f"  Batch {i}: input shape {inputs.shape}, target shape {targets.shape}")

    print()

    print("=" * 70)
    print("Dataset tests complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Implement StatefulTrainer in src/training.py")
    print("  2. Validate state management with lstm-state-debugger agent")
    print("  3. Begin training")


if __name__ == "__main__":
    main()
