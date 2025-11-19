"""
Data Generation Module for LSTM Frequency Extraction

This module implements the SignalGenerator class that creates synthetic mixed signals
with random noise patterns for training and testing the LSTM model.

Critical Implementation Note:
    Per-sample randomization is MANDATORY. Amplitude A_i(t) and phase φ_i(t) must
    vary at EVERY sample t, not vectorized. This forces the network to learn
    frequency structure rather than memorizing noise patterns.

Reference: prd/01_DATA_GENERATION_PRD.md
"""

import numpy as np
from typing import Tuple, List
from pathlib import Path


class SignalGenerator:
    """
    Generates synthetic mixed signals with random noise and clean targets
    for LSTM frequency extraction training.

    The generator creates signals with per-sample randomization of amplitude
    and phase, which is critical for forcing the LSTM to learn temporal patterns
    rather than memorizing static noise.

    Attributes:
        frequencies (List[float]): List of 4 frequencies in Hz [1, 3, 5, 7]
        fs (int): Sampling frequency in Hz (1000)
        duration (float): Signal duration in seconds (10.0)
        seed (int): Random seed for reproducibility (1 for train, 2 for test)
        n_samples (int): Total number of time samples (10,000)

    Example:
        >>> # Generate training dataset
        >>> gen = SignalGenerator(frequencies=[1,3,5,7], fs=1000, duration=10.0, seed=1)
        >>> gen.save_dataset('data/train_data.npy')
        >>> # Generate test dataset
        >>> gen = SignalGenerator(frequencies=[1,3,5,7], fs=1000, duration=10.0, seed=2)
        >>> gen.save_dataset('data/test_data.npy')
    """

    def __init__(
        self,
        frequencies: List[float] = None,
        fs: int = 1000,
        duration: float = 10.0,
        seed: int = 1
    ):
        """
        Initialize signal generator with parameters.

        Args:
            frequencies: List of frequencies in Hz. Defaults to [1, 3, 5, 7]
            fs: Sampling frequency in Hz. Default: 1000
            duration: Signal duration in seconds. Default: 10.0
            seed: Random seed for reproducibility. Use 1 for train, 2 for test

        Raises:
            ValueError: If parameters are invalid
        """
        if frequencies is None:
            frequencies = [1.0, 3.0, 5.0, 7.0]

        self.frequencies = frequencies
        self.fs = fs
        self.duration = duration
        self.seed = seed
        self.n_samples = int(fs * duration)

        # Validate parameters
        if len(self.frequencies) != 4:
            raise ValueError(f"Expected 4 frequencies, got {len(self.frequencies)}")
        if self.fs <= 0:
            raise ValueError(f"Sampling frequency must be positive, got {self.fs}")
        if self.duration <= 0:
            raise ValueError(f"Duration must be positive, got {self.duration}")

        # Set random seed for reproducibility
        np.random.seed(self.seed)

        print(f"SignalGenerator initialized:")
        print(f"  Frequencies: {self.frequencies} Hz")
        print(f"  Sampling rate: {self.fs} Hz")
        print(f"  Duration: {self.duration} sec")
        print(f"  Total samples: {self.n_samples}")
        print(f"  Random seed: {self.seed}")

    def generate_time_array(self) -> np.ndarray:
        """
        Generate time array from 0 to duration.

        Returns:
            np.ndarray: Time array of shape (n_samples,) ranging from 0 to duration
        """
        return np.linspace(0, self.duration, self.n_samples)

    def generate_noisy_sinusoid(
        self,
        freq: float,
        t_array: np.ndarray
    ) -> np.ndarray:
        """
        Generate a single noisy sinusoid with random amplitude and phase
        at EVERY sample.

        CRITICAL: This is the key pedagogical element. A_i(t) and φ_i(t) must
        be different for each time sample t. This is implemented using a loop,
        NOT vectorized operations.

        The formula for each sample:
            Sinus_i^noisy(t) = A_i(t) · sin(2π · f_i · t + φ_i(t))
        where:
            A_i(t) ~ Uniform(0.8, 1.2)   [random for EACH t]
            φ_i(t) ~ Uniform(0, 2π)      [random for EACH t]

        Args:
            freq: Frequency in Hz
            t_array: Time array of shape (n_samples,)

        Returns:
            np.ndarray: Noisy sinusoid of shape (n_samples,)
        """
        n = len(t_array)
        noisy_signal = np.zeros(n, dtype=np.float32)

        # CRITICAL: Loop over each sample to generate independent random values
        for i, t in enumerate(t_array):
            # Generate random amplitude and phase for THIS specific sample
            A_t = np.random.uniform(0.8, 1.2)
            phi_t = np.random.uniform(0, 0.01 * 2 * np.pi)

            # Compute noisy sinusoid at time t
            noisy_signal[i] = A_t * np.sin(2 * np.pi * freq * t + phi_t)

        return noisy_signal

    def generate_mixed_signal(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate mixed signal S(t) from 4 noisy sinusoids.

        The mixed signal is computed as:
            S(t) = (1/4) · Σ(i=1 to 4) Sinus_i^noisy(t)

        Returns:
            tuple:
                - t_array: Time points of shape (n_samples,)
                - S: Mixed signal of shape (n_samples,)
        """
        t_array = self.generate_time_array()

        # Generate all 4 noisy sinusoids
        noisy_sinusoids = []
        for freq in self.frequencies:
            noisy_sin = self.generate_noisy_sinusoid(freq, t_array)
            noisy_sinusoids.append(noisy_sin)

        # Sum and normalize by 1/4
        S = np.sum(noisy_sinusoids, axis=0) / 4.0

        return t_array, S

    def generate_clean_targets(self, t_array: np.ndarray) -> np.ndarray:
        """
        Generate clean target sinusoids (no noise).

        The clean targets have constant amplitude=1 and phase=0:
            Target_i(t) = sin(2π · f_i · t)

        Args:
            t_array: Time array of shape (n_samples,)

        Returns:
            np.ndarray: Clean targets of shape (4, n_samples)
                       targets[i, :] contains the clean sinusoid for frequency i
        """
        targets = []
        for freq in self.frequencies:
            # Pure sinusoid: amplitude=1, phase=0, no randomization
            target = np.sin(2 * np.pi * freq * t_array)
            targets.append(target)

        return np.array(targets, dtype=np.float32)  # Shape: (4, n_samples)

    def create_dataset(self) -> np.ndarray:
        """
        Create complete dataset with 40,000 rows.

        Dataset Structure:
            - 4 frequencies × 10,000 time samples = 40,000 rows
            - Each row: [S(t), C1, C2, C3, C4, Target_i(t)]
                - S(t): Noisy mixed signal value (1 scalar)
                - C1-C4: One-hot frequency selection vector (4 values)
                - Target_i(t): Clean target for selected frequency (1 scalar)

        Row Organization:
            - Rows 0-9,999: frequency f₁ (1Hz) with one-hot [1,0,0,0]
            - Rows 10,000-19,999: frequency f₂ (3Hz) with one-hot [0,1,0,0]
            - Rows 20,000-29,999: frequency f₃ (5Hz) with one-hot [0,0,1,0]
            - Rows 30,000-39,999: frequency f₄ (7Hz) with one-hot [0,0,0,1]

        Returns:
            np.ndarray: Dataset of shape (40,000, 6) with dtype float32
        """
        print("Generating dataset...")

        # Generate signals
        t_array, S = self.generate_mixed_signal()
        targets = self.generate_clean_targets(t_array)

        print(f"  Generated mixed signal: shape {S.shape}")
        print(f"  Generated clean targets: shape {targets.shape}")

        # Build dataset
        dataset = []

        for freq_idx in range(4):
            # Create one-hot vector for this frequency
            one_hot = np.zeros(4, dtype=np.float32)
            one_hot[freq_idx] = 1.0

            # Add all time samples for this frequency
            for t_idx in range(self.n_samples):
                row = np.concatenate([
                    [S[t_idx]],                      # Noisy mixed signal (1 value)
                    one_hot,                         # Frequency selection (4 values)
                    [targets[freq_idx, t_idx]]       # Clean target (1 value)
                ])
                dataset.append(row)

        dataset_array = np.array(dataset, dtype=np.float32)  # Shape: (40000, 6)

        print(f"  Dataset created: shape {dataset_array.shape}")
        print(f"  Memory size: {dataset_array.nbytes / 1024 / 1024:.2f} MB")

        return dataset_array

    def save_dataset(self, filepath: str) -> None:
        """
        Generate and save dataset to file.

        Creates the parent directory if it doesn't exist, generates the dataset,
        and saves it as a NumPy .npy file.

        Args:
            filepath: Path to save .npy file (e.g., 'data/train_data.npy')
        """
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Generate and save
        dataset = self.create_dataset()
        np.save(filepath, dataset)

        print(f"\nDataset saved to {filepath}")
        print(f"  Shape: {dataset.shape}")
        print(f"  Dtype: {dataset.dtype}")
        print(f"  Size: {dataset.nbytes / 1024 / 1024:.2f} MB")
        print(f"  Seed used: {self.seed}")


def main():
    """
    Main function to generate both training and test datasets.

    This creates:
        - data/train_data.npy (Seed #1)
        - data/test_data.npy (Seed #2)
    """
    print("="* 70)
    print("LSTM Frequency Extraction - Data Generation")
    print("=" * 70)
    print()

    # Generate training dataset (Seed #1)
    print("Generating TRAINING dataset (Seed #1)...")
    print("-" * 70)
    train_generator = SignalGenerator(
        frequencies=[1.0, 3.0, 5.0, 7.0],
        fs=1000,
        duration=10.0,
        seed=1
    )
    train_generator.save_dataset('data/train_data.npy')

    print()
    print("=" * 70)
    print()

    # Generate test dataset (Seed #2)
    print("Generating TEST dataset (Seed #2)...")
    print("-" * 70)
    test_generator = SignalGenerator(
        frequencies=[1.0, 3.0, 5.0, 7.0],
        fs=1000,
        duration=10.0,
        seed=2
    )
    test_generator.save_dataset('data/test_data.npy')

    print()
    print("=" * 70)
    print("Data generation complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Validate datasets using signal-validation-expert agent")
    print("  2. Verify FFT shows frequencies at 1, 3, 5, 7 Hz")
    print("  3. Confirm train/test have different noise (different seeds)")
    print("  4. Proceed to Phase 2: Model Architecture")


if __name__ == "__main__":
    main()
