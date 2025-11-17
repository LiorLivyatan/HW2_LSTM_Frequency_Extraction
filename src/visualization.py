"""
Visualization Module for LSTM Frequency Extraction

This module provides the Visualizer class for creating publication-quality
graphs demonstrating the LSTM's frequency extraction capability.

Reference: prd/05_VISUALIZATION_PRD.md
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple


class Visualizer:
    """
    Create visualizations for LSTM frequency extraction results.

    This class generates two key graphs:
    1. Single frequency comparison (Target vs LSTM vs Noisy)
    2. All 4 frequencies in a 2×2 grid

    Args:
        predictions_path: Path to predictions.npz from Phase 4
        data_path: Path to test_data.npy for noisy signal

    Example:
        >>> visualizer = Visualizer()
        >>> visualizer.create_all_visualizations()
    """

    def __init__(
        self,
        predictions_path: str = 'outputs/predictions.npz',
        data_path: str = 'data/test_data.npy'
    ):
        """
        Initialize the Visualizer.

        Args:
            predictions_path: Path to predictions from Phase 4 evaluation
            data_path: Path to test data for noisy signal
        """
        # Load predictions from Phase 4
        print(f"Loading predictions from {predictions_path}...")
        pred_data = np.load(predictions_path)
        self.test_predictions = pred_data['test_predictions']
        self.test_targets = pred_data['test_targets']

        # Load test data for noisy signal
        print(f"Loading test data from {data_path}...")
        test_data = np.load(data_path)
        self.test_noisy = test_data[:, 0]  # S(t) column

        # Data structure: 40,000 rows = 4 frequencies × 10,000 samples
        self.n_samples_per_freq = 10000
        self.n_frequencies = 4
        self.frequencies = [1, 3, 5, 7]  # Hz

        # Time array (0-10 seconds, 10,000 points)
        self.time_full = np.linspace(0, 10, self.n_samples_per_freq)

        print(f"  Predictions shape: {self.test_predictions.shape}")
        print(f"  Targets shape: {self.test_targets.shape}")
        print(f"  Noisy signal shape: {self.test_noisy.shape}")
        print(f"  Frequencies: {self.frequencies} Hz")

    def extract_frequency_data(
        self,
        freq_idx: int,
        time_window: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract data for a specific frequency.

        Data is organized as:
        - Rows 0-9,999: frequency f₁ (1Hz)
        - Rows 10,000-19,999: frequency f₂ (3Hz)
        - Rows 20,000-29,999: frequency f₃ (5Hz)
        - Rows 30,000-39,999: frequency f₄ (7Hz)

        Args:
            freq_idx: Frequency index (0-3)
            time_window: Number of samples to include (None = all 10,000)

        Returns:
            tuple: (time, target, prediction, noisy)
        """
        # Calculate row indices for this frequency
        start_idx = freq_idx * self.n_samples_per_freq
        end_idx = (freq_idx + 1) * self.n_samples_per_freq

        # Extract data
        target = self.test_targets[start_idx:end_idx]
        prediction = self.test_predictions[start_idx:end_idx]
        noisy = self.test_noisy[start_idx:end_idx]
        time = self.time_full.copy()

        # Apply time window if specified
        if time_window is not None:
            target = target[:time_window]
            prediction = prediction[:time_window]
            noisy = noisy[:time_window]
            time = time[:time_window]

        return time, target, prediction, noisy

    def plot_single_frequency_comparison(
        self,
        freq_idx: int = 1,  # Default: f₂ = 3Hz
        time_window: int = 1000,  # First 1 second
        save_path: str = 'outputs/graphs/frequency_comparison.png'
    ) -> plt.Figure:
        """
        Create Graph 1: Single frequency comparison.

        Shows target, LSTM output, and noisy input overlaid on the same plot.
        This graph demonstrates the LSTM's ability to extract a clean sinusoid
        from the noisy mixed signal.

        Args:
            freq_idx: Which frequency to plot (0-3)
            time_window: Number of samples to show (1000 = 1 second)
            save_path: Where to save the figure

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Extract data
        time, target, prediction, noisy = self.extract_frequency_data(
            freq_idx,
            time_window
        )

        freq_hz = self.frequencies[freq_idx]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot noisy input (background, semi-transparent)
        ax.plot(
            time,
            noisy,
            color='red',
            alpha=0.3,
            linewidth=1,
            label='Noisy Mixed Input S(t)'
        )

        # Plot target (ground truth)
        ax.plot(
            time,
            target,
            color='blue',
            linewidth=2,
            label='Ground Truth Target',
            linestyle='--'
        )

        # Plot LSTM output
        ax.plot(
            time,
            prediction,
            color='green',
            linewidth=1.5,
            label='LSTM Output',
            marker='.',
            markersize=3,
            markevery=10  # Show markers every 10 points
        )

        # Formatting
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title(
            f'Frequency Extraction: f{freq_idx+1} = {freq_hz}Hz (Test Set)',
            fontsize=14,
            fontweight='bold'
        )
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Tight layout
        plt.tight_layout()

        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

        return fig

    def plot_all_frequencies(
        self,
        time_window: int = 1000,
        save_path: str = 'outputs/graphs/all_frequencies.png'
    ) -> plt.Figure:
        """
        Create Graph 2: All four frequencies in 2×2 grid.

        Each subplot shows target vs LSTM output for one frequency.
        This provides a comprehensive view of the model's extraction
        capability across all frequencies.

        Args:
            time_window: Number of samples to show (1000 = 1 second)
            save_path: Where to save the figure

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Create figure with 2×2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()  # Flatten to 1D for easier indexing

        for freq_idx in range(4):
            ax = axes[freq_idx]

            # Extract data
            time, target, prediction, _ = self.extract_frequency_data(
                freq_idx,
                time_window
            )

            freq_hz = self.frequencies[freq_idx]

            # Plot target
            ax.plot(
                time,
                target,
                color='blue',
                linewidth=2,
                label='Target',
                linestyle='--',
                alpha=0.7
            )

            # Plot LSTM output
            ax.plot(
                time,
                prediction,
                color='green',
                linewidth=1.5,
                label='LSTM Output'
            )

            # Formatting
            ax.set_title(
                f'f{freq_idx+1} = {freq_hz}Hz',
                fontsize=12,
                fontweight='bold'
            )
            ax.set_xlabel('Time (seconds)', fontsize=10)
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)

        # Overall title
        fig.suptitle(
            'Extracted Frequencies (Test Set)',
            fontsize=16,
            fontweight='bold',
            y=0.995
        )

        # Tight layout
        plt.tight_layout()

        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

        return fig

    def create_all_visualizations(
        self,
        output_dir: str = 'outputs/graphs'
    ) -> None:
        """
        Create all required visualizations.

        This is the main entry point that creates both:
        1. Graph 1: Single frequency comparison (f₂ = 3Hz)
        2. Graph 2: All 4 frequencies in 2×2 grid

        Args:
            output_dir: Directory to save graphs
        """
        print("=" * 70)
        print("Phase 5: Creating Visualizations")
        print("=" * 70)
        print()

        # Graph 1: Single frequency comparison
        print("1. Creating frequency comparison graph...")
        print("   Frequency: f₂ = 3Hz")
        print("   Time window: 0-1 second (1000 samples)")
        self.plot_single_frequency_comparison(
            freq_idx=1,  # f₂ = 3Hz
            save_path=f"{output_dir}/frequency_comparison.png"
        )

        print()

        # Graph 2: All frequencies
        print("2. Creating all frequencies graph...")
        print("   Layout: 2×2 grid")
        print("   Frequencies: 1Hz, 3Hz, 5Hz, 7Hz")
        print("   Time window: 0-1 second (1000 samples)")
        self.plot_all_frequencies(
            save_path=f"{output_dir}/all_frequencies.png"
        )

        print()
        print("=" * 70)
        print("Visualization Complete!")
        print("=" * 70)
        print(f"\nGraphs saved to: {output_dir}/")
        print("  - frequency_comparison.png")
        print("  - all_frequencies.png")


def main():
    """
    Test the Visualizer with predictions from Phase 4.

    This creates all required visualizations for the assignment.
    """
    print("=" * 70)
    print("Visualizer - Test Run")
    print("=" * 70)
    print()

    # Create visualizer
    visualizer = Visualizer(
        predictions_path='outputs/predictions.npz',
        data_path='data/test_data.npy'
    )

    print()

    # Create all graphs
    visualizer.create_all_visualizations(output_dir='outputs/graphs')

    print()
    print("=" * 70)
    print("Visualization complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
