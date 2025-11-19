"""
Visualization Module for LSTM Frequency Extraction

This module provides the Visualizer class for creating publication-quality
graphs demonstrating the LSTM's frequency extraction capability.

Reference: prd/05_VISUALIZATION_PRD.md
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Dict
import json
from scipy import stats
from scipy.fft import fft, fftfreq


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
        data_path: str = 'data/test_data.npy',
        training_history_path: str = 'models/training_history.json',
        metrics_path: str = 'outputs/metrics.json',
        train_data_path: str = 'data/train_data.npy'
    ):
        """
        Initialize the Visualizer.

        Args:
            predictions_path: Path to predictions from Phase 4 evaluation
            data_path: Path to test data for noisy signal
            training_history_path: Path to training history JSON
            metrics_path: Path to metrics JSON
            train_data_path: Path to training data
        """
        # Load predictions from Phase 4
        print(f"Loading predictions from {predictions_path}...")
        pred_data = np.load(predictions_path)
        self.test_predictions = pred_data['test_predictions']
        self.test_targets = pred_data['test_targets']
        self.train_predictions = pred_data['train_predictions']
        self.train_targets = pred_data['train_targets']

        # Load test data for noisy signal
        print(f"Loading test data from {data_path}...")
        test_data = np.load(data_path)
        self.test_noisy = test_data[:, 0]  # S(t) column
        self.test_data = test_data

        # Load training data
        print(f"Loading training data from {train_data_path}...")
        self.train_data = np.load(train_data_path)
        self.train_noisy = self.train_data[:, 0]

        # Load training history
        print(f"Loading training history from {training_history_path}...")
        with open(training_history_path, 'r') as f:
            self.training_history = json.load(f)

        # Load metrics
        print(f"Loading metrics from {metrics_path}...")
        with open(metrics_path, 'r') as f:
            self.metrics = json.load(f)

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
        print(f"  Training epochs: {len(self.training_history['train_loss'])}")
        print(f"  Metrics loaded: {list(self.metrics.keys())}")

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

    def plot_training_loss_curve(
        self,
        save_path: str = 'outputs/graphs/training_loss_curve.png'
    ) -> plt.Figure:
        """Create training loss curve with annotations."""
        losses = self.training_history['train_loss']
        epochs = list(range(1, len(losses) + 1))
        best_epoch = self.training_history['best_epoch']
        best_loss = self.training_history['best_loss']

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot loss
        ax.plot(epochs, losses, 'b-', linewidth=1.5, alpha=0.6, label='Training Loss')

        # Add smoothed curve
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(losses, sigma=3)
        ax.plot(epochs, smoothed, 'r-', linewidth=2, label='Smoothed Loss')

        # Annotate best epoch
        ax.plot(best_epoch, best_loss, 'g*', markersize=15, label=f'Best Model (Epoch {best_epoch})')
        ax.annotate(f'Best: {best_loss:.4f}',
                   xy=(best_epoch, best_loss),
                   xytext=(best_epoch-15, best_loss+0.01),
                   fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', color='green'))

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        return fig

    def plot_training_loss_log_scale(
        self,
        save_path: str = 'outputs/graphs/training_loss_log.png'
    ) -> plt.Figure:
        """Create training loss curve with log scale."""
        losses = self.training_history['train_loss']
        epochs = list(range(1, len(losses) + 1))

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.semilogy(epochs, losses, 'b-', linewidth=2)

        # Mark phases
        ax.axvline(x=20, color='r', linestyle='--', alpha=0.5, label='Rapid Descent → Plateau')
        ax.axvline(x=60, color='g', linestyle='--', alpha=0.5, label='Plateau → Fine-tuning')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (MSE) - Log Scale', fontsize=12)
        ax.set_title('Training Loss Curve (Log Scale)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, which='both')
        plt.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        return fig

    def plot_per_frequency_performance(
        self,
        save_path: str = 'outputs/graphs/per_frequency_performance.png'
    ) -> plt.Figure:
        """Create per-frequency performance bar chart."""
        train_mse = [self.metrics['per_frequency']['train'][str(i)] for i in range(4)]
        test_mse = [self.metrics['per_frequency']['test'][str(i)] for i in range(4)]

        x = np.arange(4)
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, train_mse, width, label='Train MSE', color='skyblue', edgecolor='black')
        bars2 = ax.bar(x + width/2, test_mse, width, label='Test MSE', color='lightcoral', edgecolor='black')

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Frequency', fontsize=12)
        ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
        ax.set_title('Per-Frequency Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{f}Hz' for f in self.frequencies])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        return fig

    def plot_generalization_analysis(
        self,
        save_path: str = 'outputs/graphs/generalization_analysis.png'
    ) -> plt.Figure:
        """Create generalization scatter plot."""
        train_mse = [self.metrics['per_frequency']['train'][str(i)] for i in range(4)]
        test_mse = [self.metrics['per_frequency']['test'][str(i)] for i in range(4)]

        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot points
        for i in range(4):
            ax.scatter(train_mse[i], test_mse[i], s=200, alpha=0.7,
                      label=f'{self.frequencies[i]}Hz')
            ax.annotate(f'{self.frequencies[i]}Hz',
                       (train_mse[i], test_mse[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=10)

        # Perfect generalization line
        max_val = max(max(train_mse), max(test_mse)) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', label='Perfect Generalization', linewidth=2)

        # 10% tolerance bands
        ax.plot([0, max_val], [0, max_val*1.1], 'r:', alpha=0.5, label='±10% Band')
        ax.plot([0, max_val], [0, max_val*0.9], 'r:', alpha=0.5)

        ax.set_xlabel('Train MSE', fontsize=12)
        ax.set_ylabel('Test MSE', fontsize=12)
        ax.set_title('Generalization Analysis\n(Points near diagonal = Good generalization)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        return fig

    def plot_error_distributions(
        self,
        save_path: str = 'outputs/graphs/error_distributions.png'
    ) -> plt.Figure:
        """Create error distribution histograms for each frequency."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for freq_idx in range(4):
            ax = axes[freq_idx]

            # Extract data for this frequency
            start_idx = freq_idx * self.n_samples_per_freq
            end_idx = (freq_idx + 1) * self.n_samples_per_freq

            predictions = self.test_predictions[start_idx:end_idx]
            targets = self.test_targets[start_idx:end_idx]
            errors = predictions - targets

            # Plot histogram
            ax.hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)

            # Overlay normal distribution
            mu, sigma = errors.mean(), errors.std()
            x = np.linspace(errors.min(), errors.max(), 100)
            ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Normal(μ={mu:.4f}, σ={sigma:.4f})')

            # Stats
            ax.axvline(mu, color='red', linestyle='--', linewidth=2, label=f'Mean = {mu:.4f}')
            ax.axvline(0, color='green', linestyle=':', linewidth=2, label='Zero Error')

            ax.set_title(f'f{freq_idx+1} = {self.frequencies[freq_idx]}Hz', fontsize=12, fontweight='bold')
            ax.set_xlabel('Prediction Error', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle('Error Distribution Analysis (Test Set)', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        return fig

    def plot_signal_composition(
        self,
        save_path: str = 'outputs/graphs/signal_composition.png',
        time_window: int = 1000
    ) -> plt.Figure:
        """Create signal composition breakdown visualization."""
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))

        time = self.time_full[:time_window]

        # Panel 1: All 4 clean sinusoids
        ax = axes[0]
        for freq_idx in range(4):
            start_idx = freq_idx * self.n_samples_per_freq
            clean = self.test_targets[start_idx:start_idx+time_window]
            ax.plot(time, clean, label=f'{self.frequencies[freq_idx]}Hz', linewidth=1.5, alpha=0.7)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_title('Clean Target Sinusoids (Individual Frequencies)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', ncol=4)
        ax.grid(True, alpha=0.3)

        # Panel 2: Noisy components
        ax = axes[1]
        for freq_idx in range(4):
            start_idx = freq_idx * self.n_samples_per_freq
            noisy_component = self.test_noisy[start_idx:start_idx+time_window]
            # Note: noisy signal is mixed, so we can't perfectly separate it
            # We'll show the mixed signal here
        ax.plot(time, self.test_noisy[:time_window], color='orange', linewidth=1, alpha=0.7)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_title('Mixed Noisy Signal S(t)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Panel 3: Example extraction (3Hz)
        ax = axes[2]
        freq_idx = 1  # 3Hz
        start_idx = freq_idx * self.n_samples_per_freq
        target = self.test_targets[start_idx:start_idx+time_window]
        prediction = self.test_predictions[start_idx:start_idx+time_window]
        ax.plot(time, target, 'b--', linewidth=2, label='Target (3Hz)', alpha=0.7)
        ax.plot(time, prediction, 'g-', linewidth=1.5, label='LSTM Extraction')
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_title('Extracted Frequency: 3Hz', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Panel 4: All extractions overlaid
        ax = axes[3]
        for freq_idx in range(4):
            start_idx = freq_idx * self.n_samples_per_freq
            prediction = self.test_predictions[start_idx:start_idx+time_window]
            ax.plot(time, prediction, label=f'{self.frequencies[freq_idx]}Hz Extracted',
                   linewidth=1, alpha=0.7)
        ax.set_xlabel('Time (seconds)', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_title('All Extracted Frequencies (LSTM Outputs)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', ncol=4)
        ax.grid(True, alpha=0.3)

        fig.suptitle('Signal Composition and Extraction Pipeline', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        return fig

    def plot_fft_analysis(
        self,
        save_path: str = 'outputs/graphs/fft_analysis.png'
    ) -> plt.Figure:
        """Create FFT analysis visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # Analyze each frequency
        for freq_idx in range(4):
            ax = axes[freq_idx]
            start_idx = freq_idx * self.n_samples_per_freq
            end_idx = start_idx + self.n_samples_per_freq

            # Get signals
            target = self.test_targets[start_idx:end_idx]
            prediction = self.test_predictions[start_idx:end_idx]
            noisy = self.test_noisy[start_idx:end_idx]

            # Compute FFT
            N = len(target)
            T = 1.0 / 1000.0  # sampling interval
            freqs = fftfreq(N, T)[:N//2]

            target_fft = np.abs(fft(target))[:N//2] / N
            prediction_fft = np.abs(fft(prediction))[:N//2] / N
            noisy_fft = np.abs(fft(noisy))[:N//2] / N

            # Plot
            ax.plot(freqs, noisy_fft, 'r-', alpha=0.3, linewidth=1, label='Noisy Mixed Signal')
            ax.plot(freqs, target_fft, 'b--', linewidth=2, label='Target')
            ax.plot(freqs, prediction_fft, 'g-', linewidth=1.5, label='LSTM Prediction')

            # Mark expected frequency
            ax.axvline(x=self.frequencies[freq_idx], color='purple', linestyle=':', linewidth=2,
                      label=f'Expected: {self.frequencies[freq_idx]}Hz')

            ax.set_xlim([0, 10])
            ax.set_xlabel('Frequency (Hz)', fontsize=10)
            ax.set_ylabel('Magnitude', fontsize=10)
            ax.set_title(f'FFT Analysis: {self.frequencies[freq_idx]}Hz Extraction', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle('Frequency Domain Analysis (FFT)', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        return fig

    def plot_error_over_time(
        self,
        save_path: str = 'outputs/graphs/error_over_time.png'
    ) -> plt.Figure:
        """Create error over time visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for freq_idx in range(4):
            ax = axes[freq_idx]
            start_idx = freq_idx * self.n_samples_per_freq
            end_idx = start_idx + self.n_samples_per_freq

            predictions = self.test_predictions[start_idx:end_idx]
            targets = self.test_targets[start_idx:end_idx]
            errors = np.abs(predictions - targets)

            # Plot rolling mean of error
            window = 100
            rolling_mean = np.convolve(errors, np.ones(window)/window, mode='valid')
            time_rolling = self.time_full[window-1:]

            ax.plot(self.time_full, errors, 'b-', alpha=0.3, linewidth=0.5, label='Absolute Error')
            ax.plot(time_rolling, rolling_mean, 'r-', linewidth=2, label=f'Rolling Mean (window={window})')

            mean_error = errors.mean()
            ax.axhline(y=mean_error, color='g', linestyle='--', linewidth=2, label=f'Mean = {mean_error:.4f}')

            ax.set_xlabel('Time (seconds)', fontsize=10)
            ax.set_ylabel('Absolute Error', fontsize=10)
            ax.set_title(f'{self.frequencies[freq_idx]}Hz - Error Over Time', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle('Prediction Error Over Time (Test Set)', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        return fig

    def plot_training_dashboard(
        self,
        save_path: str = 'outputs/graphs/training_dashboard.png'
    ) -> plt.Figure:
        """Create comprehensive training dashboard."""
        losses = np.array(self.training_history['train_loss'])
        times = np.array(self.training_history['epoch_times'])
        epochs = np.arange(1, len(losses) + 1)

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Panel 1: Loss vs Epoch
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, losses, 'b-', linewidth=2)
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(losses, sigma=3)
        ax1.plot(epochs, smoothed, 'r-', linewidth=2, label='Smoothed')
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss (MSE)', fontsize=11)
        ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: Loss Improvement Rate
        ax2 = fig.add_subplot(gs[0, 1])
        improvement = -np.diff(losses)  # negative because we want reduction
        ax2.plot(epochs[1:], improvement, 'g-', linewidth=2)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Loss Reduction', fontsize=11)
        ax2.set_title('Loss Improvement Rate (Negative = Getting Worse)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Panel 3: Cumulative Training Time
        ax3 = fig.add_subplot(gs[1, 0])
        cumulative_time = np.cumsum(times)
        ax3.plot(epochs, cumulative_time, 'purple', linewidth=2)
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Cumulative Time (seconds)', fontsize=11)
        ax3.set_title(f'Training Time (Total: {cumulative_time[-1]:.1f}s)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Panel 4: Efficiency (Loss per unit time)
        ax4 = fig.add_subplot(gs[1, 1])
        efficiency = losses / cumulative_time
        ax4.plot(epochs, efficiency, 'orange', linewidth=2)
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('Loss / Total Time', fontsize=11)
        ax4.set_title('Training Efficiency', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        fig.suptitle('Training Efficiency Dashboard', fontsize=16, fontweight='bold')

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        return fig

    def create_all_visualizations(
        self,
        output_dir: str = 'outputs/graphs'
    ) -> None:
        """
        Create all visualizations for the assignment.

        This creates 12+ graphs covering training analysis, performance metrics,
        error analysis, and signal processing validations.

        Args:
            output_dir: Directory to save graphs
        """
        print("=" * 70)
        print("COMPREHENSIVE VISUALIZATION GENERATION")
        print("=" * 70)
        print()

        # === ORIGINAL GRAPHS ===
        print("=" * 70)
        print("PART 1: Original Frequency Extraction Visualizations")
        print("=" * 70)

        print("\n1. Creating frequency comparison graph...")
        self.plot_single_frequency_comparison(
            freq_idx=1,
            save_path=f"{output_dir}/frequency_comparison.png"
        )

        print("\n2. Creating all frequencies graph...")
        self.plot_all_frequencies(
            save_path=f"{output_dir}/all_frequencies.png"
        )

        # === TRAINING ANALYSIS ===
        print("\n" + "=" * 70)
        print("PART 2: Training Analysis")
        print("=" * 70)

        print("\n3. Creating training loss curve...")
        self.plot_training_loss_curve(
            save_path=f"{output_dir}/training_loss_curve.png"
        )

        print("\n4. Creating training loss (log scale)...")
        self.plot_training_loss_log_scale(
            save_path=f"{output_dir}/training_loss_log.png"
        )

        print("\n5. Creating training efficiency dashboard...")
        self.plot_training_dashboard(
            save_path=f"{output_dir}/training_dashboard.png"
        )

        # === PERFORMANCE ANALYSIS ===
        print("\n" + "=" * 70)
        print("PART 3: Performance Analysis")
        print("=" * 70)

        print("\n6. Creating per-frequency performance chart...")
        self.plot_per_frequency_performance(
            save_path=f"{output_dir}/per_frequency_performance.png"
        )

        print("\n7. Creating generalization analysis...")
        self.plot_generalization_analysis(
            save_path=f"{output_dir}/generalization_analysis.png"
        )

        # === ERROR ANALYSIS ===
        print("\n" + "=" * 70)
        print("PART 4: Error Analysis")
        print("=" * 70)

        print("\n8. Creating error distributions...")
        self.plot_error_distributions(
            save_path=f"{output_dir}/error_distributions.png"
        )

        print("\n9. Creating error over time...")
        self.plot_error_over_time(
            save_path=f"{output_dir}/error_over_time.png"
        )

        # === SIGNAL ANALYSIS ===
        print("\n" + "=" * 70)
        print("PART 5: Signal Processing Analysis")
        print("=" * 70)

        print("\n10. Creating signal composition breakdown...")
        self.plot_signal_composition(
            save_path=f"{output_dir}/signal_composition.png"
        )

        print("\n11. Creating FFT analysis...")
        self.plot_fft_analysis(
            save_path=f"{output_dir}/fft_analysis.png"
        )

        print("\n" + "=" * 70)
        print("ALL VISUALIZATIONS COMPLETE!")
        print("=" * 70)
        print(f"\nTotal graphs created: 11")
        print(f"Saved to: {output_dir}/")
        print("\nGenerated files:")
        print("  [Original]")
        print("    - frequency_comparison.png")
        print("    - all_frequencies.png")
        print("  [Training Analysis]")
        print("    - training_loss_curve.png")
        print("    - training_loss_log.png")
        print("    - training_dashboard.png")
        print("  [Performance]")
        print("    - per_frequency_performance.png")
        print("    - generalization_analysis.png")
        print("  [Error Analysis]")
        print("    - error_distributions.png")
        print("    - error_over_time.png")
        print("  [Signal Processing]")
        print("    - signal_composition.png")
        print("    - fft_analysis.png")


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
