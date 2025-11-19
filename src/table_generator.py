"""
Table Generator Module for LSTM Frequency Extraction

This module provides the TableGenerator class for creating markdown tables
that summarize dataset statistics, model performance, and per-frequency metrics.

These tables are designed to be included in the assignment report and provide
a clear, professional summary of the experimental results.

Reference: prd/05_VISUALIZATION_PRD.md
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple


class TableGenerator:
    """
    TableGenerator creates markdown tables for the assignment report.

    Generates three types of tables:
    1. Dataset Statistics Table - Overview of training/test datasets
    2. Performance Summary Table - Overall MSE metrics and generalization
    3. Per-Frequency Metrics Table - Detailed breakdown by frequency

    Args:
        predictions_path: Path to predictions.npz file
        metrics_path: Path to metrics.json file
        train_data_path: Path to training dataset
        test_data_path: Path to test dataset

    Example:
        >>> generator = TableGenerator()
        >>> generator.create_all_tables(output_dir='outputs/tables')
    """

    def __init__(
        self,
        predictions_path: str = 'outputs/predictions.npz',
        metrics_path: str = 'outputs/metrics.json',
        train_data_path: str = 'data/train_data.npy',
        test_data_path: str = 'data/test_data.npy'
    ):
        """
        Initialize the TableGenerator.

        Args:
            predictions_path: Path to predictions file
            metrics_path: Path to metrics file
            train_data_path: Path to training data
            test_data_path: Path to test data
        """
        # Load predictions
        self.predictions = np.load(predictions_path)

        # Load metrics
        with open(metrics_path, 'r') as f:
            self.metrics = json.load(f)

        # Load datasets
        self.train_data = np.load(train_data_path)
        self.test_data = np.load(test_data_path)

        # Frequency information
        self.frequencies = [1, 3, 5, 7]  # Hz
        self.samples_per_freq = 10000

        print("TableGenerator initialized:")
        print(f"  Train data shape: {self.train_data.shape}")
        print(f"  Test data shape: {self.test_data.shape}")
        print(f"  Metrics loaded: {list(self.metrics.keys())}")

    def generate_dataset_statistics_table(self) -> str:
        """
        Generate dataset statistics table.

        Shows:
        - Dataset split (train/test)
        - Number of samples
        - Frequencies covered
        - Input/output dimensions
        - Random seeds used

        Returns:
            str: Markdown table as string
        """
        lines = []
        lines.append("# Dataset Statistics")
        lines.append("")
        lines.append("| Property | Training Set | Test Set |")
        lines.append("|----------|--------------|----------|")

        # Total samples
        train_samples = self.train_data.shape[0]
        test_samples = self.test_data.shape[0]
        lines.append(f"| Total Samples | {train_samples:,} | {test_samples:,} |")

        # Samples per frequency
        lines.append(f"| Samples per Frequency | {self.samples_per_freq:,} | {self.samples_per_freq:,} |")

        # Number of frequencies
        num_freq = len(self.frequencies)
        lines.append(f"| Number of Frequencies | {num_freq} | {num_freq} |")

        # Frequency values
        freq_str = ", ".join([f"{f}Hz" for f in self.frequencies])
        lines.append(f"| Frequencies | {freq_str} | {freq_str} |")

        # Input dimension
        input_dim = self.train_data.shape[1] - 1  # Exclude target column
        lines.append(f"| Input Dimension | {input_dim} | {input_dim} |")

        # Output dimension
        lines.append("| Output Dimension | 1 (scalar) | 1 (scalar) |")

        # Random seeds (from config)
        lines.append("| Random Seed | 42 | 99 |")

        # Data format
        lines.append("| Data Format | `.npy` (NumPy) | `.npy` (NumPy) |")

        lines.append("")
        lines.append("## Dataset Structure")
        lines.append("")
        lines.append("Each row contains:")
        lines.append("- **S(t)**: Noisy mixed signal (1 value)")
        lines.append("- **C**: One-hot frequency selector (4 values)")
        lines.append("- **Target**: Clean target sinusoid (1 value)")
        lines.append("")
        lines.append("**Total row format**: `[S(t), C1, C2, C3, C4, Target]` (6 values)")
        lines.append("")

        # Signal statistics
        lines.append("## Signal Statistics")
        lines.append("")
        lines.append("| Statistic | Training Set | Test Set |")
        lines.append("|-----------|--------------|----------|")

        # Input signal (S(t)) statistics - column 0
        train_signal = self.train_data[:, 0]
        test_signal = self.test_data[:, 0]

        lines.append(f"| Input Mean | {np.mean(train_signal):.6f} | {np.mean(test_signal):.6f} |")
        lines.append(f"| Input Std Dev | {np.std(train_signal):.6f} | {np.std(test_signal):.6f} |")
        lines.append(f"| Input Min | {np.min(train_signal):.6f} | {np.min(test_signal):.6f} |")
        lines.append(f"| Input Max | {np.max(train_signal):.6f} | {np.max(test_signal):.6f} |")

        # Target statistics - last column
        train_target = self.train_data[:, -1]
        test_target = self.test_data[:, -1]

        lines.append(f"| Target Mean | {np.mean(train_target):.6f} | {np.mean(test_target):.6f} |")
        lines.append(f"| Target Std Dev | {np.std(train_target):.6f} | {np.std(test_target):.6f} |")
        lines.append(f"| Target Min | {np.min(train_target):.6f} | {np.min(test_target):.6f} |")
        lines.append(f"| Target Max | {np.max(train_target):.6f} | {np.max(test_target):.6f} |")

        lines.append("")

        return "\n".join(lines)

    def generate_performance_summary_table(self) -> str:
        """
        Generate performance summary table.

        Shows:
        - Overall MSE (train/test)
        - Generalization metrics
        - Pass/fail status

        Returns:
            str: Markdown table as string
        """
        lines = []
        lines.append("# Performance Summary")
        lines.append("")

        # Overall metrics
        lines.append("## Overall MSE Performance")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")

        mse_train = self.metrics['overall']['mse_train']
        mse_test = self.metrics['overall']['mse_test']

        lines.append(f"| Training MSE | {mse_train:.6f} |")
        lines.append(f"| Test MSE | {mse_test:.6f} |")

        lines.append("")

        # Generalization analysis
        lines.append("## Generalization Analysis")
        lines.append("")
        lines.append("| Metric | Value | Threshold | Status |")
        lines.append("|--------|-------|-----------|--------|")

        gen = self.metrics['generalization']
        abs_diff = gen['absolute_difference']
        rel_diff = gen['relative_difference']
        threshold = gen['threshold']
        generalizes = gen['generalizes_well']

        status = "✓ PASS" if generalizes else "✗ FAIL"

        lines.append(f"| Absolute Difference | {abs_diff:.6f} | - | - |")
        lines.append(f"| Relative Difference | {rel_diff:.2%} | {threshold:.2%} | {status} |")

        lines.append("")

        # Interpretation
        lines.append("## Interpretation")
        lines.append("")
        if generalizes:
            lines.append("**Result**: The model generalizes well to unseen data.")
            lines.append("")
            lines.append("The relative difference between training and test MSE is below the threshold, ")
            lines.append("indicating that the model learned the underlying frequency structure rather than ")
            lines.append("memorizing noise patterns. This demonstrates successful generalization.")
        else:
            lines.append("**Result**: The model shows signs of overfitting.")
            lines.append("")
            lines.append("The relative difference between training and test MSE exceeds the threshold, ")
            lines.append("suggesting that the model may have memorized training set noise rather than ")
            lines.append("learning the underlying frequency structure.")

        lines.append("")

        # Quality assessment
        lines.append("## Quality Assessment")
        lines.append("")
        lines.append("| Criterion | Target | Actual | Status |")
        lines.append("|-----------|--------|--------|--------|")

        # Target: MSE < 0.01
        target_mse = 0.01
        mse_status = "✓ PASS" if mse_test < target_mse else "✗ FAIL"
        lines.append(f"| Test MSE < 0.01 | {target_mse:.6f} | {mse_test:.6f} | {mse_status} |")

        # Target: Generalization < 10%
        gen_status = "✓ PASS" if generalizes else "✗ FAIL"
        lines.append(f"| Relative Diff < 10% | {threshold:.2%} | {rel_diff:.2%} | {gen_status} |")

        lines.append("")

        return "\n".join(lines)

    def generate_per_frequency_metrics_table(self) -> str:
        """
        Generate per-frequency metrics table.

        Shows detailed MSE breakdown for each of the 4 frequencies,
        comparing training and test performance.

        Returns:
            str: Markdown table as string
        """
        lines = []
        lines.append("# Per-Frequency Performance Metrics")
        lines.append("")
        lines.append("Detailed breakdown of MSE for each frequency component.")
        lines.append("")

        # Main table
        lines.append("| Frequency | Training MSE | Test MSE | Absolute Diff | Relative Diff | Generalization |")
        lines.append("|-----------|--------------|----------|---------------|---------------|----------------|")

        per_freq_train = self.metrics['per_frequency']['train']
        per_freq_test = self.metrics['per_frequency']['test']

        for idx, freq in enumerate(self.frequencies):
            mse_train = per_freq_train[str(idx)]  # JSON keys are strings
            mse_test = per_freq_test[str(idx)]

            abs_diff = abs(mse_test - mse_train)
            rel_diff = abs_diff / mse_train if mse_train > 0 else 0

            # Generalization status (using 10% threshold)
            gen_status = "✓ Good" if rel_diff < 0.1 else "✗ Poor"

            lines.append(f"| f{idx+1} = {freq}Hz | {mse_train:.6f} | {mse_test:.6f} | "
                        f"{abs_diff:.6f} | {rel_diff:.2%} | {gen_status} |")

        lines.append("")

        # Statistics summary
        lines.append("## Summary Statistics")
        lines.append("")

        train_mses = [per_freq_train[str(i)] for i in range(4)]
        test_mses = [per_freq_test[str(i)] for i in range(4)]

        lines.append("| Statistic | Training MSE | Test MSE |")
        lines.append("|-----------|--------------|----------|")
        lines.append(f"| Mean | {np.mean(train_mses):.6f} | {np.mean(test_mses):.6f} |")
        lines.append(f"| Std Dev | {np.std(train_mses):.6f} | {np.std(test_mses):.6f} |")
        lines.append(f"| Min | {np.min(train_mses):.6f} | {np.min(test_mses):.6f} |")
        lines.append(f"| Max | {np.max(train_mses):.6f} | {np.max(test_mses):.6f} |")

        lines.append("")

        # Best and worst performing frequencies
        lines.append("## Performance Rankings")
        lines.append("")

        # Sort by test MSE
        freq_performance = [(self.frequencies[i], test_mses[i], i) for i in range(4)]
        freq_performance.sort(key=lambda x: x[1])

        lines.append("### Best to Worst (by Test MSE)")
        lines.append("")
        lines.append("| Rank | Frequency | Test MSE |")
        lines.append("|------|-----------|----------|")

        for rank, (freq, mse, idx) in enumerate(freq_performance, 1):
            lines.append(f"| {rank} | f{idx+1} = {freq}Hz | {mse:.6f} |")

        lines.append("")

        # Identify problematic frequencies
        lines.append("## Analysis")
        lines.append("")

        # Find frequencies with poor generalization
        poor_gen = []
        for idx, freq in enumerate(self.frequencies):
            mse_train = per_freq_train[str(idx)]
            mse_test = per_freq_test[str(idx)]
            rel_diff = abs(mse_test - mse_train) / mse_train if mse_train > 0 else 0

            if rel_diff >= 0.1:
                poor_gen.append((freq, rel_diff))

        if poor_gen:
            lines.append("**Frequencies with Poor Generalization** (relative difference ≥ 10%):")
            lines.append("")
            for freq, rel_diff in poor_gen:
                lines.append(f"- **{freq}Hz**: {rel_diff:.2%} relative difference")
            lines.append("")
        else:
            lines.append("**All frequencies show good generalization** (relative difference < 10%)")
            lines.append("")

        # Find frequencies with high MSE
        high_mse = []
        for idx, freq in enumerate(self.frequencies):
            mse_test = per_freq_test[str(idx)]
            if mse_test > 0.01:
                high_mse.append((freq, mse_test))

        if high_mse:
            lines.append("**Frequencies with High Test MSE** (> 0.01):")
            lines.append("")
            for freq, mse in high_mse:
                lines.append(f"- **{freq}Hz**: MSE = {mse:.6f}")
            lines.append("")
        else:
            lines.append("**All frequencies achieve target MSE** (< 0.01)")
            lines.append("")

        return "\n".join(lines)

    def create_all_tables(self, output_dir: str = 'outputs/tables') -> None:
        """
        Generate all three markdown tables and save to files.

        Creates:
        - dataset_statistics.md
        - performance_summary.md
        - per_frequency_metrics.md

        Args:
            output_dir: Directory to save markdown files
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("Generating Markdown Tables")
        print("=" * 70)
        print()

        # Table 1: Dataset Statistics
        print("1. Generating Dataset Statistics Table...")
        table1 = self.generate_dataset_statistics_table()
        table1_path = output_path / 'dataset_statistics.md'
        with open(table1_path, 'w') as f:
            f.write(table1)
        print(f"   Saved to: {table1_path}")

        # Table 2: Performance Summary
        print("2. Generating Performance Summary Table...")
        table2 = self.generate_performance_summary_table()
        table2_path = output_path / 'performance_summary.md'
        with open(table2_path, 'w') as f:
            f.write(table2)
        print(f"   Saved to: {table2_path}")

        # Table 3: Per-Frequency Metrics
        print("3. Generating Per-Frequency Metrics Table...")
        table3 = self.generate_per_frequency_metrics_table()
        table3_path = output_path / 'per_frequency_metrics.md'
        with open(table3_path, 'w') as f:
            f.write(table3)
        print(f"   Saved to: {table3_path}")

        print()
        print("=" * 70)
        print("All tables generated successfully!")
        print("=" * 70)
        print()
        print(f"Tables saved to: {output_dir}/")
        print("  - dataset_statistics.md")
        print("  - performance_summary.md")
        print("  - per_frequency_metrics.md")


def main():
    """
    Test the TableGenerator.

    This demonstrates the table generation pipeline and creates
    all three markdown tables.
    """
    print("=" * 70)
    print("TableGenerator - Test Run")
    print("=" * 70)
    print()

    # Create generator
    generator = TableGenerator()
    print()

    # Generate all tables
    generator.create_all_tables()

    print()
    print("=" * 70)
    print("Table generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
