"""
Evaluation Module for LSTM Frequency Extraction

This module provides the Evaluator class for calculating MSE metrics and
verifying generalization of the trained LSTM model.

IMPORTANT: Evaluation uses the same L=1 state preservation pattern as training,
where hidden state is manually preserved across all samples in the dataset.

Reference: prd/04_EVALUATION_PRD.md
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import json
from typing import Dict, Tuple


class Evaluator:
    """
    Evaluator for FrequencyLSTM with proper state management.

    Calculates MSE metrics and verifies generalization by evaluating
    the model on both training and test sets with state preservation.

    Args:
        model: Trained FrequencyLSTM model
        device: Device for inference ('cpu' or 'cuda')

    Example:
        >>> model = FrequencyLSTM(hidden_size=128)
        >>> checkpoint = torch.load('models/best_model.pth')
        >>> model.load_state_dict(checkpoint['model_state_dict'])
        >>> evaluator = Evaluator(model)
        >>> results = evaluator.evaluate_all(train_loader, test_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None
    ):
        """
        Initialize the Evaluator.

        Args:
            model: Trained LSTM model
            device: Device for inference (auto-detected if None)
        """
        # Auto-detect device if not specified
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(device)
        self.model.eval()  # Set to evaluation mode (disables dropout, etc.)
        self.device = device

        print(f"Evaluator initialized:")
        print(f"  Device: {self.device}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    @torch.no_grad()  # Disable gradient computation for efficiency
    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        dataset_name: str = 'dataset'
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate model on a dataset with state preservation.

        CRITICAL: Uses same state preservation pattern as training:
        - Initialize hidden_state = None at start
        - Preserve state across all batches
        - Handle variable batch sizes
        - Detach state after each forward pass

        Args:
            dataloader: DataLoader (any batch_size, shuffle=False)
            dataset_name: Name for progress bar

        Returns:
            tuple:
                - mse: Mean squared error
                - predictions: Array of predictions (N,)
                - targets: Array of ground truth (N,)
        """
        # Storage for results
        all_predictions = []
        all_targets = []

        # Initialize state (same as training)
        hidden_state = None
        batch_size = dataloader.batch_size

        # Progress bar
        pbar = tqdm(
            dataloader,
            desc=f"Evaluating {dataset_name}",
            ncols=80
        )

        for inputs, targets in pbar:
            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Get current batch size (might be smaller for last batch)
            current_batch_size = inputs.size(0)

            # Handle variable batch sizes
            hidden_state = self.model.get_or_reset_hidden(
                current_batch_size=current_batch_size,
                expected_batch_size=batch_size,
                hidden=hidden_state,
                device=self.device
            )

            # Reshape for LSTM: (batch, seq_len, features)
            inputs = inputs.unsqueeze(1)  # (batch_size, 1, 5)

            # Forward pass with state preservation
            output, hidden_state = self.model(inputs, hidden_state)

            # CRITICAL: Detach state (same as training)
            hidden_state = tuple(h.detach() for h in hidden_state)

            # Store results (move to CPU and flatten)
            all_predictions.append(output.cpu().numpy().flatten())
            all_targets.append(targets.cpu().numpy().flatten())

        # Convert lists to numpy arrays
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)

        # Calculate MSE
        mse = np.mean((predictions - targets) ** 2)

        print(f"  MSE: {mse:.6f}")

        return mse, predictions, targets

    def calculate_per_frequency_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        samples_per_freq: int = 10000
    ) -> Dict[int, float]:
        """
        Calculate MSE for each frequency separately.

        Data structure (from Phase 1):
        - Total: 40,000 samples = 4 frequencies × 10,000 samples
        - Rows 0-9,999: frequency f₁ (1Hz)
        - Rows 10,000-19,999: frequency f₂ (3Hz)
        - Rows 20,000-29,999: frequency f₃ (5Hz)
        - Rows 30,000-39,999: frequency f₄ (7Hz)

        Args:
            predictions: Predictions array (40,000,)
            targets: Targets array (40,000,)
            samples_per_freq: Samples per frequency (default: 10,000)

        Returns:
            dict: {frequency_idx: mse} for each of 4 frequencies
        """
        per_freq_mse = {}

        for freq_idx in range(4):
            start_idx = freq_idx * samples_per_freq
            end_idx = (freq_idx + 1) * samples_per_freq

            freq_predictions = predictions[start_idx:end_idx]
            freq_targets = targets[start_idx:end_idx]

            mse = np.mean((freq_predictions - freq_targets) ** 2)
            per_freq_mse[freq_idx] = float(mse)

        return per_freq_mse

    def check_generalization(
        self,
        mse_train: float,
        mse_test: float,
        threshold: float = 0.1
    ) -> Dict:
        """
        Check if model generalizes well.

        Good generalization means:
        - MSE_test ≈ MSE_train (model learned frequencies, not noise)
        - Relative difference < threshold (default: 10%)

        Args:
            mse_train: MSE on training set
            mse_test: MSE on test set
            threshold: Maximum acceptable relative difference (default: 0.1 = 10%)

        Returns:
            dict: Generalization analysis with pass/fail status
        """
        abs_diff = abs(mse_test - mse_train)
        rel_diff = abs_diff / mse_train if mse_train > 0 else float('inf')

        generalizes_well = rel_diff < threshold

        return {
            'mse_train': float(mse_train),
            'mse_test': float(mse_test),
            'absolute_difference': float(abs_diff),
            'relative_difference': float(rel_diff),
            'threshold': float(threshold),
            'generalizes_well': bool(generalizes_well)  # Convert numpy bool to Python bool
        }

    def evaluate_all(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader
    ) -> Dict:
        """
        Comprehensive evaluation on both train and test sets.

        This is the main evaluation entry point that:
        1. Evaluates on training set
        2. Evaluates on test set
        3. Calculates per-frequency metrics
        4. Checks generalization
        5. Compiles and prints results

        Args:
            train_loader: DataLoader for training set
            test_loader: DataLoader for test set

        Returns:
            dict: Complete metrics including predictions for visualization
        """
        print("=" * 70)
        print("Phase 4: Model Evaluation")
        print("=" * 70)

        # Evaluate training set
        print("\n1. Evaluating Training Set...")
        print("-" * 70)
        mse_train, pred_train, target_train = self.evaluate_dataset(
            train_loader,
            dataset_name='Training Set'
        )

        # Per-frequency metrics (train)
        per_freq_train = self.calculate_per_frequency_metrics(
            pred_train,
            target_train
        )

        # Evaluate test set
        print("\n2. Evaluating Test Set...")
        print("-" * 70)
        mse_test, pred_test, target_test = self.evaluate_dataset(
            test_loader,
            dataset_name='Test Set'
        )

        # Per-frequency metrics (test)
        per_freq_test = self.calculate_per_frequency_metrics(
            pred_test,
            target_test
        )

        # Generalization check
        print("\n3. Checking Generalization...")
        print("-" * 70)
        generalization = self.check_generalization(mse_train, mse_test)

        # Compile results
        results = {
            'overall': {
                'mse_train': float(mse_train),
                'mse_test': float(mse_test)
            },
            'per_frequency': {
                'train': per_freq_train,
                'test': per_freq_test
            },
            'generalization': generalization,
            'predictions': {
                'train': {
                    'predictions': pred_train.tolist(),
                    'targets': target_train.tolist()
                },
                'test': {
                    'predictions': pred_test.tolist(),
                    'targets': target_test.tolist()
                }
            }
        }

        # Print summary
        self.print_summary(results)

        return results

    def print_summary(self, results: Dict) -> None:
        """
        Print human-readable summary of evaluation results.

        Args:
            results: Evaluation results dictionary
        """
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)

        # Overall metrics
        print("\nOverall Performance:")
        print(f"  MSE (Training):  {results['overall']['mse_train']:.6f}")
        print(f"  MSE (Test):      {results['overall']['mse_test']:.6f}")

        # Per-frequency (train)
        print("\nPer-Frequency MSE (Training):")
        frequencies = [1, 3, 5, 7]
        for idx, freq in enumerate(frequencies):
            mse = results['per_frequency']['train'][idx]
            print(f"  f{idx+1} = {freq}Hz:  {mse:.6f}")

        # Per-frequency (test)
        print("\nPer-Frequency MSE (Test):")
        for idx, freq in enumerate(frequencies):
            mse = results['per_frequency']['test'][idx]
            print(f"  f{idx+1} = {freq}Hz:  {mse:.6f}")

        # Generalization
        gen = results['generalization']
        print("\nGeneralization Analysis:")
        print(f"  Absolute Difference: {gen['absolute_difference']:.6f}")
        print(f"  Relative Difference: {gen['relative_difference']:.2%}")
        print(f"  Threshold:           {gen['threshold']:.2%}")

        status = "PASS ✓" if gen['generalizes_well'] else "FAIL ✗"
        print(f"  Status: {status}")

        print("=" * 70)

    def save_metrics(self, results: Dict, save_path: str) -> None:
        """
        Save metrics to JSON file.

        Note: Predictions are excluded from saved metrics (too large).
        Use save_predictions() to save predictions separately.

        Args:
            results: Evaluation results dictionary
            save_path: Path to save JSON file
        """
        # Remove predictions from saved metrics (too large for JSON)
        save_results = {
            'overall': results['overall'],
            'per_frequency': results['per_frequency'],
            'generalization': results['generalization']
        }

        # Create output directory if needed
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Save as formatted JSON
        with open(save_path, 'w') as f:
            json.dump(save_results, f, indent=2)

        print(f"\nMetrics saved to: {save_path}")

    def save_predictions(
        self,
        results: Dict,
        save_path: str
    ) -> None:
        """
        Save predictions and targets to numpy file for visualization.

        Args:
            results: Evaluation results with predictions
            save_path: Path to save .npz file
        """
        # Create output directory if needed
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Save as compressed numpy archive
        np.savez(
            save_path,
            train_predictions=results['predictions']['train']['predictions'],
            train_targets=results['predictions']['train']['targets'],
            test_predictions=results['predictions']['test']['predictions'],
            test_targets=results['predictions']['test']['targets']
        )

        print(f"Predictions saved to: {save_path}")


def main():
    """
    Test the Evaluator with actual trained model.

    This demonstrates the evaluation pipeline and verifies that
    all components work correctly.
    """
    print("=" * 70)
    print("Evaluator - Test Run")
    print("=" * 70)
    print()

    # Import dependencies
    from src.model import FrequencyLSTM
    from src.dataset import FrequencyDataset

    # Load model (using default hidden_size=128 from model definition)
    print("Loading trained model...")
    model = FrequencyLSTM(
        input_size=5,
        hidden_size=128,
        num_layers=1
    )

    checkpoint_path = 'models/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"  Loaded from: {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Training loss: {checkpoint['loss']:.6f}")
    print()

    # Load datasets
    print("Loading datasets...")
    train_dataset = FrequencyDataset('data/train_data.npy')
    test_dataset = FrequencyDataset('data/test_data.npy')
    print()

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    # Create evaluator
    print("Creating evaluator...")
    evaluator = Evaluator(model)
    print()

    # Evaluate
    results = evaluator.evaluate_all(train_loader, test_loader)

    # Save results
    print("\nSaving results...")
    evaluator.save_metrics(results, 'outputs/metrics.json')
    evaluator.save_predictions(results, 'outputs/predictions.npz')

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
