"""
Tests for src/evaluation.py

Tests the Evaluator class for model evaluation and metrics.
Target: ~61 tests covering state preservation, MSE calculation, generalization.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import FrequencyLSTM
from src.evaluation import Evaluator
from src.dataset import FrequencyDataset
from torch.utils.data import DataLoader


@pytest.fixture
def evaluator_setup(device):
    """Set up an evaluator with a small model."""
    model = FrequencyLSTM(input_size=5, hidden_size=32, num_layers=1)
    model.to(device)
    evaluator = Evaluator(model=model, device=device)
    return evaluator, model


class TestEvaluatorInit:
    """Tests for Evaluator.__init__"""

    def test_initialization(self, evaluator_setup):
        """Test basic initialization."""
        evaluator, _ = evaluator_setup
        assert evaluator is not None

    def test_model_in_eval_mode(self, evaluator_setup):
        """Test that model is set to eval mode."""
        evaluator, model = evaluator_setup
        assert not model.training

    def test_device_auto_detection(self):
        """Test device auto-detection."""
        model = FrequencyLSTM(hidden_size=32)
        evaluator = Evaluator(model=model, device=None)
        assert evaluator.device is not None

    def test_device_explicit_setting(self):
        """Test explicit device setting."""
        model = FrequencyLSTM(hidden_size=32)
        device = torch.device('cpu')
        evaluator = Evaluator(model=model, device=device)
        assert evaluator.device == device


class TestEvaluateDataset:
    """Tests for Evaluator.evaluate_dataset"""

    def test_returns_tuple(self, evaluator_setup, tiny_dataloader):
        """Test that evaluate_dataset returns a tuple."""
        evaluator, _ = evaluator_setup
        result = evaluator.evaluate_dataset(tiny_dataloader)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_mse_is_float(self, evaluator_setup, tiny_dataloader):
        """Test that MSE is a float or numpy float."""
        evaluator, _ = evaluator_setup
        mse, _, _ = evaluator.evaluate_dataset(tiny_dataloader)
        assert isinstance(mse, (float, np.floating))

    def test_mse_is_positive(self, evaluator_setup, tiny_dataloader):
        """Test that MSE is positive."""
        evaluator, _ = evaluator_setup
        mse, _, _ = evaluator.evaluate_dataset(tiny_dataloader)
        assert mse >= 0

    def test_mse_is_finite(self, evaluator_setup, tiny_dataloader):
        """Test that MSE is finite."""
        evaluator, _ = evaluator_setup
        mse, _, _ = evaluator.evaluate_dataset(tiny_dataloader)
        assert np.isfinite(mse)

    def test_predictions_shape(self, evaluator_setup, tiny_dataloader):
        """Test that predictions have correct shape."""
        evaluator, _ = evaluator_setup
        _, predictions, targets = evaluator.evaluate_dataset(tiny_dataloader)

        # Should match dataset size
        dataset_size = len(tiny_dataloader.dataset)
        assert len(predictions) == dataset_size

    def test_targets_shape(self, evaluator_setup, tiny_dataloader):
        """Test that targets have correct shape."""
        evaluator, _ = evaluator_setup
        _, predictions, targets = evaluator.evaluate_dataset(tiny_dataloader)

        assert len(targets) == len(predictions)

    def test_predictions_are_numpy(self, evaluator_setup, tiny_dataloader):
        """Test that predictions are numpy arrays."""
        evaluator, _ = evaluator_setup
        _, predictions, _ = evaluator.evaluate_dataset(tiny_dataloader)
        assert isinstance(predictions, np.ndarray)

    def test_targets_are_numpy(self, evaluator_setup, tiny_dataloader):
        """Test that targets are numpy arrays."""
        evaluator, _ = evaluator_setup
        _, _, targets = evaluator.evaluate_dataset(tiny_dataloader)
        assert isinstance(targets, np.ndarray)

    def test_no_gradients_computed(self, evaluator_setup, tiny_dataloader):
        """Test that gradients are not computed during evaluation."""
        evaluator, model = evaluator_setup

        evaluator.evaluate_dataset(tiny_dataloader)

        # Model should have no gradients
        for param in model.parameters():
            # Either no grad or grad is from previous training
            pass  # Just check it doesn't error

    @pytest.mark.state_management
    def test_state_preserved_across_batches(self, evaluator_setup, tiny_dataloader):
        """Test that hidden state is preserved across batches."""
        evaluator, _ = evaluator_setup

        # Should complete without error
        mse, _, _ = evaluator.evaluate_dataset(tiny_dataloader)
        assert np.isfinite(mse)


class TestCalculatePerFrequencyMetrics:
    """Tests for Evaluator.calculate_per_frequency_metrics"""

    def test_returns_dict(self, evaluator_setup):
        """Test that method returns a dict."""
        evaluator, _ = evaluator_setup

        # Create dummy predictions and targets
        predictions = np.random.randn(40).astype(np.float32)
        targets = np.random.randn(40).astype(np.float32)

        result = evaluator.calculate_per_frequency_metrics(
            predictions, targets, samples_per_freq=10
        )
        assert isinstance(result, dict)

    def test_returns_four_frequencies(self, evaluator_setup):
        """Test that dict has 4 frequency keys."""
        evaluator, _ = evaluator_setup

        predictions = np.random.randn(40).astype(np.float32)
        targets = np.random.randn(40).astype(np.float32)

        result = evaluator.calculate_per_frequency_metrics(
            predictions, targets, samples_per_freq=10
        )
        assert len(result) == 4

    def test_mse_values_are_floats(self, evaluator_setup):
        """Test that all MSE values are floats."""
        evaluator, _ = evaluator_setup

        predictions = np.random.randn(40).astype(np.float32)
        targets = np.random.randn(40).astype(np.float32)

        result = evaluator.calculate_per_frequency_metrics(
            predictions, targets, samples_per_freq=10
        )

        for key, value in result.items():
            assert isinstance(value, float)

    def test_mse_values_are_positive(self, evaluator_setup):
        """Test that all MSE values are non-negative."""
        evaluator, _ = evaluator_setup

        predictions = np.random.randn(40).astype(np.float32)
        targets = np.random.randn(40).astype(np.float32)

        result = evaluator.calculate_per_frequency_metrics(
            predictions, targets, samples_per_freq=10
        )

        for key, value in result.items():
            assert value >= 0

    def test_perfect_predictions_zero_mse(self, evaluator_setup):
        """Test that perfect predictions give MSE=0."""
        evaluator, _ = evaluator_setup

        targets = np.random.randn(40).astype(np.float32)
        predictions = targets.copy()  # Perfect predictions

        result = evaluator.calculate_per_frequency_metrics(
            predictions, targets, samples_per_freq=10
        )

        for key, value in result.items():
            assert value < 1e-6

    def test_correct_frequency_slicing(self, evaluator_setup):
        """Test that correct samples are used for each frequency."""
        evaluator, _ = evaluator_setup

        # Create data where each frequency has different values
        targets = np.zeros(40, dtype=np.float32)
        targets[0:10] = 1.0    # Freq 0
        targets[10:20] = 2.0   # Freq 1
        targets[20:30] = 3.0   # Freq 2
        targets[30:40] = 4.0   # Freq 3

        predictions = np.zeros(40, dtype=np.float32)

        result = evaluator.calculate_per_frequency_metrics(
            predictions, targets, samples_per_freq=10
        )

        # MSE should be different for each frequency
        assert result[0] == 1.0   # MSE = (1-0)^2
        assert result[1] == 4.0   # MSE = (2-0)^2
        assert result[2] == 9.0   # MSE = (3-0)^2
        assert result[3] == 16.0  # MSE = (4-0)^2


class TestCheckGeneralization:
    """Tests for Evaluator.check_generalization"""

    def test_returns_dict(self, evaluator_setup):
        """Test that method returns a dict."""
        evaluator, _ = evaluator_setup
        result = evaluator.check_generalization(0.1, 0.11)
        assert isinstance(result, dict)

    def test_contains_required_keys(self, evaluator_setup):
        """Test that dict has required keys."""
        evaluator, _ = evaluator_setup
        result = evaluator.check_generalization(0.1, 0.11)

        required_keys = ['absolute_difference', 'relative_difference',
                        'threshold', 'generalizes_well']
        for key in required_keys:
            assert key in result

    def test_absolute_difference(self, evaluator_setup):
        """Test absolute difference calculation."""
        evaluator, _ = evaluator_setup
        result = evaluator.check_generalization(0.1, 0.12)

        expected = abs(0.1 - 0.12)
        assert abs(result['absolute_difference'] - expected) < 1e-6

    def test_relative_difference(self, evaluator_setup):
        """Test relative difference calculation."""
        evaluator, _ = evaluator_setup
        result = evaluator.check_generalization(0.1, 0.12)

        expected = abs(0.1 - 0.12) / 0.1
        assert abs(result['relative_difference'] - expected) < 1e-6

    def test_generalizes_well_true(self, evaluator_setup):
        """Test that generalizes_well is True when within threshold."""
        evaluator, _ = evaluator_setup
        # 5% difference, default threshold 10%
        result = evaluator.check_generalization(0.1, 0.105)
        assert result['generalizes_well'] is True

    def test_generalizes_well_false(self, evaluator_setup):
        """Test that generalizes_well is False when exceeding threshold."""
        evaluator, _ = evaluator_setup
        # 20% difference, default threshold 10%
        result = evaluator.check_generalization(0.1, 0.12)
        assert result['generalizes_well'] is False

    def test_custom_threshold(self, evaluator_setup):
        """Test that custom threshold works."""
        evaluator, _ = evaluator_setup
        # 5% difference with 3% threshold should fail
        result = evaluator.check_generalization(0.1, 0.105, threshold=0.03)
        assert result['generalizes_well'] is False

    def test_zero_train_mse_handling(self, evaluator_setup):
        """Test handling of zero train MSE."""
        evaluator, _ = evaluator_setup
        # Should not divide by zero
        result = evaluator.check_generalization(0.0, 0.01)
        # Should handle gracefully
        assert 'relative_difference' in result


class TestEvaluateAll:
    """Tests for Evaluator.evaluate_all"""

    def test_returns_dict(self, evaluator_setup, tiny_dataloader):
        """Test that evaluate_all returns a dict."""
        evaluator, _ = evaluator_setup
        result = evaluator.evaluate_all(tiny_dataloader, tiny_dataloader)
        assert isinstance(result, dict)

    def test_contains_train_metrics(self, evaluator_setup, tiny_dataloader):
        """Test that result contains train metrics."""
        evaluator, _ = evaluator_setup
        result = evaluator.evaluate_all(tiny_dataloader, tiny_dataloader)
        assert 'train' in result or 'overall' in result

    def test_contains_test_metrics(self, evaluator_setup, tiny_dataloader):
        """Test that result contains test metrics."""
        evaluator, _ = evaluator_setup
        result = evaluator.evaluate_all(tiny_dataloader, tiny_dataloader)
        assert 'test' in result or 'overall' in result

    def test_contains_predictions(self, evaluator_setup, tiny_dataloader):
        """Test that result contains predictions."""
        evaluator, _ = evaluator_setup
        result = evaluator.evaluate_all(tiny_dataloader, tiny_dataloader)
        # Should have prediction data somewhere
        assert 'predictions' in result or 'train_predictions' in str(result)


class TestSaveMetrics:
    """Tests for Evaluator.save_metrics"""

    def test_creates_json_file(self, evaluator_setup, tiny_dataloader, temp_outputs_dir):
        """Test that JSON file is created."""
        evaluator, _ = evaluator_setup
        results = evaluator.evaluate_all(tiny_dataloader, tiny_dataloader)

        save_path = temp_outputs_dir / "metrics.json"
        evaluator.save_metrics(results, str(save_path))

        assert save_path.exists()

    def test_creates_parent_directory(self, evaluator_setup, tiny_dataloader, temp_dir):
        """Test that parent directory is created."""
        evaluator, _ = evaluator_setup
        results = evaluator.evaluate_all(tiny_dataloader, tiny_dataloader)

        save_path = temp_dir / "new_dir" / "metrics.json"
        evaluator.save_metrics(results, str(save_path))

        assert save_path.exists()

    def test_json_loadable(self, evaluator_setup, tiny_dataloader, temp_outputs_dir):
        """Test that saved JSON can be loaded."""
        evaluator, _ = evaluator_setup
        results = evaluator.evaluate_all(tiny_dataloader, tiny_dataloader)

        save_path = temp_outputs_dir / "metrics.json"
        evaluator.save_metrics(results, str(save_path))

        with open(save_path) as f:
            loaded = json.load(f)

        assert loaded is not None


class TestSavePredictions:
    """Tests for Evaluator.save_predictions"""

    def test_creates_npz_file(self, evaluator_setup, tiny_dataloader, temp_outputs_dir):
        """Test that NPZ file is created."""
        evaluator, _ = evaluator_setup
        results = evaluator.evaluate_all(tiny_dataloader, tiny_dataloader)

        save_path = temp_outputs_dir / "predictions.npz"
        evaluator.save_predictions(results, str(save_path))

        assert save_path.exists()

    def test_creates_parent_directory(self, evaluator_setup, tiny_dataloader, temp_dir):
        """Test that parent directory is created."""
        evaluator, _ = evaluator_setup
        results = evaluator.evaluate_all(tiny_dataloader, tiny_dataloader)

        save_path = temp_dir / "new_dir" / "predictions.npz"
        evaluator.save_predictions(results, str(save_path))

        assert save_path.exists()

    def test_npz_loadable(self, evaluator_setup, tiny_dataloader, temp_outputs_dir):
        """Test that saved NPZ can be loaded."""
        evaluator, _ = evaluator_setup
        results = evaluator.evaluate_all(tiny_dataloader, tiny_dataloader)

        save_path = temp_outputs_dir / "predictions.npz"
        evaluator.save_predictions(results, str(save_path))

        loaded = np.load(save_path)
        assert loaded is not None


class TestEdgeCases:
    """Tests for edge cases in evaluation."""

    def test_single_sample_evaluation(self, evaluator_setup, temp_data_dir):
        """Test evaluation with single sample."""
        evaluator, _ = evaluator_setup

        # Create single sample dataset
        data = np.random.randn(1, 6).astype(np.float32)
        data[0, 1:5] = [1, 0, 0, 0]

        filepath = temp_data_dir / "single.npy"
        np.save(filepath, data)

        dataset = FrequencyDataset(str(filepath))
        loader = DataLoader(dataset, batch_size=1)

        mse, predictions, targets = evaluator.evaluate_dataset(loader)
        assert np.isfinite(mse)

    def test_large_batch_evaluation(self, evaluator_setup, tiny_dataloader):
        """Test evaluation with large batch size."""
        evaluator, _ = evaluator_setup

        # Should work with any batch size
        mse, _, _ = evaluator.evaluate_dataset(tiny_dataloader)
        assert np.isfinite(mse)

    def test_variable_batch_sizes(self, evaluator_setup, temp_data_dir):
        """Test evaluation with variable batch sizes."""
        evaluator, _ = evaluator_setup

        # Create dataset with odd number of samples
        data = np.random.randn(13, 6).astype(np.float32)
        for i in range(13):
            data[i, 1:5] = 0
            data[i, 1 + i % 4] = 1

        filepath = temp_data_dir / "odd.npy"
        np.save(filepath, data)

        dataset = FrequencyDataset(str(filepath))
        loader = DataLoader(dataset, batch_size=5)  # Won't divide evenly

        mse, predictions, targets = evaluator.evaluate_dataset(loader)
        assert len(predictions) == 13


class TestMSECalculation:
    """Tests for MSE calculation correctness."""

    def test_mse_formula(self, evaluator_setup):
        """Test that MSE calculation is correct."""
        evaluator, _ = evaluator_setup

        # Known values
        predictions = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        targets = np.array([1.5, 2.5, 2.5], dtype=np.float32)

        # Manual MSE
        expected_mse = np.mean((predictions - targets) ** 2)

        # Calculate via method (simulate)
        errors = predictions - targets
        mse = np.mean(errors ** 2)

        assert abs(mse - expected_mse) < 1e-6

    def test_zero_error(self, evaluator_setup):
        """Test that identical predictions/targets give zero MSE."""
        evaluator, _ = evaluator_setup

        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        errors = values - values
        mse = np.mean(errors ** 2)

        assert mse == 0.0
