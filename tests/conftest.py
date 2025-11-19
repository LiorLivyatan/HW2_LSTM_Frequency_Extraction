"""
Shared pytest fixtures for the LSTM Frequency Extraction test suite.

This module provides common fixtures used across multiple test files.
"""

import pytest
import numpy as np
import torch
import tempfile
import json
import yaml
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation import SignalGenerator
from src.dataset import FrequencyDataset
from src.model import FrequencyLSTM


# =============================================================================
# Basic Configuration Fixtures
# =============================================================================

@pytest.fixture
def sample_frequencies():
    """Standard frequencies used in the assignment."""
    return [1.0, 3.0, 5.0, 7.0]


@pytest.fixture
def default_config():
    """Default configuration dictionary for tests."""
    return {
        'data': {
            'frequencies': [1, 3, 5, 7],
            'sampling_rate': 1000,
            'duration': 10.0,
            'train_seed': 42,
            'test_seed': 99
        },
        'model': {
            'input_size': 5,
            'hidden_size': 128,
            'num_layers': 1,
            'dropout': 0.0
        },
        'training': {
            'learning_rate': 0.0001,
            'num_epochs': 100,
            'batch_size': 32,
            'device': 'cpu'
        },
        'paths': {
            'train_data': 'data/train_data.npy',
            'test_data': 'data/test_data.npy',
            'model_checkpoint': 'models/best_model.pth'
        }
    }


@pytest.fixture
def small_config():
    """Small configuration for fast testing."""
    return {
        'data': {
            'frequencies': [1, 3, 5, 7],
            'sampling_rate': 100,  # Smaller for speed
            'duration': 0.1,  # Much shorter
            'train_seed': 42,
            'test_seed': 99
        },
        'model': {
            'input_size': 5,
            'hidden_size': 32,  # Smaller
            'num_layers': 1,
            'dropout': 0.0
        },
        'training': {
            'learning_rate': 0.001,
            'num_epochs': 2,  # Very few epochs
            'batch_size': 4,
            'device': 'cpu'
        }
    }


# =============================================================================
# Directory and Path Fixtures
# =============================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test files."""
    return tmp_path


@pytest.fixture
def temp_data_dir(tmp_path):
    """Temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def temp_models_dir(tmp_path):
    """Temporary models directory."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return models_dir


@pytest.fixture
def temp_outputs_dir(tmp_path):
    """Temporary outputs directory."""
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    return outputs_dir


# =============================================================================
# Data Generation Fixtures
# =============================================================================

@pytest.fixture
def signal_generator():
    """Standard SignalGenerator instance."""
    return SignalGenerator(
        frequencies=[1.0, 3.0, 5.0, 7.0],
        fs=1000,
        duration=10.0,
        seed=42
    )


@pytest.fixture
def small_signal_generator():
    """Small SignalGenerator for fast testing."""
    return SignalGenerator(
        frequencies=[1.0, 3.0, 5.0, 7.0],
        fs=100,
        duration=0.1,
        seed=42
    )


@pytest.fixture
def tiny_dataset():
    """Tiny synthetic dataset (40 samples) for fast tests."""
    np.random.seed(42)
    # 4 frequencies x 10 samples = 40 total
    n_samples = 10
    n_freqs = 4
    total_rows = n_samples * n_freqs

    dataset = np.zeros((total_rows, 6), dtype=np.float32)

    # Generate data for each frequency
    for freq_idx in range(n_freqs):
        start_idx = freq_idx * n_samples
        end_idx = start_idx + n_samples

        # Mixed signal (random for testing)
        dataset[start_idx:end_idx, 0] = np.random.randn(n_samples)

        # One-hot encoding
        dataset[start_idx:end_idx, 1:5] = 0
        dataset[start_idx:end_idx, 1 + freq_idx] = 1

        # Target (random for testing)
        dataset[start_idx:end_idx, 5] = np.random.randn(n_samples)

    return dataset


@pytest.fixture
def small_dataset(small_signal_generator, temp_data_dir):
    """Small dataset created by SignalGenerator."""
    filepath = temp_data_dir / "small_data.npy"
    dataset = small_signal_generator.create_dataset()
    np.save(filepath, dataset)
    return filepath, dataset


@pytest.fixture
def full_size_dataset_shape():
    """Expected shape for full-size dataset."""
    return (40000, 6)


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def small_model():
    """Small LSTM model for fast testing."""
    return FrequencyLSTM(
        input_size=5,
        hidden_size=32,
        num_layers=1,
        dropout=0.0
    )


@pytest.fixture
def default_model():
    """Default LSTM model with standard parameters."""
    return FrequencyLSTM(
        input_size=5,
        hidden_size=128,
        num_layers=1,
        dropout=0.0
    )


@pytest.fixture
def multi_layer_model():
    """Multi-layer LSTM model for testing."""
    return FrequencyLSTM(
        input_size=5,
        hidden_size=64,
        num_layers=2,
        dropout=0.1
    )


@pytest.fixture
def trained_small_model(small_model):
    """Small model with random weights (simulates trained)."""
    # Just return the model - weights are already random
    return small_model


# =============================================================================
# DataLoader Fixtures
# =============================================================================

@pytest.fixture
def tiny_dataloader(tiny_dataset, temp_data_dir):
    """Tiny DataLoader for fast testing."""
    from torch.utils.data import DataLoader

    # Save dataset to file
    filepath = temp_data_dir / "tiny_data.npy"
    np.save(filepath, tiny_dataset)

    # Create dataset and loader
    dataset = FrequencyDataset(str(filepath))
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    return loader


@pytest.fixture
def small_dataloader(small_dataset):
    """Small DataLoader for testing."""
    from torch.utils.data import DataLoader

    filepath, _ = small_dataset
    dataset = FrequencyDataset(str(filepath))
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    return loader


# =============================================================================
# Training Fixtures
# =============================================================================

@pytest.fixture
def criterion():
    """Standard MSE loss criterion."""
    return torch.nn.MSELoss()


@pytest.fixture
def optimizer(small_model):
    """Adam optimizer for small model."""
    return torch.optim.Adam(small_model.parameters(), lr=0.001)


@pytest.fixture
def device():
    """CPU device for testing."""
    return torch.device('cpu')


# =============================================================================
# Metrics and Results Fixtures
# =============================================================================

@pytest.fixture
def sample_metrics():
    """Sample metrics dictionary for testing."""
    return {
        'overall': {
            'mse_train': 0.0993,
            'mse_test': 0.0994
        },
        'per_frequency': {
            'train': {
                '1Hz': 0.0098,
                '3Hz': 0.0414,
                '5Hz': 0.0439,
                '7Hz': 0.1974
            },
            'test': {
                '1Hz': 0.0103,
                '3Hz': 0.0224,
                '5Hz': 0.0427,
                '7Hz': 0.1964
            }
        },
        'generalization': {
            'absolute_difference': 0.0001,
            'relative_difference': 0.001,
            'threshold': 0.1,
            'generalizes_well': True
        }
    }


@pytest.fixture
def sample_predictions(tiny_dataset):
    """Sample predictions for testing."""
    n_samples = len(tiny_dataset)
    return {
        'train_predictions': np.random.randn(n_samples).astype(np.float32),
        'train_targets': tiny_dataset[:, 5],
        'test_predictions': np.random.randn(n_samples).astype(np.float32),
        'test_targets': tiny_dataset[:, 5]
    }


@pytest.fixture
def sample_training_history():
    """Sample training history for testing."""
    return {
        'train_loss': [0.5, 0.3, 0.2, 0.15, 0.1],
        'epoch_times': [1.0, 1.1, 1.0, 1.2, 1.1],
        'best_epoch': 4,
        'best_loss': 0.1,
        'total_time': 5.4
    }


# =============================================================================
# File Content Fixtures
# =============================================================================

@pytest.fixture
def sample_config_yaml(temp_dir, default_config):
    """Create a sample config.yaml file."""
    config_path = temp_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f)
    return config_path


@pytest.fixture
def sample_metrics_json(temp_dir, sample_metrics):
    """Create a sample metrics.json file."""
    metrics_path = temp_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(sample_metrics, f)
    return metrics_path


@pytest.fixture
def sample_predictions_npz(temp_dir, sample_predictions):
    """Create a sample predictions.npz file."""
    predictions_path = temp_dir / "predictions.npz"
    np.savez(predictions_path, **sample_predictions)
    return predictions_path


@pytest.fixture
def sample_history_json(temp_dir, sample_training_history):
    """Create a sample training_history.json file."""
    history_path = temp_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(sample_training_history, f)
    return history_path


# =============================================================================
# Checkpoint Fixtures
# =============================================================================

@pytest.fixture
def sample_checkpoint(temp_models_dir, small_model, optimizer):
    """Create a sample model checkpoint."""
    checkpoint_path = temp_models_dir / "checkpoint.pth"

    checkpoint = {
        'epoch': 10,
        'model_state_dict': small_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': 0.05,
        'history': {
            'train_loss': [0.5, 0.3, 0.2, 0.15, 0.1],
            'epoch_times': [1.0, 1.1, 1.0, 1.2, 1.1]
        }
    }

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


# =============================================================================
# Tensor Fixtures
# =============================================================================

@pytest.fixture
def sample_input_tensor():
    """Sample input tensor for model testing."""
    # Shape: (batch_size=4, seq_len=1, features=5)
    return torch.randn(4, 1, 5)


@pytest.fixture
def sample_target_tensor():
    """Sample target tensor for model testing."""
    # Shape: (batch_size=4, 1)
    return torch.randn(4, 1)


@pytest.fixture
def single_sample_input():
    """Single sample input for L=1 testing."""
    # Shape: (batch_size=1, seq_len=1, features=5)
    return torch.randn(1, 1, 5)


# =============================================================================
# State Management Fixtures
# =============================================================================

@pytest.fixture
def initial_hidden_state(small_model, device):
    """Initial hidden state for LSTM."""
    batch_size = 4
    return small_model.init_hidden(batch_size, device)


# =============================================================================
# Helper Functions (as fixtures)
# =============================================================================

@pytest.fixture
def assert_tensor_shape():
    """Helper to assert tensor shapes."""
    def _assert(tensor, expected_shape):
        assert tensor.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {tensor.shape}"
    return _assert


@pytest.fixture
def assert_close():
    """Helper to assert numerical closeness."""
    def _assert(a, b, rtol=1e-5, atol=1e-8):
        if isinstance(a, torch.Tensor):
            a = a.detach().numpy()
        if isinstance(b, torch.Tensor):
            b = b.detach().numpy()
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    return _assert


# =============================================================================
# Markers for Test Categories
# =============================================================================

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "critical: mark test as critical for core functionality"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (requires full training)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "state_management: mark test as state management test"
    )
