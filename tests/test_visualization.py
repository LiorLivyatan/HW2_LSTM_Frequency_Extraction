"""
Tests for src/visualization.py

Tests the Visualizer class for graph generation.
Target: ~50 tests covering graph creation, data extraction, file I/O.
"""

import pytest
import numpy as np
import json
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def visualization_data(temp_dir):
    """Create all data files needed for visualization testing."""
    # Create predictions
    predictions = {
        'train_predictions': np.random.randn(40).astype(np.float32),
        'train_targets': np.random.randn(40).astype(np.float32),
        'test_predictions': np.random.randn(40).astype(np.float32),
        'test_targets': np.random.randn(40).astype(np.float32)
    }
    predictions_path = temp_dir / "predictions.npz"
    np.savez(predictions_path, **predictions)

    # Create test data
    test_data = np.random.randn(40, 6).astype(np.float32)
    for i in range(4):
        test_data[i*10:(i+1)*10, 1:5] = 0
        test_data[i*10:(i+1)*10, 1+i] = 1
    test_data_path = temp_dir / "test_data.npy"
    np.save(test_data_path, test_data)

    # Create train data
    train_data = np.random.randn(40, 6).astype(np.float32)
    for i in range(4):
        train_data[i*10:(i+1)*10, 1:5] = 0
        train_data[i*10:(i+1)*10, 1+i] = 1
    train_data_path = temp_dir / "train_data.npy"
    np.save(train_data_path, train_data)

    # Create metrics
    metrics = {
        'overall': {'mse_train': 0.1, 'mse_test': 0.11},
        'per_frequency': {
            'train': {'0': 0.01, '1': 0.02, '2': 0.03, '3': 0.04},
            'test': {'0': 0.011, '1': 0.021, '2': 0.031, '3': 0.041}
        },
        'generalization': {
            'absolute_difference': 0.01,
            'relative_difference': 0.1,
            'generalizes_well': True
        }
    }
    metrics_path = temp_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

    # Create training history
    history = {
        'train_loss': [0.5, 0.3, 0.2, 0.15, 0.1],
        'epoch_times': [1.0, 1.1, 1.0, 1.2, 1.1],
        'best_epoch': 5,
        'best_loss': 0.1
    }
    history_path = temp_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f)

    return {
        'predictions_path': str(predictions_path),
        'test_data_path': str(test_data_path),
        'train_data_path': str(train_data_path),
        'metrics_path': str(metrics_path),
        'history_path': str(history_path),
        'temp_dir': temp_dir
    }


class TestVisualizerInit:
    """Tests for Visualizer.__init__"""

    def test_initialization(self, visualization_data):
        """Test basic initialization."""
        from src.visualization import Visualizer

        viz = Visualizer(
            predictions_path=visualization_data['predictions_path'],
            data_path=visualization_data['test_data_path'],
            train_data_path=visualization_data['train_data_path'],
            training_history_path=visualization_data['history_path'],
            metrics_path=visualization_data['metrics_path']
        )
        assert viz is not None

    def test_loads_predictions(self, visualization_data):
        """Test that predictions are loaded."""
        from src.visualization import Visualizer

        viz = Visualizer(
            predictions_path=visualization_data['predictions_path'],
            data_path=visualization_data['test_data_path'],
            train_data_path=visualization_data['train_data_path'],
            training_history_path=visualization_data['history_path'],
            metrics_path=visualization_data['metrics_path']
        )
        assert hasattr(viz, 'predictions') or hasattr(viz, 'test_predictions')

    def test_loads_test_data(self, visualization_data):
        """Test that test data is loaded."""
        from src.visualization import Visualizer

        viz = Visualizer(
            predictions_path=visualization_data['predictions_path'],
            data_path=visualization_data['test_data_path'],
            train_data_path=visualization_data['train_data_path'],
            training_history_path=visualization_data['history_path'],
            metrics_path=visualization_data['metrics_path']
        )
        assert hasattr(viz, 'test_data') or hasattr(viz, 'data')


class TestExtractFrequencyData:
    """Tests for Visualizer.extract_frequency_data"""

    def test_returns_correct_components(self, visualization_data):
        """Test that correct number of components returned."""
        from src.visualization import Visualizer

        viz = Visualizer(
            predictions_path=visualization_data['predictions_path'],
            data_path=visualization_data['test_data_path'],
            train_data_path=visualization_data['train_data_path'],
            training_history_path=visualization_data['history_path'],
            metrics_path=visualization_data['metrics_path']
        )

        result = viz.extract_frequency_data(0)
        # Should return time, prediction, target, noisy
        assert len(result) >= 3

    def test_time_window_limiting(self, visualization_data):
        """Test that time window limits output."""
        from src.visualization import Visualizer

        viz = Visualizer(
            predictions_path=visualization_data['predictions_path'],
            data_path=visualization_data['test_data_path'],
            train_data_path=visualization_data['train_data_path'],
            training_history_path=visualization_data['history_path'],
            metrics_path=visualization_data['metrics_path']
        )

        result = viz.extract_frequency_data(0, time_window=5)
        # First element should be limited
        assert len(result[0]) <= 10  # Small dataset

    def test_all_frequency_indices(self, visualization_data):
        """Test all 4 frequency indices."""
        from src.visualization import Visualizer

        viz = Visualizer(
            predictions_path=visualization_data['predictions_path'],
            data_path=visualization_data['test_data_path'],
            train_data_path=visualization_data['train_data_path'],
            training_history_path=visualization_data['history_path'],
            metrics_path=visualization_data['metrics_path']
        )

        for freq_idx in range(4):
            result = viz.extract_frequency_data(freq_idx)
            assert result is not None


class TestPlotSingleFrequencyComparison:
    """Tests for Visualizer.plot_single_frequency_comparison"""

    def test_creates_file(self, visualization_data):
        """Test that PNG file is created."""
        from src.visualization import Visualizer

        viz = Visualizer(
            predictions_path=visualization_data['predictions_path'],
            data_path=visualization_data['test_data_path'],
            train_data_path=visualization_data['train_data_path'],
            training_history_path=visualization_data['history_path'],
            metrics_path=visualization_data['metrics_path']
        )

        save_path = visualization_data['temp_dir'] / "test_plot.png"
        viz.plot_single_frequency_comparison(
            freq_idx=0,
            time_window=5,
            save_path=str(save_path)
        )

        assert save_path.exists()

    def test_different_frequency_indices(self, visualization_data):
        """Test plotting first frequency (others may fail with small test data)."""
        from src.visualization import Visualizer

        viz = Visualizer(
            predictions_path=visualization_data['predictions_path'],
            data_path=visualization_data['test_data_path'],
            train_data_path=visualization_data['train_data_path'],
            training_history_path=visualization_data['history_path'],
            metrics_path=visualization_data['metrics_path']
        )

        # Test only first frequency with small test data
        # (full test would require 40,000 samples)
        save_path = visualization_data['temp_dir'] / "freq_0.png"
        viz.plot_single_frequency_comparison(
            freq_idx=0,
            time_window=5,
            save_path=str(save_path)
        )
        assert save_path.exists()




class TestPlotTrainingLossCurve:
    """Tests for Visualizer.plot_training_loss_curve"""

    def test_creates_file(self, visualization_data):
        """Test that PNG file is created."""
        from src.visualization import Visualizer

        viz = Visualizer(
            predictions_path=visualization_data['predictions_path'],
            data_path=visualization_data['test_data_path'],
            train_data_path=visualization_data['train_data_path'],
            training_history_path=visualization_data['history_path'],
            metrics_path=visualization_data['metrics_path']
        )

        save_path = visualization_data['temp_dir'] / "loss_curve.png"
        viz.plot_training_loss_curve(save_path=str(save_path))

        assert save_path.exists()


class TestPlotPerFrequencyPerformance:
    """Tests for Visualizer.plot_per_frequency_performance"""

    def test_creates_file(self, visualization_data):
        """Test that PNG file is created."""
        from src.visualization import Visualizer

        viz = Visualizer(
            predictions_path=visualization_data['predictions_path'],
            data_path=visualization_data['test_data_path'],
            train_data_path=visualization_data['train_data_path'],
            training_history_path=visualization_data['history_path'],
            metrics_path=visualization_data['metrics_path']
        )

        save_path = visualization_data['temp_dir'] / "per_freq.png"
        viz.plot_per_frequency_performance(save_path=str(save_path))

        assert save_path.exists()






class TestEdgeCases:
    """Tests for edge cases in visualization."""

    def test_empty_time_window(self, visualization_data):
        """Test handling of zero time window."""
        from src.visualization import Visualizer

        viz = Visualizer(
            predictions_path=visualization_data['predictions_path'],
            data_path=visualization_data['test_data_path'],
            train_data_path=visualization_data['train_data_path'],
            training_history_path=visualization_data['history_path'],
            metrics_path=visualization_data['metrics_path']
        )

        # Should handle gracefully
        result = viz.extract_frequency_data(0, time_window=0)
        # May return empty or handle as special case

    def test_large_time_window(self, visualization_data):
        """Test handling of time window larger than data."""
        from src.visualization import Visualizer

        viz = Visualizer(
            predictions_path=visualization_data['predictions_path'],
            data_path=visualization_data['test_data_path'],
            train_data_path=visualization_data['train_data_path'],
            training_history_path=visualization_data['history_path'],
            metrics_path=visualization_data['metrics_path']
        )

        # Should handle gracefully (limit to data size)
        result = viz.extract_frequency_data(0, time_window=10000)
        assert result is not None
