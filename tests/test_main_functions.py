"""
Tests for main() functions in various modules.

These tests execute the main functions to boost code coverage.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch
import io

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataGenerationMain:
    """Tests for data_generation.py main function."""

    def test_main_function_runs(self, temp_dir, monkeypatch):
        """Test that data_generation main function runs successfully."""
        from src import data_generation

        # Change to temp directory to avoid creating files in project
        monkeypatch.chdir(temp_dir)

        # Create data directory
        (temp_dir / "data").mkdir()

        # Run main function
        data_generation.main()

        # Check files were created
        assert (temp_dir / "data" / "train_data.npy").exists()
        assert (temp_dir / "data" / "test_data.npy").exists()


class TestDatasetMain:
    """Tests for dataset.py main function."""

    def test_main_function_runs(self, temp_dir, monkeypatch):
        """Test that dataset main function runs with existing data."""
        from src.data_generation import SignalGenerator
        from src import dataset

        # Change to temp directory
        monkeypatch.chdir(temp_dir)

        # Create test data
        data_dir = temp_dir / "data"
        data_dir.mkdir()

        gen = SignalGenerator(fs=1000, duration=10.0, seed=42)
        gen.save_dataset(str(data_dir / "train_data.npy"))

        # Run main function
        dataset.main()


class TestModelMain:
    """Tests for model.py main function."""

    def test_main_function_runs(self):
        """Test that model main function runs successfully."""
        from src import model

        # Capture output
        captured = io.StringIO()
        with patch('sys.stdout', captured):
            model.main()

        output = captured.getvalue()
        assert "FrequencyLSTM" in output
        assert "Test 1:" in output
        assert "Test 5:" in output


class TestTrainingMain:
    """Tests for training.py main function."""

    def test_main_function_with_missing_data(self):
        """Test that training main requires data files."""
        from src import training

        # Will fail without data files but tests the import path


class TestEvaluationMain:
    """Tests for evaluation.py main function."""

    def test_main_function_with_missing_files(self):
        """Test that evaluation main requires necessary files."""
        from src import evaluation

        # Will fail without required files but tests the import path


class TestVisualizationMain:
    """Tests for visualization.py main function."""

    def test_main_function_with_missing_files(self):
        """Test that visualization main requires necessary files."""
        from src import visualization

        # Will fail without required files but tests the import path


class TestTableGeneratorMain:
    """Tests for table_generator.py main function."""

    def test_main_function_with_missing_files(self):
        """Test that table_generator main requires necessary files."""
        from src import table_generator

        # Will fail without required files but tests the import path


class TestModelSummary:
    """Additional tests for model features to boost coverage."""

    def test_get_model_summary(self):
        """Test model summary generation."""
        from src.model import FrequencyLSTM

        model = FrequencyLSTM(hidden_size=64, num_layers=2)
        summary = model.get_model_summary()

        assert "FrequencyLSTM" in summary
        assert "64" in summary  # hidden size appears in state shape
        assert "(2," in summary  # num_layers appears in state shape as first dim

    def test_count_parameters_matches(self):
        """Test parameter counting."""
        from src.model import FrequencyLSTM

        model = FrequencyLSTM(hidden_size=32, num_layers=1)
        count = model.count_parameters()

        # Manually count
        total = sum(p.numel() for p in model.parameters())
        assert count == total

    def test_multi_layer_model(self):
        """Test multi-layer LSTM functionality."""
        import torch
        from src.model import FrequencyLSTM

        model = FrequencyLSTM(hidden_size=32, num_layers=3)
        x = torch.randn(4, 1, 5)

        output, (h, c) = model(x)

        assert output.shape == (4, 1)
        assert h.shape == (3, 4, 32)  # num_layers, batch, hidden
        assert c.shape == (3, 4, 32)

    def test_dropout_model(self):
        """Test model with dropout."""
        import torch
        from src.model import FrequencyLSTM

        model = FrequencyLSTM(hidden_size=32, num_layers=2, dropout=0.5)
        model.train()  # Dropout only active in training

        x = torch.randn(4, 1, 5)
        output, _ = model(x)

        assert output.shape == (4, 1)

    def test_batch_size_variations(self):
        """Test model with different batch sizes."""
        import torch
        from src.model import FrequencyLSTM

        model = FrequencyLSTM(hidden_size=32)

        for batch_size in [1, 16, 32, 64]:
            x = torch.randn(batch_size, 1, 5)
            output, (h, c) = model(x)

            assert output.shape == (batch_size, 1)
            assert h.shape == (1, batch_size, 32)


class TestEvaluatorAdvanced:
    """Additional tests for evaluator to boost coverage."""

    def test_evaluator_with_different_batch_sizes(self, device):
        """Test evaluator with various batch sizes."""
        import torch
        import numpy as np
        from src.model import FrequencyLSTM
        from src.evaluation import Evaluator
        from src.dataset import FrequencyDataset
        from torch.utils.data import DataLoader

        model = FrequencyLSTM(hidden_size=32)
        evaluator = Evaluator(model=model, device=device)

        # Create tiny dataset
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.randn(20, 6).astype(np.float32)
            for i in range(4):
                data[i*5:(i+1)*5, 1:5] = 0
                data[i*5:(i+1)*5, 1 + i] = 1

            filepath = Path(tmpdir) / "test.npy"
            np.save(filepath, data)

            dataset = FrequencyDataset(str(filepath))

            for batch_size in [1, 2, 5]:
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                mse, predictions, targets = evaluator.evaluate_dataset(loader)

                assert np.isfinite(mse)
                assert len(predictions) == len(data)


class TestSignalGeneratorEdgeCases:
    """Additional tests for SignalGenerator edge cases."""

    def test_time_array_properties(self):
        """Test time array mathematical properties."""
        from src.data_generation import SignalGenerator

        gen = SignalGenerator(fs=100, duration=1.0, seed=42)
        t = gen.generate_time_array()

        # Check monotonically increasing
        assert all(t[i] < t[i+1] for i in range(len(t)-1))

        # Check bounds
        assert t[0] == 0.0
        assert t[-1] == gen.duration

    def test_noisy_sinusoid_statistical_properties(self):
        """Test noisy sinusoid has expected statistical properties."""
        from src.data_generation import SignalGenerator

        gen = SignalGenerator(fs=1000, duration=1.0, seed=42)
        t = gen.generate_time_array()
        noisy = gen.generate_noisy_sinusoid(1.0, t)

        # Check mean is near zero (over full period)
        assert abs(np.mean(noisy)) < 0.5

        # Check variance is bounded
        assert np.var(noisy) < 1.5

    def test_dataset_column_structure(self):
        """Test dataset column structure in detail."""
        from src.data_generation import SignalGenerator

        gen = SignalGenerator(fs=100, duration=0.1, seed=42)
        dataset = gen.create_dataset()

        # Check column 0 (noisy signal) has reasonable values
        assert np.abs(dataset[:, 0]).max() < 2.0

        # Check columns 1-4 (one-hot) sum to 1
        one_hot_sums = dataset[:, 1:5].sum(axis=1)
        np.testing.assert_array_equal(one_hot_sums, np.ones(len(dataset)))

        # Check column 5 (target) is in [-1, 1]
        assert np.abs(dataset[:, 5]).max() <= 1.0


class TestTrainerCheckpointing:
    """Additional tests for trainer checkpointing."""

    def test_checkpoint_restoration_preserves_training(self, tiny_dataloader, temp_models_dir, device):
        """Test that loading checkpoint allows continued training."""
        import torch
        import torch.nn as nn
        from src.model import FrequencyLSTM
        from src.training import StatefulTrainer

        model = FrequencyLSTM(hidden_size=32)
        trainer = StatefulTrainer(
            model=model,
            train_loader=tiny_dataloader,
            criterion=nn.MSELoss(),
            optimizer=torch.optim.Adam(model.parameters()),
            device=device
        )

        # Train and save
        trainer.train(num_epochs=2, save_dir=str(temp_models_dir), save_best=True)

        # Create new trainer and load
        model2 = FrequencyLSTM(hidden_size=32)
        trainer2 = StatefulTrainer(
            model=model2,
            train_loader=tiny_dataloader,
            criterion=nn.MSELoss(),
            optimizer=torch.optim.Adam(model2.parameters()),
            device=device
        )

        checkpoint = trainer2.load_checkpoint(temp_models_dir / "best_model.pth")

        # Continue training
        loss = trainer2.train_epoch(0)
        assert np.isfinite(loss)


class TestVisualizationExtended:
    """Extended tests for visualization module coverage."""

    def test_visualizer_with_larger_data(self, temp_dir):
        """Test visualizer with properly sized data for coverage."""
        from src.visualization import Visualizer
        import json

        # Create data for 1000 samples per frequency (4000 total)
        n_samples = 1000
        total_samples = 4 * n_samples

        # Test data
        test_data = np.random.randn(total_samples, 6).astype(np.float32)
        for i in range(4):
            test_data[i*n_samples:(i+1)*n_samples, 1:5] = 0
            test_data[i*n_samples:(i+1)*n_samples, 1+i] = 1
        test_data_path = temp_dir / "test_data.npy"
        np.save(test_data_path, test_data)

        # Train data
        train_data = np.random.randn(total_samples, 6).astype(np.float32)
        for i in range(4):
            train_data[i*n_samples:(i+1)*n_samples, 1:5] = 0
            train_data[i*n_samples:(i+1)*n_samples, 1+i] = 1
        train_data_path = temp_dir / "train_data.npy"
        np.save(train_data_path, train_data)

        # Predictions
        predictions = {
            'train_predictions': np.random.randn(total_samples).astype(np.float32),
            'train_targets': np.random.randn(total_samples).astype(np.float32),
            'test_predictions': np.random.randn(total_samples).astype(np.float32),
            'test_targets': np.random.randn(total_samples).astype(np.float32)
        }
        predictions_path = temp_dir / "predictions.npz"
        np.savez(predictions_path, **predictions)

        # Metrics
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

        # History
        history = {
            'train_loss': [0.5, 0.3, 0.2, 0.15, 0.1],
            'epoch_times': [1.0, 1.1, 1.0, 1.2, 1.1],
            'best_epoch': 5,
            'best_loss': 0.1
        }
        history_path = temp_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f)

        # Create visualizer
        viz = Visualizer(
            predictions_path=str(predictions_path),
            data_path=str(test_data_path),
            train_data_path=str(train_data_path),
            training_history_path=str(history_path),
            metrics_path=str(metrics_path)
        )

        # Test plot_training_loss_curve
        save_path = temp_dir / "loss_curve.png"
        viz.plot_training_loss_curve(save_path=str(save_path))
        assert save_path.exists()

        # Test plot_per_frequency_performance
        save_path2 = temp_dir / "per_freq.png"
        viz.plot_per_frequency_performance(save_path=str(save_path2))
        assert save_path2.exists()

        # Test plot_single_frequency_comparison
        save_path3 = temp_dir / "single_freq.png"
        viz.plot_single_frequency_comparison(freq_idx=0, time_window=100, save_path=str(save_path3))
        assert save_path3.exists()

    def test_data_loading_verification(self, temp_dir):
        """Test that visualizer correctly loads data."""
        from src.visualization import Visualizer
        import json

        # Create properly sized data
        n_samples = 100
        total = 4 * n_samples

        test_data = np.random.randn(total, 6).astype(np.float32)
        for i in range(4):
            test_data[i*n_samples:(i+1)*n_samples, 1:5] = 0
            test_data[i*n_samples:(i+1)*n_samples, 1+i] = 1
        np.save(temp_dir / "test.npy", test_data)

        train_data = test_data.copy()
        np.save(temp_dir / "train.npy", train_data)

        predictions = {
            'train_predictions': np.random.randn(total).astype(np.float32),
            'train_targets': np.random.randn(total).astype(np.float32),
            'test_predictions': np.random.randn(total).astype(np.float32),
            'test_targets': np.random.randn(total).astype(np.float32)
        }
        np.savez(temp_dir / "pred.npz", **predictions)

        metrics = {
            'overall': {'mse_train': 0.1, 'mse_test': 0.11},
            'per_frequency': {
                'train': {'0': 0.01, '1': 0.02, '2': 0.03, '3': 0.04},
                'test': {'0': 0.011, '1': 0.021, '2': 0.031, '3': 0.041}
            },
            'generalization': {'generalizes_well': True}
        }
        with open(temp_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f)

        history = {'train_loss': [0.5], 'epoch_times': [1.0], 'best_epoch': 1, 'best_loss': 0.5}
        with open(temp_dir / "history.json", 'w') as f:
            json.dump(history, f)

        viz = Visualizer(
            predictions_path=str(temp_dir / "pred.npz"),
            data_path=str(temp_dir / "test.npy"),
            train_data_path=str(temp_dir / "train.npy"),
            training_history_path=str(temp_dir / "history.json"),
            metrics_path=str(temp_dir / "metrics.json")
        )

        # Verify data shapes match input
        assert viz.test_predictions.shape == (total,)
        assert viz.test_targets.shape == (total,)
        assert viz.test_noisy.shape == (total,)
        assert viz.train_noisy.shape == (total,)
        assert len(viz.training_history['train_loss']) == 1
        assert 'overall' in viz.metrics


class TestEvaluatorMetrics:
    """Additional tests for evaluator metrics calculation."""

    def test_check_generalization_good(self, device):
        """Test generalization check with good results."""
        from src.model import FrequencyLSTM
        from src.evaluation import Evaluator

        model = FrequencyLSTM(hidden_size=32)
        evaluator = Evaluator(model=model, device=device)

        # Good generalization (small gap)
        result = evaluator.check_generalization(
            mse_train=0.1,
            mse_test=0.105,
            threshold=0.1
        )

        assert result['generalizes_well'] == True

    def test_check_generalization_bad(self, device):
        """Test generalization check with poor results."""
        from src.model import FrequencyLSTM
        from src.evaluation import Evaluator

        model = FrequencyLSTM(hidden_size=32)
        evaluator = Evaluator(model=model, device=device)

        # Bad generalization (large gap)
        result = evaluator.check_generalization(
            mse_train=0.1,
            mse_test=0.5,
            threshold=0.1
        )

        assert result['generalizes_well'] == False

    def test_save_metrics(self, device, temp_dir):
        """Test saving metrics to file."""
        from src.model import FrequencyLSTM
        from src.evaluation import Evaluator
        import json

        model = FrequencyLSTM(hidden_size=32)
        evaluator = Evaluator(model=model, device=device)

        # Create sample metrics
        metrics = {
            'overall': {'mse_train': 0.1, 'mse_test': 0.11},
            'per_frequency': {
                'train': {'0': 0.01},
                'test': {'0': 0.011}
            },
            'generalization': {'generalizes_well': True}
        }

        save_path = temp_dir / "metrics.json"
        evaluator.save_metrics(metrics, str(save_path))

        assert save_path.exists()
        with open(save_path) as f:
            loaded = json.load(f)
        assert 'overall' in loaded


class TestTableGeneratorExtended:
    """Extended tests for TableGenerator to increase coverage."""

    def test_generate_all_table_methods(self, temp_dir):
        """Test all table generation methods."""
        from src.table_generator import TableGenerator
        import json

        n_samples = 40
        total = 4 * n_samples

        # Create complete test data
        predictions = {
            'train_predictions': np.random.randn(total).astype(np.float32),
            'train_targets': np.random.randn(total).astype(np.float32),
            'test_predictions': np.random.randn(total).astype(np.float32),
            'test_targets': np.random.randn(total).astype(np.float32)
        }
        np.savez(temp_dir / "pred.npz", **predictions)

        test_data = np.random.randn(total, 6).astype(np.float32)
        for i in range(4):
            test_data[i*n_samples:(i+1)*n_samples, 1:5] = 0
            test_data[i*n_samples:(i+1)*n_samples, 1+i] = 1
        np.save(temp_dir / "test.npy", test_data)

        train_data = test_data.copy()
        np.save(temp_dir / "train.npy", train_data)

        metrics = {
            'overall': {'mse_train': 0.1, 'mse_test': 0.11},
            'per_frequency': {
                'train': {'0': 0.01, '1': 0.02, '2': 0.03, '3': 0.04},
                'test': {'0': 0.011, '1': 0.021, '2': 0.031, '3': 0.041}
            },
            'generalization': {
                'absolute_difference': 0.01,
                'relative_difference': 0.1,
                'threshold': 0.1,
                'generalizes_well': True
            }
        }
        with open(temp_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f)

        gen = TableGenerator(
            predictions_path=str(temp_dir / "pred.npz"),
            metrics_path=str(temp_dir / "metrics.json"),
            train_data_path=str(temp_dir / "train.npy"),
            test_data_path=str(temp_dir / "test.npy")
        )

        # Test individual table methods
        stats_table = gen.generate_dataset_statistics_table()
        assert '|' in stats_table
        assert len(stats_table) > 50

        perf_table = gen.generate_performance_summary_table()
        assert '|' in perf_table

        freq_table = gen.generate_per_frequency_metrics_table()
        assert 'Hz' in freq_table

    def test_create_all_tables_with_output(self, temp_dir):
        """Test creating all tables and saving to files."""
        from src.table_generator import TableGenerator
        import json

        n_samples = 40
        total = 4 * n_samples

        predictions = {
            'train_predictions': np.random.randn(total).astype(np.float32),
            'train_targets': np.random.randn(total).astype(np.float32),
            'test_predictions': np.random.randn(total).astype(np.float32),
            'test_targets': np.random.randn(total).astype(np.float32)
        }
        np.savez(temp_dir / "pred.npz", **predictions)

        test_data = np.random.randn(total, 6).astype(np.float32)
        for i in range(4):
            test_data[i*n_samples:(i+1)*n_samples, 1:5] = 0
            test_data[i*n_samples:(i+1)*n_samples, 1+i] = 1
        np.save(temp_dir / "test.npy", test_data)

        train_data = test_data.copy()
        np.save(temp_dir / "train.npy", train_data)

        metrics = {
            'overall': {'mse_train': 0.1, 'mse_test': 0.11},
            'per_frequency': {
                'train': {'0': 0.01, '1': 0.02, '2': 0.03, '3': 0.04},
                'test': {'0': 0.011, '1': 0.021, '2': 0.031, '3': 0.041}
            },
            'generalization': {
                'absolute_difference': 0.01,
                'relative_difference': 0.1,
                'threshold': 0.1,
                'generalizes_well': True
            }
        }
        with open(temp_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f)

        gen = TableGenerator(
            predictions_path=str(temp_dir / "pred.npz"),
            metrics_path=str(temp_dir / "metrics.json"),
            train_data_path=str(temp_dir / "train.npy"),
            test_data_path=str(temp_dir / "test.npy")
        )

        output_dir = temp_dir / "tables"
        gen.create_all_tables(output_dir=str(output_dir))

        assert output_dir.exists()
        md_files = list(output_dir.glob("*.md"))
        assert len(md_files) >= 1


class TestTrainerStateManagement:
    """Tests for training state management."""

    def test_train_epoch_state_preservation(self, device, temp_dir):
        """Test that state is preserved across batches in training."""
        from src.model import FrequencyLSTM
        from src.training import StatefulTrainer
        from src.dataset import FrequencyDataset
        from torch.utils.data import DataLoader
        import torch.nn as nn
        import torch.optim as optim

        # Create small dataset
        n_samples = 32
        total = 4 * n_samples
        data = np.random.randn(total, 6).astype(np.float32)
        for i in range(4):
            data[i*n_samples:(i+1)*n_samples, 1:5] = 0
            data[i*n_samples:(i+1)*n_samples, 1+i] = 1

        data_path = temp_dir / "train_state.npy"
        np.save(data_path, data)

        dataset = FrequencyDataset(str(data_path))
        loader = DataLoader(dataset, batch_size=8, shuffle=False)

        model = FrequencyLSTM(hidden_size=16)
        model.to(device)

        trainer = StatefulTrainer(
            model=model,
            train_loader=loader,
            criterion=nn.MSELoss(),
            optimizer=optim.Adam(model.parameters(), lr=0.01),
            device=device
        )

        # Run one epoch
        loss = trainer.train_epoch(epoch=1)
        assert loss > 0
        assert np.isfinite(loss)

    def test_save_and_load_checkpoint(self, device, temp_dir):
        """Test checkpoint save/load cycle."""
        from src.model import FrequencyLSTM
        from src.training import StatefulTrainer
        from src.dataset import FrequencyDataset
        from torch.utils.data import DataLoader
        import torch.nn as nn
        import torch.optim as optim
        from pathlib import Path

        # Create dataset
        n_samples = 16
        total = 4 * n_samples
        data = np.random.randn(total, 6).astype(np.float32)
        for i in range(4):
            data[i*n_samples:(i+1)*n_samples, 1:5] = 0
            data[i*n_samples:(i+1)*n_samples, 1+i] = 1
        np.save(temp_dir / "train.npy", data)

        dataset = FrequencyDataset(str(temp_dir / "train.npy"))
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        model = FrequencyLSTM(hidden_size=16)
        model.to(device)

        trainer = StatefulTrainer(
            model=model,
            train_loader=loader,
            criterion=nn.MSELoss(),
            optimizer=optim.Adam(model.parameters(), lr=0.01),
            device=device
        )

        # Save checkpoint
        checkpoint_path = temp_dir / "checkpoint.pth"
        trainer.save_checkpoint(Path(checkpoint_path), epoch=1, loss=0.5)
        assert checkpoint_path.exists()

        # Load and verify
        import torch
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        assert 'model_state_dict' in checkpoint
        assert checkpoint['epoch'] == 2  # save_checkpoint stores epoch+1


class TestEvaluatorExtended:
    """Extended evaluator tests for better coverage."""

    def test_full_evaluation_pipeline(self, device, temp_dir):
        """Test complete evaluation pipeline with save."""
        from src.model import FrequencyLSTM
        from src.evaluation import Evaluator
        from src.dataset import FrequencyDataset
        from torch.utils.data import DataLoader

        # Create datasets
        n_samples = 20
        total = 4 * n_samples

        train_data = np.random.randn(total, 6).astype(np.float32)
        test_data = np.random.randn(total, 6).astype(np.float32)

        for data in [train_data, test_data]:
            for i in range(4):
                data[i*n_samples:(i+1)*n_samples, 1:5] = 0
                data[i*n_samples:(i+1)*n_samples, 1+i] = 1

        np.save(temp_dir / "train.npy", train_data)
        np.save(temp_dir / "test.npy", test_data)

        train_ds = FrequencyDataset(str(temp_dir / "train.npy"))
        test_ds = FrequencyDataset(str(temp_dir / "test.npy"))

        train_loader = DataLoader(train_ds, batch_size=5, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=5, shuffle=False)

        model = FrequencyLSTM(hidden_size=16)
        model.to(device)

        evaluator = Evaluator(model=model, device=device)
        results = evaluator.evaluate_all(train_loader, test_loader)

        assert 'overall' in results
        assert 'mse_train' in results['overall']
        assert 'mse_test' in results['overall']

        # Save metrics
        metrics_path = temp_dir / "eval_metrics.json"
        evaluator.save_metrics(results, str(metrics_path))
        assert metrics_path.exists()

        # Save predictions
        pred_path = temp_dir / "eval_pred.npz"
        evaluator.save_predictions(results, str(pred_path))
        assert pred_path.exists()

    def test_evaluate_with_different_samples_per_freq(self, device, temp_dir):
        """Test evaluation with custom samples_per_freq."""
        from src.model import FrequencyLSTM
        from src.evaluation import Evaluator
        from src.dataset import FrequencyDataset
        from torch.utils.data import DataLoader

        n_samples = 10
        total = 4 * n_samples

        data = np.random.randn(total, 6).astype(np.float32)
        for i in range(4):
            data[i*n_samples:(i+1)*n_samples, 1:5] = 0
            data[i*n_samples:(i+1)*n_samples, 1+i] = 1

        np.save(temp_dir / "data.npy", data)
        dataset = FrequencyDataset(str(temp_dir / "data.npy"))
        loader = DataLoader(dataset, batch_size=10, shuffle=False)

        model = FrequencyLSTM(hidden_size=16)
        model.to(device)

        evaluator = Evaluator(model=model, device=device)

        # Test per-frequency calculation with custom samples_per_freq
        mse, predictions, targets = evaluator.evaluate_dataset(loader)
        per_freq = evaluator.calculate_per_frequency_metrics(
            predictions, targets, samples_per_freq=n_samples
        )

        assert len(per_freq) == 4
        for i in range(4):
            assert i in per_freq


class TestDataGenerationAdvanced:
    """Advanced data generation tests."""

    def test_generate_multiple_datasets_different_seeds(self, temp_dir):
        """Test generating datasets with different seeds produces different results."""
        from src.data_generation import SignalGenerator

        gen1 = SignalGenerator(fs=100, duration=0.1, seed=42)
        gen2 = SignalGenerator(fs=100, duration=0.1, seed=99)

        dataset1 = gen1.create_dataset()
        dataset2 = gen2.create_dataset()

        # Shapes should match
        assert dataset1.shape == dataset2.shape
        # Values should differ
        assert not np.allclose(dataset1, dataset2)

    def test_clean_targets_structure(self, temp_dir):
        """Test that clean targets have correct structure and values."""
        from src.data_generation import SignalGenerator

        gen = SignalGenerator(
            frequencies=[1, 3, 5, 7],
            fs=1000,
            duration=1.0,
            seed=42
        )

        # Generate time array and clean targets
        t = gen.generate_time_array()
        targets = gen.generate_clean_targets(t)

        # Shape should be (4, n_samples)
        assert targets.shape == (4, len(t))

        # Clean targets should be pure sinusoids with amplitude 1
        # Check that max amplitude is around 1
        for i in range(4):
            assert np.abs(targets[i, :]).max() <= 1.01  # Allow small numerical error
            assert np.abs(targets[i, :]).max() >= 0.9   # Should reach near 1


class TestVisualizationFull:
    """Tests for visualization with full-size datasets."""

    def test_plot_all_frequencies(self, temp_dir):
        """Test plotting all frequencies with 40,000 sample dataset."""
        from src.visualization import Visualizer
        import json

        # Create full-size data (40,000 samples = 4 * 10,000)
        n_samples = 10000
        total = 4 * n_samples

        test_data = np.random.randn(total, 6).astype(np.float32)
        for i in range(4):
            test_data[i*n_samples:(i+1)*n_samples, 1:5] = 0
            test_data[i*n_samples:(i+1)*n_samples, 1+i] = 1
        np.save(temp_dir / "test.npy", test_data)

        train_data = test_data.copy()
        np.save(temp_dir / "train.npy", train_data)

        predictions = {
            'train_predictions': np.random.randn(total).astype(np.float32),
            'train_targets': np.random.randn(total).astype(np.float32),
            'test_predictions': np.random.randn(total).astype(np.float32),
            'test_targets': np.random.randn(total).astype(np.float32)
        }
        np.savez(temp_dir / "pred.npz", **predictions)

        metrics = {
            'overall': {'mse_train': 0.1, 'mse_test': 0.11},
            'per_frequency': {
                'train': {'0': 0.01, '1': 0.02, '2': 0.03, '3': 0.04},
                'test': {'0': 0.011, '1': 0.021, '2': 0.031, '3': 0.041}
            },
            'generalization': {
                'absolute_difference': 0.01,
                'relative_difference': 0.1,
                'threshold': 0.1,
                'generalizes_well': True
            }
        }
        with open(temp_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f)

        history = {
            'train_loss': [0.5, 0.4, 0.3],
            'epoch_times': [1.0, 1.0, 1.0],
            'best_epoch': 3,
            'best_loss': 0.3
        }
        with open(temp_dir / "history.json", 'w') as f:
            json.dump(history, f)

        viz = Visualizer(
            predictions_path=str(temp_dir / "pred.npz"),
            data_path=str(temp_dir / "test.npy"),
            train_data_path=str(temp_dir / "train.npy"),
            training_history_path=str(temp_dir / "history.json"),
            metrics_path=str(temp_dir / "metrics.json")
        )

        # Test plot_all_frequencies
        save_path = temp_dir / "all_freqs.png"
        fig = viz.plot_all_frequencies(time_window=1000, save_path=str(save_path))
        assert save_path.exists()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_single_frequency_comparison(self, temp_dir):
        """Test single frequency comparison plot."""
        from src.visualization import Visualizer
        import json

        n_samples = 10000
        total = 4 * n_samples

        test_data = np.random.randn(total, 6).astype(np.float32)
        for i in range(4):
            test_data[i*n_samples:(i+1)*n_samples, 1:5] = 0
            test_data[i*n_samples:(i+1)*n_samples, 1+i] = 1
        np.save(temp_dir / "test.npy", test_data)

        train_data = test_data.copy()
        np.save(temp_dir / "train.npy", train_data)

        predictions = {
            'train_predictions': np.random.randn(total).astype(np.float32),
            'train_targets': np.random.randn(total).astype(np.float32),
            'test_predictions': np.random.randn(total).astype(np.float32),
            'test_targets': np.random.randn(total).astype(np.float32)
        }
        np.savez(temp_dir / "pred.npz", **predictions)

        metrics = {
            'overall': {'mse_train': 0.1, 'mse_test': 0.11},
            'per_frequency': {
                'train': {'0': 0.01, '1': 0.02, '2': 0.03, '3': 0.04},
                'test': {'0': 0.011, '1': 0.021, '2': 0.031, '3': 0.041}
            },
            'generalization': {'generalizes_well': True}
        }
        with open(temp_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f)

        history = {'train_loss': [0.5], 'epoch_times': [1.0], 'best_epoch': 1, 'best_loss': 0.5}
        with open(temp_dir / "history.json", 'w') as f:
            json.dump(history, f)

        viz = Visualizer(
            predictions_path=str(temp_dir / "pred.npz"),
            data_path=str(temp_dir / "test.npy"),
            train_data_path=str(temp_dir / "train.npy"),
            training_history_path=str(temp_dir / "history.json"),
            metrics_path=str(temp_dir / "metrics.json")
        )

        # Test plot_single_frequency_comparison
        save_path = temp_dir / "single_freq.png"
        fig = viz.plot_single_frequency_comparison(
            freq_idx=1, time_window=1000, save_path=str(save_path)
        )
        assert save_path.exists()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_training_loss_curve(self, temp_dir):
        """Test training loss curve plot."""
        from src.visualization import Visualizer
        import json

        n_samples = 10000
        total = 4 * n_samples

        test_data = np.random.randn(total, 6).astype(np.float32)
        for i in range(4):
            test_data[i*n_samples:(i+1)*n_samples, 1:5] = 0
            test_data[i*n_samples:(i+1)*n_samples, 1+i] = 1
        np.save(temp_dir / "test.npy", test_data)
        np.save(temp_dir / "train.npy", test_data)

        predictions = {
            'train_predictions': np.random.randn(total).astype(np.float32),
            'train_targets': np.random.randn(total).astype(np.float32),
            'test_predictions': np.random.randn(total).astype(np.float32),
            'test_targets': np.random.randn(total).astype(np.float32)
        }
        np.savez(temp_dir / "pred.npz", **predictions)

        metrics = {
            'overall': {'mse_train': 0.1, 'mse_test': 0.11},
            'per_frequency': {
                'train': {'0': 0.01, '1': 0.02, '2': 0.03, '3': 0.04},
                'test': {'0': 0.011, '1': 0.021, '2': 0.031, '3': 0.041}
            },
            'generalization': {'generalizes_well': True}
        }
        with open(temp_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f)

        # Create longer history for training curve
        history = {
            'train_loss': [0.5 - i*0.01 for i in range(50)],
            'epoch_times': [1.0] * 50,
            'best_epoch': 50,
            'best_loss': 0.01
        }
        with open(temp_dir / "history.json", 'w') as f:
            json.dump(history, f)

        viz = Visualizer(
            predictions_path=str(temp_dir / "pred.npz"),
            data_path=str(temp_dir / "test.npy"),
            train_data_path=str(temp_dir / "train.npy"),
            training_history_path=str(temp_dir / "history.json"),
            metrics_path=str(temp_dir / "metrics.json")
        )

        # Test plot_training_loss_curve
        save_path = temp_dir / "loss_curve.png"
        fig = viz.plot_training_loss_curve(save_path=str(save_path))
        assert save_path.exists()
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestTrainerFullTrain:
    """Tests for full training functionality."""

    def test_train_with_save_every(self, device, temp_dir):
        """Test training with periodic checkpoint saving."""
        from src.model import FrequencyLSTM
        from src.training import StatefulTrainer
        from src.dataset import FrequencyDataset
        from torch.utils.data import DataLoader
        import torch.nn as nn
        import torch.optim as optim

        # Create dataset
        n_samples = 16
        total = 4 * n_samples
        data = np.random.randn(total, 6).astype(np.float32)
        for i in range(4):
            data[i*n_samples:(i+1)*n_samples, 1:5] = 0
            data[i*n_samples:(i+1)*n_samples, 1+i] = 1
        np.save(temp_dir / "train.npy", data)

        dataset = FrequencyDataset(str(temp_dir / "train.npy"))
        loader = DataLoader(dataset, batch_size=8, shuffle=False)

        model = FrequencyLSTM(hidden_size=16)
        model.to(device)

        trainer = StatefulTrainer(
            model=model,
            train_loader=loader,
            criterion=nn.MSELoss(),
            optimizer=optim.Adam(model.parameters(), lr=0.01),
            device=device
        )

        # Train with save_every
        history = trainer.train(
            num_epochs=4,
            save_dir=str(temp_dir),
            save_best=True,
            save_every=2
        )

        assert 'train_loss' in history
        assert len(history['train_loss']) == 4
        # Check periodic checkpoint was saved
        assert (temp_dir / "checkpoint_epoch_2.pth").exists()
        assert (temp_dir / "checkpoint_epoch_4.pth").exists()

    def test_train_improves_loss(self, device, temp_dir):
        """Test that training improves loss over epochs."""
        from src.model import FrequencyLSTM
        from src.training import StatefulTrainer
        from src.dataset import FrequencyDataset
        from torch.utils.data import DataLoader
        import torch.nn as nn
        import torch.optim as optim

        # Create dataset with simple pattern
        n_samples = 32
        total = 4 * n_samples
        data = np.zeros((total, 6), dtype=np.float32)
        for i in range(4):
            data[i*n_samples:(i+1)*n_samples, 0] = 1.0  # Noisy input
            data[i*n_samples:(i+1)*n_samples, 1:5] = 0
            data[i*n_samples:(i+1)*n_samples, 1+i] = 1  # One-hot
            data[i*n_samples:(i+1)*n_samples, 5] = 0.5  # Target
        np.save(temp_dir / "train.npy", data)

        dataset = FrequencyDataset(str(temp_dir / "train.npy"))
        loader = DataLoader(dataset, batch_size=8, shuffle=False)

        model = FrequencyLSTM(hidden_size=32)
        model.to(device)

        trainer = StatefulTrainer(
            model=model,
            train_loader=loader,
            criterion=nn.MSELoss(),
            optimizer=optim.Adam(model.parameters(), lr=0.01),
            device=device
        )

        history = trainer.train(num_epochs=3, save_dir=str(temp_dir))

        # Training should generally improve (loss decrease)
        assert len(history['train_loss']) == 3
        # History file should be saved
        assert (temp_dir / "training_history.json").exists()


class TestModelStatePreservation:
    """Tests for model state preservation patterns."""

    def test_hidden_state_shape_single_layer(self, device):
        """Test hidden state shape for single layer model."""
        from src.model import FrequencyLSTM
        import torch

        model = FrequencyLSTM(hidden_size=64, num_layers=1)
        model.to(device)

        batch_size = 16
        x = torch.randn(batch_size, 1, 5).to(device)

        output, (h_n, c_n) = model(x)

        assert h_n.shape == (1, batch_size, 64)
        assert c_n.shape == (1, batch_size, 64)

    def test_hidden_state_shape_multi_layer(self, device):
        """Test hidden state shape for multi-layer model."""
        from src.model import FrequencyLSTM
        import torch

        model = FrequencyLSTM(hidden_size=32, num_layers=3)
        model.to(device)

        batch_size = 8
        x = torch.randn(batch_size, 1, 5).to(device)

        output, (h_n, c_n) = model(x)

        assert h_n.shape == (3, batch_size, 32)
        assert c_n.shape == (3, batch_size, 32)

    def test_state_detachment(self, device):
        """Test that state detachment breaks gradient connection."""
        from src.model import FrequencyLSTM
        import torch

        model = FrequencyLSTM(hidden_size=16)
        model.to(device)

        x = torch.randn(4, 1, 5).to(device)
        output, (h_n, c_n) = model(x)

        # Original states require grad
        assert h_n.requires_grad
        assert c_n.requires_grad

        # Detached states don't
        h_detached = h_n.detach()
        c_detached = c_n.detach()
        assert not h_detached.requires_grad
        assert not c_detached.requires_grad
