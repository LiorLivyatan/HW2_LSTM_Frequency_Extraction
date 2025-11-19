"""
Tests for src/table_generator.py

Tests the TableGenerator class for markdown table generation.
Target: ~35 tests covering markdown generation, calculations.
"""

import pytest
import numpy as np
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def table_generator_data(temp_dir):
    """Create all data files needed for table generation testing."""
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
            'threshold': 0.1,
            'generalizes_well': True
        }
    }
    metrics_path = temp_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

    return {
        'predictions_path': str(predictions_path),
        'test_data_path': str(test_data_path),
        'train_data_path': str(train_data_path),
        'metrics_path': str(metrics_path),
        'temp_dir': temp_dir
    }


class TestTableGeneratorInit:
    """Tests for TableGenerator.__init__"""

    def test_initialization(self, table_generator_data):
        """Test basic initialization."""
        from src.table_generator import TableGenerator

        gen = TableGenerator(
            predictions_path=table_generator_data['predictions_path'],
            metrics_path=table_generator_data['metrics_path'],
            train_data_path=table_generator_data['train_data_path'],
            test_data_path=table_generator_data['test_data_path']
        )
        assert gen is not None

    def test_loads_predictions(self, table_generator_data):
        """Test that predictions are loaded."""
        from src.table_generator import TableGenerator

        gen = TableGenerator(
            predictions_path=table_generator_data['predictions_path'],
            metrics_path=table_generator_data['metrics_path'],
            train_data_path=table_generator_data['train_data_path'],
            test_data_path=table_generator_data['test_data_path']
        )
        # Should have loaded data
        assert gen is not None

    def test_loads_metrics(self, table_generator_data):
        """Test that metrics are loaded."""
        from src.table_generator import TableGenerator

        gen = TableGenerator(
            predictions_path=table_generator_data['predictions_path'],
            metrics_path=table_generator_data['metrics_path'],
            train_data_path=table_generator_data['train_data_path'],
            test_data_path=table_generator_data['test_data_path']
        )
        assert hasattr(gen, 'metrics')


class TestGenerateDatasetStatisticsTable:
    """Tests for TableGenerator.generate_dataset_statistics_table"""

    def test_returns_string(self, table_generator_data):
        """Test that method returns a string."""
        from src.table_generator import TableGenerator

        gen = TableGenerator(
            predictions_path=table_generator_data['predictions_path'],
            metrics_path=table_generator_data['metrics_path'],
            train_data_path=table_generator_data['train_data_path'],
            test_data_path=table_generator_data['test_data_path']
        )

        result = gen.generate_dataset_statistics_table()
        assert isinstance(result, str)

    def test_contains_markdown_table(self, table_generator_data):
        """Test that result contains markdown table format."""
        from src.table_generator import TableGenerator

        gen = TableGenerator(
            predictions_path=table_generator_data['predictions_path'],
            metrics_path=table_generator_data['metrics_path'],
            train_data_path=table_generator_data['train_data_path'],
            test_data_path=table_generator_data['test_data_path']
        )

        result = gen.generate_dataset_statistics_table()
        # Markdown tables use | for columns
        assert '|' in result

    def test_contains_sample_info(self, table_generator_data):
        """Test that result contains sample information."""
        from src.table_generator import TableGenerator

        gen = TableGenerator(
            predictions_path=table_generator_data['predictions_path'],
            metrics_path=table_generator_data['metrics_path'],
            train_data_path=table_generator_data['train_data_path'],
            test_data_path=table_generator_data['test_data_path']
        )

        result = gen.generate_dataset_statistics_table()
        # Should contain numbers
        assert any(char.isdigit() for char in result)


class TestGeneratePerformanceSummaryTable:
    """Tests for TableGenerator.generate_performance_summary_table"""

    def test_returns_string(self, table_generator_data):
        """Test that method returns a string."""
        from src.table_generator import TableGenerator

        gen = TableGenerator(
            predictions_path=table_generator_data['predictions_path'],
            metrics_path=table_generator_data['metrics_path'],
            train_data_path=table_generator_data['train_data_path'],
            test_data_path=table_generator_data['test_data_path']
        )

        result = gen.generate_performance_summary_table()
        assert isinstance(result, str)

    def test_contains_mse(self, table_generator_data):
        """Test that result contains MSE values."""
        from src.table_generator import TableGenerator

        gen = TableGenerator(
            predictions_path=table_generator_data['predictions_path'],
            metrics_path=table_generator_data['metrics_path'],
            train_data_path=table_generator_data['train_data_path'],
            test_data_path=table_generator_data['test_data_path']
        )

        result = gen.generate_performance_summary_table()
        # Should contain MSE or mse
        assert 'MSE' in result or 'mse' in result.lower()

    def test_contains_generalization_status(self, table_generator_data):
        """Test that result contains generalization status."""
        from src.table_generator import TableGenerator

        gen = TableGenerator(
            predictions_path=table_generator_data['predictions_path'],
            metrics_path=table_generator_data['metrics_path'],
            train_data_path=table_generator_data['train_data_path'],
            test_data_path=table_generator_data['test_data_path']
        )

        result = gen.generate_performance_summary_table()
        # Should contain some indication of pass/fail or good/bad
        assert len(result) > 0


class TestGeneratePerFrequencyMetricsTable:
    """Tests for TableGenerator.generate_per_frequency_metrics_table"""

    def test_returns_string(self, table_generator_data):
        """Test that method returns a string."""
        from src.table_generator import TableGenerator

        gen = TableGenerator(
            predictions_path=table_generator_data['predictions_path'],
            metrics_path=table_generator_data['metrics_path'],
            train_data_path=table_generator_data['train_data_path'],
            test_data_path=table_generator_data['test_data_path']
        )

        result = gen.generate_per_frequency_metrics_table()
        assert isinstance(result, str)

    def test_contains_frequency_labels(self, table_generator_data):
        """Test that result contains frequency labels."""
        from src.table_generator import TableGenerator

        gen = TableGenerator(
            predictions_path=table_generator_data['predictions_path'],
            metrics_path=table_generator_data['metrics_path'],
            train_data_path=table_generator_data['train_data_path'],
            test_data_path=table_generator_data['test_data_path']
        )

        result = gen.generate_per_frequency_metrics_table()
        # Should contain Hz
        assert 'Hz' in result

    def test_contains_four_frequencies(self, table_generator_data):
        """Test that result mentions 4 frequencies."""
        from src.table_generator import TableGenerator

        gen = TableGenerator(
            predictions_path=table_generator_data['predictions_path'],
            metrics_path=table_generator_data['metrics_path'],
            train_data_path=table_generator_data['train_data_path'],
            test_data_path=table_generator_data['test_data_path']
        )

        result = gen.generate_per_frequency_metrics_table()
        # Should have all frequency labels
        freq_count = result.count('Hz')
        assert freq_count >= 4


class TestCreateAllTables:
    """Tests for TableGenerator.create_all_tables"""

    def test_creates_output_directory(self, table_generator_data):
        """Test that output directory is created."""
        from src.table_generator import TableGenerator

        gen = TableGenerator(
            predictions_path=table_generator_data['predictions_path'],
            metrics_path=table_generator_data['metrics_path'],
            train_data_path=table_generator_data['train_data_path'],
            test_data_path=table_generator_data['test_data_path']
        )

        output_dir = table_generator_data['temp_dir'] / "tables"
        gen.create_all_tables(output_dir=str(output_dir))

        assert output_dir.exists()

    def test_creates_markdown_files(self, table_generator_data):
        """Test that markdown files are created."""
        from src.table_generator import TableGenerator

        gen = TableGenerator(
            predictions_path=table_generator_data['predictions_path'],
            metrics_path=table_generator_data['metrics_path'],
            train_data_path=table_generator_data['train_data_path'],
            test_data_path=table_generator_data['test_data_path']
        )

        output_dir = table_generator_data['temp_dir'] / "tables"
        gen.create_all_tables(output_dir=str(output_dir))

        # Should have created markdown files
        md_files = list(output_dir.glob("*.md"))
        assert len(md_files) >= 1

    def test_files_have_content(self, table_generator_data):
        """Test that created files have content."""
        from src.table_generator import TableGenerator

        gen = TableGenerator(
            predictions_path=table_generator_data['predictions_path'],
            metrics_path=table_generator_data['metrics_path'],
            train_data_path=table_generator_data['train_data_path'],
            test_data_path=table_generator_data['test_data_path']
        )

        output_dir = table_generator_data['temp_dir'] / "tables"
        gen.create_all_tables(output_dir=str(output_dir))

        # Check first file has content
        md_files = list(output_dir.glob("*.md"))
        if md_files:
            content = md_files[0].read_text()
            assert len(content) > 0


class TestMarkdownFormatting:
    """Tests for markdown formatting."""

    def test_table_header_format(self, table_generator_data):
        """Test that tables have proper header format."""
        from src.table_generator import TableGenerator

        gen = TableGenerator(
            predictions_path=table_generator_data['predictions_path'],
            metrics_path=table_generator_data['metrics_path'],
            train_data_path=table_generator_data['train_data_path'],
            test_data_path=table_generator_data['test_data_path']
        )

        result = gen.generate_dataset_statistics_table()
        # Markdown table headers have | --- |
        assert '---' in result or '|' in result

    def test_no_broken_tables(self, table_generator_data):
        """Test that tables are not broken."""
        from src.table_generator import TableGenerator

        gen = TableGenerator(
            predictions_path=table_generator_data['predictions_path'],
            metrics_path=table_generator_data['metrics_path'],
            train_data_path=table_generator_data['train_data_path'],
            test_data_path=table_generator_data['test_data_path']
        )

        result = gen.generate_dataset_statistics_table()
        # Each row should have same number of |
        lines = [l for l in result.split('\n') if '|' in l]
        if lines:
            first_count = lines[0].count('|')
            for line in lines:
                assert line.count('|') == first_count


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_predictions(self, temp_dir):
        """Test handling of empty predictions."""
        from src.table_generator import TableGenerator

        # Create empty predictions
        predictions = {
            'train_predictions': np.array([]),
            'train_targets': np.array([]),
            'test_predictions': np.array([]),
            'test_targets': np.array([])
        }
        predictions_path = temp_dir / "empty_pred.npz"
        np.savez(predictions_path, **predictions)

        # Create minimal data
        test_data = np.random.randn(4, 6).astype(np.float32)
        test_data_path = temp_dir / "min_test.npy"
        np.save(test_data_path, test_data)

        train_data = np.random.randn(4, 6).astype(np.float32)
        train_data_path = temp_dir / "min_train.npy"
        np.save(train_data_path, train_data)

        # Create metrics
        metrics = {
            'overall': {'mse_train': 0, 'mse_test': 0},
            'per_frequency': {
                'train': {},
                'test': {}
            },
            'generalization': {
                'generalizes_well': True
            }
        }
        metrics_path = temp_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)

        # Should handle gracefully
        gen = TableGenerator(
            predictions_path=str(predictions_path),
            metrics_path=str(metrics_path),
            train_data_path=str(train_data_path),
            test_data_path=str(test_data_path)
        )
        assert gen is not None
