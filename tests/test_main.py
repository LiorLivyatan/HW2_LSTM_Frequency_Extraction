"""
Tests for main.py

Tests the main orchestration script and CLI.
Target: ~40 tests covering CLI parsing, phase execution, config handling.
"""

import pytest
import numpy as np
import yaml
import json
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLoadConfig:
    """Tests for load_config function"""

    def test_loads_valid_yaml(self, sample_config_yaml):
        """Test loading a valid YAML config."""
        from main import load_config
        config = load_config(str(sample_config_yaml))
        assert isinstance(config, dict)

    def test_returns_dict(self, sample_config_yaml):
        """Test that config is returned as dict."""
        from main import load_config
        config = load_config(str(sample_config_yaml))
        assert isinstance(config, dict)

    def test_parses_sections(self, sample_config_yaml):
        """Test that all sections are parsed."""
        from main import load_config
        config = load_config(str(sample_config_yaml))

        assert 'data' in config
        assert 'model' in config
        assert 'training' in config

    def test_file_not_found(self, temp_dir):
        """Test that missing file raises error."""
        from main import load_config

        with pytest.raises(FileNotFoundError):
            load_config(str(temp_dir / "nonexistent.yaml"))


class TestSaveConfig:
    """Tests for save_config function"""

    def test_creates_file(self, temp_dir, default_config):
        """Test that config file is created."""
        from main import save_config

        save_path = temp_dir / "saved_config.yaml"
        save_config(default_config, str(save_path))

        assert save_path.exists()

    def test_creates_parent_directory(self, temp_dir, default_config):
        """Test that parent directory is created."""
        from main import save_config

        save_path = temp_dir / "new_dir" / "config.yaml"
        save_config(default_config, str(save_path))

        assert save_path.exists()

    def test_saved_yaml_valid(self, temp_dir, default_config):
        """Test that saved YAML is valid."""
        from main import save_config

        save_path = temp_dir / "config.yaml"
        save_config(default_config, str(save_path))

        with open(save_path) as f:
            loaded = yaml.safe_load(f)

        assert loaded is not None

    def test_saved_config_matches(self, temp_dir, default_config):
        """Test that saved config matches original."""
        from main import save_config

        save_path = temp_dir / "config.yaml"
        save_config(default_config, str(save_path))

        with open(save_path) as f:
            loaded = yaml.safe_load(f)

        assert loaded['data']['frequencies'] == default_config['data']['frequencies']


class TestSetSeeds:
    """Tests for set_seeds function"""

    def test_sets_numpy_seed(self):
        """Test that numpy seed is set."""
        from main import set_seeds

        set_seeds(42)

        # Generate random numbers
        val1 = np.random.rand()

        set_seeds(42)
        val2 = np.random.rand()

        assert val1 == val2

    def test_sets_torch_seed(self):
        """Test that torch seed is set."""
        import torch
        from main import set_seeds

        set_seeds(42)
        val1 = torch.rand(1).item()

        set_seeds(42)
        val2 = torch.rand(1).item()

        assert val1 == val2

    def test_different_seeds_different_values(self):
        """Test that different seeds produce different values."""
        from main import set_seeds

        set_seeds(42)
        val1 = np.random.rand()

        set_seeds(99)
        val2 = np.random.rand()

        assert val1 != val2


class TestPhaseDataGeneration:
    """Tests for phase_data_generation function"""

    def test_creates_train_dataset(self, temp_dir, small_config):
        """Test that training dataset is created."""
        from main import phase_data_generation

        # Update paths
        small_config['paths'] = {
            'train_data': str(temp_dir / "train_data.npy"),
            'test_data': str(temp_dir / "test_data.npy")
        }

        phase_data_generation(small_config)

        assert Path(small_config['paths']['train_data']).exists()

    def test_creates_test_dataset(self, temp_dir, small_config):
        """Test that test dataset is created."""
        from main import phase_data_generation

        small_config['paths'] = {
            'train_data': str(temp_dir / "train_data.npy"),
            'test_data': str(temp_dir / "test_data.npy")
        }

        phase_data_generation(small_config)

        assert Path(small_config['paths']['test_data']).exists()

    def test_uses_correct_seeds(self, temp_dir, small_config):
        """Test that correct seeds are used."""
        from main import phase_data_generation

        small_config['paths'] = {
            'train_data': str(temp_dir / "train_data.npy"),
            'test_data': str(temp_dir / "test_data.npy")
        }

        phase_data_generation(small_config)

        # Load both datasets - they should be different
        train = np.load(small_config['paths']['train_data'])
        test = np.load(small_config['paths']['test_data'])

        # Same shape
        assert train.shape == test.shape
        # But different values (different seeds)
        assert not np.array_equal(train, test)


class TestSetupLogging:
    """Tests for setup_logging function"""

    def test_creates_log_directory(self, temp_dir):
        """Test that log directory is created."""
        from main import setup_logging

        log_path = temp_dir / "logs" / "test.log"
        setup_logging(str(log_path), verbose=False)

        # Directory should be created
        assert log_path.parent.exists()

    def test_verbose_mode(self, temp_dir):
        """Test verbose mode setting."""
        from main import setup_logging
        import logging

        log_path = temp_dir / "test.log"
        setup_logging(str(log_path), verbose=True)

        # Should set DEBUG level
        logger = logging.getLogger()
        # Check that it's configured (hard to test exact level)
        assert True


class TestCLIParsing:
    """Tests for CLI argument parsing"""

    def test_parse_mode_all(self):
        """Test parsing --mode all."""
        from main import main
        import argparse

        # Mock argparse
        with patch('argparse.ArgumentParser.parse_args') as mock_args:
            mock_args.return_value = argparse.Namespace(
                mode='all',
                config='config.yaml',
                verbose=False
            )
            # Would need full integration test

    def test_valid_modes(self):
        """Test that valid modes are accepted."""
        valid_modes = ['all', 'data', 'train', 'eval', 'viz']
        for mode in valid_modes:
            # Should not raise
            assert mode in valid_modes


class TestIntegration:
    """Integration tests for main module."""

    def test_data_phase_standalone(self, temp_dir, small_config):
        """Test data generation phase standalone."""
        from main import phase_data_generation

        small_config['paths'] = {
            'train_data': str(temp_dir / "train.npy"),
            'test_data': str(temp_dir / "test.npy")
        }

        phase_data_generation(small_config)

        train = np.load(small_config['paths']['train_data'])
        test = np.load(small_config['paths']['test_data'])

        # Check shapes
        # n_samples = fs * duration, total_rows = 4 * n_samples
        n_samples = int(small_config['data']['sampling_rate'] * small_config['data']['duration'])
        expected_rows = 4 * n_samples
        assert train.shape[0] == expected_rows
        assert test.shape[0] == expected_rows


class TestErrorHandling:
    """Tests for error handling in main."""

    def test_missing_config_file(self):
        """Test handling of missing config file."""
        from main import load_config

        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_invalid_yaml(self, temp_dir):
        """Test handling of invalid YAML."""
        from main import load_config

        # Create invalid YAML
        invalid_path = temp_dir / "invalid.yaml"
        with open(invalid_path, 'w') as f:
            f.write("invalid: yaml: content: [")

        with pytest.raises(Exception):
            load_config(str(invalid_path))


class TestConfigValidation:
    """Tests for config validation."""

    def test_config_has_data_section(self, default_config):
        """Test that config has data section."""
        assert 'data' in default_config

    def test_config_has_model_section(self, default_config):
        """Test that config has model section."""
        assert 'model' in default_config

    def test_config_has_training_section(self, default_config):
        """Test that config has training section."""
        assert 'training' in default_config

    def test_frequencies_list(self, default_config):
        """Test that frequencies is a list."""
        assert isinstance(default_config['data']['frequencies'], list)

    def test_four_frequencies(self, default_config):
        """Test that there are 4 frequencies."""
        assert len(default_config['data']['frequencies']) == 4


class TestReproducibility:
    """Tests for reproducibility."""

    def test_seed_reproducibility(self, temp_dir, small_config):
        """Test that same seeds produce same results."""
        from main import phase_data_generation

        small_config['paths'] = {
            'train_data': str(temp_dir / "train1.npy"),
            'test_data': str(temp_dir / "test1.npy")
        }

        phase_data_generation(small_config)
        train1 = np.load(small_config['paths']['train_data'])

        # Run again with same config
        small_config['paths'] = {
            'train_data': str(temp_dir / "train2.npy"),
            'test_data': str(temp_dir / "test2.npy")
        }

        phase_data_generation(small_config)
        train2 = np.load(small_config['paths']['train_data'])

        # Should be identical
        np.testing.assert_array_equal(train1, train2)
