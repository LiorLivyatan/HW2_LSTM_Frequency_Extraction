"""
Tests for src/dataset.py

Tests the FrequencyDataset PyTorch Dataset wrapper.
Target: ~33 tests covering tensor shapes, one-hot encoding, indexing.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import FrequencyDataset
from torch.utils.data import DataLoader


class TestFrequencyDatasetInit:
    """Tests for FrequencyDataset.__init__"""

    def test_load_valid_dataset(self, small_dataset):
        """Test loading a valid dataset."""
        filepath, _ = small_dataset
        dataset = FrequencyDataset(str(filepath))
        assert dataset is not None

    def test_correct_length(self, small_dataset):
        """Test that dataset reports correct length."""
        filepath, data = small_dataset
        dataset = FrequencyDataset(str(filepath))
        assert len(dataset) == len(data)

    def test_input_target_split(self, small_dataset):
        """Test that input and target are correctly split."""
        filepath, data = small_dataset
        dataset = FrequencyDataset(str(filepath))

        # Input should be first 5 columns, target should be column 5
        inputs, target = dataset[0]
        assert inputs.shape == (5,)
        assert target.shape == (1,)

    def test_data_conversion_to_float32(self, small_dataset):
        """Test that data is converted to float32."""
        filepath, _ = small_dataset
        dataset = FrequencyDataset(str(filepath))

        inputs, target = dataset[0]
        assert inputs.dtype == torch.float32
        assert target.dtype == torch.float32

    def test_file_not_found_raises_error(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            FrequencyDataset("nonexistent_file.npy")

    def test_wrong_column_count_raises_error(self, temp_data_dir):
        """Test that wrong number of columns raises ValueError."""
        # Create dataset with wrong shape
        wrong_data = np.random.randn(100, 5).astype(np.float32)  # 5 cols instead of 6
        filepath = temp_data_dir / "wrong_shape.npy"
        np.save(filepath, wrong_data)

        with pytest.raises(ValueError):
            FrequencyDataset(str(filepath))


class TestFrequencyDatasetLen:
    """Tests for FrequencyDataset.__len__"""

    def test_returns_correct_count(self, small_dataset):
        """Test that __len__ returns correct count."""
        filepath, data = small_dataset
        dataset = FrequencyDataset(str(filepath))
        assert len(dataset) == len(data)

    def test_matches_numpy_length(self, small_dataset):
        """Test that length matches numpy array length."""
        filepath, data = small_dataset
        dataset = FrequencyDataset(str(filepath))
        assert len(dataset) == data.shape[0]


class TestFrequencyDatasetGetitem:
    """Tests for FrequencyDataset.__getitem__"""

    def test_returns_tuple(self, small_dataset):
        """Test that __getitem__ returns a tuple."""
        filepath, _ = small_dataset
        dataset = FrequencyDataset(str(filepath))
        result = dataset[0]
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_input_tensor_shape(self, small_dataset):
        """Test that input tensor has shape (5,)."""
        filepath, _ = small_dataset
        dataset = FrequencyDataset(str(filepath))
        inputs, _ = dataset[0]
        assert inputs.shape == (5,)

    def test_target_tensor_shape(self, small_dataset):
        """Test that target tensor has shape (1,)."""
        filepath, _ = small_dataset
        dataset = FrequencyDataset(str(filepath))
        _, target = dataset[0]
        assert target.shape == (1,)

    def test_input_tensor_dtype(self, small_dataset):
        """Test that input tensor is float32."""
        filepath, _ = small_dataset
        dataset = FrequencyDataset(str(filepath))
        inputs, _ = dataset[0]
        assert inputs.dtype == torch.float32

    def test_target_tensor_dtype(self, small_dataset):
        """Test that target tensor is float32."""
        filepath, _ = small_dataset
        dataset = FrequencyDataset(str(filepath))
        _, target = dataset[0]
        assert target.dtype == torch.float32

    def test_first_index(self, small_dataset):
        """Test accessing first sample."""
        filepath, data = small_dataset
        dataset = FrequencyDataset(str(filepath))
        inputs, target = dataset[0]

        expected_inputs = torch.tensor(data[0, :5], dtype=torch.float32)
        expected_target = torch.tensor([data[0, 5]], dtype=torch.float32)

        torch.testing.assert_close(inputs, expected_inputs)
        torch.testing.assert_close(target, expected_target)

    def test_last_index(self, small_dataset):
        """Test accessing last sample."""
        filepath, data = small_dataset
        dataset = FrequencyDataset(str(filepath))
        inputs, target = dataset[-1]

        expected_inputs = torch.tensor(data[-1, :5], dtype=torch.float32)
        expected_target = torch.tensor([data[-1, 5]], dtype=torch.float32)

        torch.testing.assert_close(inputs, expected_inputs)
        torch.testing.assert_close(target, expected_target)

    def test_negative_index(self, small_dataset):
        """Test that negative indexing works."""
        filepath, data = small_dataset
        dataset = FrequencyDataset(str(filepath))

        inputs_neg, target_neg = dataset[-1]
        inputs_pos, target_pos = dataset[len(dataset) - 1]

        torch.testing.assert_close(inputs_neg, inputs_pos)
        torch.testing.assert_close(target_neg, target_pos)

    def test_out_of_bounds_raises_error(self, small_dataset):
        """Test that out-of-bounds index raises IndexError."""
        filepath, _ = small_dataset
        dataset = FrequencyDataset(str(filepath))

        with pytest.raises(IndexError):
            _ = dataset[len(dataset)]

    def test_correct_values_extracted(self, small_dataset):
        """Test that correct values are extracted from data."""
        filepath, data = small_dataset
        dataset = FrequencyDataset(str(filepath))

        # Check middle sample
        mid_idx = len(data) // 2
        inputs, target = dataset[mid_idx]

        expected_inputs = torch.tensor(data[mid_idx, :5], dtype=torch.float32)
        expected_target = torch.tensor([data[mid_idx, 5]], dtype=torch.float32)

        torch.testing.assert_close(inputs, expected_inputs)
        torch.testing.assert_close(target, expected_target)

    def test_one_hot_vector_sum(self, small_dataset):
        """Test that one-hot vector (inputs[1:5]) sums to 1."""
        filepath, _ = small_dataset
        dataset = FrequencyDataset(str(filepath))

        for i in range(len(dataset)):
            inputs, _ = dataset[i]
            one_hot_sum = inputs[1:5].sum().item()
            assert one_hot_sum == 1.0, f"One-hot sum at index {i} is {one_hot_sum}"

    def test_tensor_independence(self, small_dataset):
        """Test that tensors can be converted to independent copies."""
        filepath, _ = small_dataset
        dataset = FrequencyDataset(str(filepath))

        inputs1, target1 = dataset[0]
        inputs2, target2 = dataset[0]

        # Clone to make independent copies
        inputs1_clone = inputs1.clone()
        inputs1_clone[0] = 999.0

        # Original should be unchanged
        assert inputs1[0].item() != 999.0
        assert inputs2[0].item() != 999.0


class TestGetSampleInfo:
    """Tests for FrequencyDataset.get_sample_info"""

    def test_returns_dict(self, small_dataset):
        """Test that get_sample_info returns a dict."""
        filepath, _ = small_dataset
        dataset = FrequencyDataset(str(filepath))
        info = dataset.get_sample_info(0)
        assert isinstance(info, dict)

    def test_dict_has_required_keys(self, small_dataset):
        """Test that returned dict has all required keys."""
        filepath, _ = small_dataset
        dataset = FrequencyDataset(str(filepath))
        info = dataset.get_sample_info(0)

        required_keys = ['index', 'selected_frequency', 'noisy_signal', 'one_hot', 'target']
        for key in required_keys:
            assert key in info, f"Missing key: {key}"

    def test_frequency_mapping_1hz(self, tiny_dataset, temp_data_dir):
        """Test frequency mapping for 1Hz (index 0)."""
        filepath = temp_data_dir / "tiny.npy"
        np.save(filepath, tiny_dataset)
        dataset = FrequencyDataset(str(filepath))

        info = dataset.get_sample_info(0)  # First frequency block
        assert info['selected_frequency'] == '1Hz'

    def test_frequency_mapping_3hz(self, tiny_dataset, temp_data_dir):
        """Test frequency mapping for 3Hz (index 1)."""
        filepath = temp_data_dir / "tiny.npy"
        np.save(filepath, tiny_dataset)
        dataset = FrequencyDataset(str(filepath))

        info = dataset.get_sample_info(10)  # Second frequency block
        assert info['selected_frequency'] == '3Hz'

    def test_frequency_mapping_5hz(self, tiny_dataset, temp_data_dir):
        """Test frequency mapping for 5Hz (index 2)."""
        filepath = temp_data_dir / "tiny.npy"
        np.save(filepath, tiny_dataset)
        dataset = FrequencyDataset(str(filepath))

        info = dataset.get_sample_info(20)  # Third frequency block
        assert info['selected_frequency'] == '5Hz'

    def test_frequency_mapping_7hz(self, tiny_dataset, temp_data_dir):
        """Test frequency mapping for 7Hz (index 3)."""
        filepath = temp_data_dir / "tiny.npy"
        np.save(filepath, tiny_dataset)
        dataset = FrequencyDataset(str(filepath))

        info = dataset.get_sample_info(30)  # Fourth frequency block
        assert info['selected_frequency'] == '7Hz'

    def test_noisy_signal_matches_input(self, small_dataset):
        """Test that noisy_signal matches input[0]."""
        filepath, data = small_dataset
        dataset = FrequencyDataset(str(filepath))

        info = dataset.get_sample_info(0)
        inputs, _ = dataset[0]

        assert info['noisy_signal'] == inputs[0].item()

    def test_one_hot_matches_input(self, small_dataset):
        """Test that one_hot matches input[1:5]."""
        filepath, _ = small_dataset
        dataset = FrequencyDataset(str(filepath))

        info = dataset.get_sample_info(0)
        inputs, _ = dataset[0]

        expected_one_hot = inputs[1:5].numpy()
        np.testing.assert_array_equal(info['one_hot'], expected_one_hot)

    def test_target_matches(self, small_dataset):
        """Test that target value matches."""
        filepath, _ = small_dataset
        dataset = FrequencyDataset(str(filepath))

        info = dataset.get_sample_info(0)
        _, target = dataset[0]

        assert info['target'] == target[0].item()


class TestDataLoaderIntegration:
    """Tests for DataLoader integration."""

    def test_dataloader_creation(self, small_dataset):
        """Test that DataLoader can be created."""
        filepath, _ = small_dataset
        dataset = FrequencyDataset(str(filepath))
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        assert loader is not None

    def test_dataloader_iteration(self, small_dataset):
        """Test that DataLoader can be iterated."""
        filepath, _ = small_dataset
        dataset = FrequencyDataset(str(filepath))
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        batch_count = 0
        for inputs, targets in loader:
            batch_count += 1
            assert inputs.shape[1] == 5
            assert targets.shape[1] == 1

        assert batch_count > 0

    def test_batch_shapes(self, small_dataset):
        """Test that batches have correct shapes."""
        filepath, _ = small_dataset
        dataset = FrequencyDataset(str(filepath))
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        inputs, targets = next(iter(loader))
        assert inputs.shape == (4, 5)
        assert targets.shape == (4, 1)

    def test_last_batch_size(self, small_dataset):
        """Test handling of last batch with different size."""
        filepath, data = small_dataset
        dataset = FrequencyDataset(str(filepath))

        batch_size = 7  # Won't divide evenly
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        total_samples = 0
        for inputs, targets in loader:
            total_samples += len(inputs)

        assert total_samples == len(data)

    def test_shuffle_false_preserves_order(self, small_dataset):
        """Test that shuffle=False preserves data order."""
        filepath, data = small_dataset
        dataset = FrequencyDataset(str(filepath))
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        inputs, _ = next(iter(loader))

        # First batch should match first 4 samples
        for i in range(4):
            expected = torch.tensor(data[i, :5], dtype=torch.float32)
            torch.testing.assert_close(inputs[i], expected)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_sample_dataset(self, temp_data_dir):
        """Test dataset with single sample."""
        data = np.random.randn(1, 6).astype(np.float32)
        data[0, 1:5] = [1, 0, 0, 0]  # Valid one-hot
        filepath = temp_data_dir / "single.npy"
        np.save(filepath, data)

        dataset = FrequencyDataset(str(filepath))
        assert len(dataset) == 1

        inputs, target = dataset[0]
        assert inputs.shape == (5,)
        assert target.shape == (1,)

    def test_large_values(self, temp_data_dir):
        """Test dataset with large values."""
        data = np.ones((10, 6), dtype=np.float32) * 1e6
        # Set valid one-hot
        data[:, 1:5] = 0
        data[:, 1] = 1

        filepath = temp_data_dir / "large.npy"
        np.save(filepath, data)

        dataset = FrequencyDataset(str(filepath))
        inputs, target = dataset[0]

        assert torch.isfinite(inputs).all()
        assert torch.isfinite(target).all()

    def test_small_values(self, temp_data_dir):
        """Test dataset with very small values."""
        data = np.ones((10, 6), dtype=np.float32) * 1e-6
        # Set valid one-hot
        data[:, 1:5] = 0
        data[:, 1] = 1

        filepath = temp_data_dir / "small.npy"
        np.save(filepath, data)

        dataset = FrequencyDataset(str(filepath))
        inputs, target = dataset[0]

        assert torch.isfinite(inputs).all()
        assert torch.isfinite(target).all()

    def test_negative_values(self, temp_data_dir):
        """Test dataset with negative values."""
        data = np.ones((10, 6), dtype=np.float32) * -1.0
        # Set valid one-hot (these must be positive)
        data[:, 1:5] = 0
        data[:, 1] = 1

        filepath = temp_data_dir / "negative.npy"
        np.save(filepath, data)

        dataset = FrequencyDataset(str(filepath))
        inputs, target = dataset[0]

        assert inputs[0].item() == -1.0
        assert target[0].item() == -1.0
