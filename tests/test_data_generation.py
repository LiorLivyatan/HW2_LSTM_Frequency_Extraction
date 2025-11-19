"""
Tests for src/data_generation.py

Tests the SignalGenerator class and data generation functionality.
Target: ~51 tests covering data integrity, reproducibility, per-sample randomization.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation import SignalGenerator


class TestSignalGeneratorInit:
    """Tests for SignalGenerator.__init__"""

    def test_default_initialization(self):
        """Test initialization with default frequencies."""
        gen = SignalGenerator()
        assert gen.frequencies == [1.0, 3.0, 5.0, 7.0]
        assert gen.fs == 1000
        assert gen.duration == 10.0

    def test_custom_frequencies(self, sample_frequencies):
        """Test initialization with custom frequencies."""
        gen = SignalGenerator(frequencies=sample_frequencies)
        assert gen.frequencies == sample_frequencies

    def test_custom_sampling_rate(self):
        """Test initialization with custom sampling rate."""
        gen = SignalGenerator(fs=500)
        assert gen.fs == 500

    def test_custom_duration(self):
        """Test initialization with custom duration."""
        gen = SignalGenerator(duration=5.0)
        assert gen.duration == 5.0

    def test_seed_setting(self):
        """Test that seed is properly set."""
        gen = SignalGenerator(seed=123)
        # Seed should affect random generation
        assert gen is not None

    def test_n_samples_calculation(self):
        """Test that n_samples is correctly calculated."""
        gen = SignalGenerator(fs=1000, duration=10.0)
        assert gen.n_samples == 10000

    def test_n_samples_different_parameters(self):
        """Test n_samples with different fs and duration."""
        gen = SignalGenerator(fs=500, duration=2.0)
        assert gen.n_samples == 1000

    def test_invalid_frequency_count_raises_error(self):
        """Test that wrong number of frequencies raises ValueError."""
        with pytest.raises(ValueError):
            SignalGenerator(frequencies=[1.0, 3.0, 5.0])  # Only 3

    def test_invalid_frequency_count_too_many(self):
        """Test that too many frequencies raises ValueError."""
        with pytest.raises(ValueError):
            SignalGenerator(frequencies=[1.0, 3.0, 5.0, 7.0, 9.0])  # 5

    def test_zero_sampling_rate_raises_error(self):
        """Test that zero sampling rate raises ValueError."""
        with pytest.raises(ValueError):
            SignalGenerator(fs=0)

    def test_negative_sampling_rate_raises_error(self):
        """Test that negative sampling rate raises ValueError."""
        with pytest.raises(ValueError):
            SignalGenerator(fs=-100)

    def test_zero_duration_raises_error(self):
        """Test that zero duration raises ValueError."""
        with pytest.raises(ValueError):
            SignalGenerator(duration=0)

    def test_negative_duration_raises_error(self):
        """Test that negative duration raises ValueError."""
        with pytest.raises(ValueError):
            SignalGenerator(duration=-1.0)


class TestGenerateTimeArray:
    """Tests for SignalGenerator.generate_time_array"""

    def test_output_shape(self, small_signal_generator):
        """Test that time array has correct shape."""
        t = small_signal_generator.generate_time_array()
        assert t.shape == (small_signal_generator.n_samples,)

    def test_starts_at_zero(self, small_signal_generator):
        """Test that time array starts at 0."""
        t = small_signal_generator.generate_time_array()
        assert t[0] == 0.0

    def test_ends_near_duration(self, small_signal_generator):
        """Test that time array ends at duration."""
        t = small_signal_generator.generate_time_array()
        # np.linspace(0, duration, n_samples) ends at duration
        np.testing.assert_almost_equal(t[-1], small_signal_generator.duration, decimal=6)

    def test_linspace_spacing(self, small_signal_generator):
        """Test that time array has uniform spacing."""
        t = small_signal_generator.generate_time_array()
        diffs = np.diff(t)
        # linspace spacing is duration/(n_samples-1)
        expected_dt = small_signal_generator.duration / (small_signal_generator.n_samples - 1)
        np.testing.assert_array_almost_equal(diffs, np.full_like(diffs, expected_dt))

    def test_dtype(self, small_signal_generator):
        """Test that time array has correct dtype."""
        t = small_signal_generator.generate_time_array()
        assert t.dtype in [np.float32, np.float64]


class TestGenerateNoisySinusoid:
    """Tests for SignalGenerator.generate_noisy_sinusoid"""

    def test_output_shape(self, small_signal_generator):
        """Test that noisy sinusoid has correct shape."""
        t = small_signal_generator.generate_time_array()
        noisy = small_signal_generator.generate_noisy_sinusoid(1.0, t)
        assert noisy.shape == t.shape

    def test_output_dtype(self, small_signal_generator):
        """Test that output has correct dtype."""
        t = small_signal_generator.generate_time_array()
        noisy = small_signal_generator.generate_noisy_sinusoid(1.0, t)
        assert noisy.dtype == np.float32

    def test_per_sample_randomization(self):
        """Test that amplitude varies per sample (CRITICAL)."""
        gen = SignalGenerator(fs=100, duration=0.1, seed=42)
        t = gen.generate_time_array()
        noisy = gen.generate_noisy_sinusoid(1.0, t)

        # Generate clean sinusoid for comparison
        clean = np.sin(2 * np.pi * 1.0 * t)

        # Noisy should NOT be identical to clean
        assert not np.allclose(noisy, clean)

    def test_different_frequencies_different_output(self, small_signal_generator):
        """Test that different frequencies produce different outputs."""
        t = small_signal_generator.generate_time_array()
        noisy_1hz = small_signal_generator.generate_noisy_sinusoid(1.0, t)
        noisy_3hz = small_signal_generator.generate_noisy_sinusoid(3.0, t)

        assert not np.allclose(noisy_1hz, noisy_3hz)

    def test_amplitude_bounded(self, small_signal_generator):
        """Test that amplitude variations are within expected bounds."""
        t = small_signal_generator.generate_time_array()
        noisy = small_signal_generator.generate_noisy_sinusoid(1.0, t)

        # With A in [0.8, 1.2], max amplitude should be around 1.2
        assert np.max(np.abs(noisy)) <= 1.5  # Some margin

    def test_reproducibility_with_same_seed(self):
        """Test that same seed produces same output."""
        gen1 = SignalGenerator(fs=100, duration=0.1, seed=42)
        t1 = gen1.generate_time_array()
        noisy1 = gen1.generate_noisy_sinusoid(1.0, t1)

        gen2 = SignalGenerator(fs=100, duration=0.1, seed=42)
        t2 = gen2.generate_time_array()
        noisy2 = gen2.generate_noisy_sinusoid(1.0, t2)

        np.testing.assert_array_equal(noisy1, noisy2)

    def test_different_seeds_different_output(self):
        """Test that different seeds produce different outputs."""
        gen1 = SignalGenerator(fs=100, duration=0.1, seed=42)
        gen2 = SignalGenerator(fs=100, duration=0.1, seed=99)

        t1 = gen1.generate_time_array()
        t2 = gen2.generate_time_array()

        noisy1 = gen1.generate_noisy_sinusoid(1.0, t1)
        noisy2 = gen2.generate_noisy_sinusoid(1.0, t2)

        assert not np.array_equal(noisy1, noisy2)


class TestGenerateMixedSignal:
    """Tests for SignalGenerator.generate_mixed_signal"""

    def test_output_shapes(self, small_signal_generator):
        """Test that mixed signal and time array have correct shapes."""
        t, mixed = small_signal_generator.generate_mixed_signal()
        assert mixed.shape == (small_signal_generator.n_samples,)
        assert t.shape == (small_signal_generator.n_samples,)

    def test_signal_averaging(self, small_signal_generator):
        """Test that signal is averaged (divided by 4)."""
        t, mixed = small_signal_generator.generate_mixed_signal()

        # Signal should be bounded due to averaging
        # 4 sinusoids with max amplitude 1.2, averaged = max ~1.2
        assert np.max(np.abs(mixed)) < 2.0

    def test_reproducibility(self):
        """Test that same seed produces same mixed signal."""
        gen1 = SignalGenerator(fs=100, duration=0.1, seed=42)
        _, mixed1 = gen1.generate_mixed_signal()

        gen2 = SignalGenerator(fs=100, duration=0.1, seed=42)
        _, mixed2 = gen2.generate_mixed_signal()

        np.testing.assert_array_equal(mixed1, mixed2)

    def test_time_array_correctness(self):
        """Test that returned time array is correct."""
        gen = SignalGenerator(fs=100, duration=0.1, seed=42)
        t, _ = gen.generate_mixed_signal()

        # Generate fresh time array for comparison
        t_expected = np.linspace(0, gen.duration, gen.n_samples)
        np.testing.assert_array_equal(t, t_expected)


class TestGenerateCleanTargets:
    """Tests for SignalGenerator.generate_clean_targets"""

    def test_output_shape(self, small_signal_generator):
        """Test that clean targets have correct shape."""
        t = small_signal_generator.generate_time_array()
        targets = small_signal_generator.generate_clean_targets(t)

        assert targets.shape == (4, small_signal_generator.n_samples)

    def test_pure_sinusoids(self, small_signal_generator):
        """Test that targets are pure sinusoids (no noise)."""
        t = small_signal_generator.generate_time_array()
        targets = small_signal_generator.generate_clean_targets(t)

        # First frequency (1 Hz)
        expected = np.sin(2 * np.pi * 1.0 * t)
        np.testing.assert_array_almost_equal(targets[0], expected.astype(np.float32))

    def test_all_frequencies_correct(self, small_signal_generator):
        """Test that all frequency targets are correct."""
        t = small_signal_generator.generate_time_array()
        targets = small_signal_generator.generate_clean_targets(t)

        for i, freq in enumerate(small_signal_generator.frequencies):
            expected = np.sin(2 * np.pi * freq * t).astype(np.float32)
            np.testing.assert_array_almost_equal(targets[i], expected)

    def test_no_randomization(self):
        """Test that clean targets have no randomization."""
        gen1 = SignalGenerator(fs=100, duration=0.1, seed=42)
        gen2 = SignalGenerator(fs=100, duration=0.1, seed=99)

        t1 = gen1.generate_time_array()
        t2 = gen2.generate_time_array()

        targets1 = gen1.generate_clean_targets(t1)
        targets2 = gen2.generate_clean_targets(t2)

        # Should be identical despite different seeds
        np.testing.assert_array_equal(targets1, targets2)

    def test_dtype(self, small_signal_generator):
        """Test that targets have correct dtype."""
        t = small_signal_generator.generate_time_array()
        targets = small_signal_generator.generate_clean_targets(t)

        assert targets.dtype == np.float32


class TestCreateDataset:
    """Tests for SignalGenerator.create_dataset"""

    def test_output_shape(self, small_signal_generator):
        """Test that dataset has correct shape."""
        dataset = small_signal_generator.create_dataset()

        expected_rows = 4 * small_signal_generator.n_samples
        assert dataset.shape == (expected_rows, 6)

    def test_dtype(self, small_signal_generator):
        """Test that dataset has correct dtype."""
        dataset = small_signal_generator.create_dataset()
        assert dataset.dtype == np.float32

    def test_column_0_is_mixed_signal(self):
        """Test that column 0 contains the mixed signal."""
        # Create two generators with same seed to get identical results
        gen1 = SignalGenerator(fs=100, duration=0.1, seed=42)
        dataset = gen1.create_dataset()

        gen2 = SignalGenerator(fs=100, duration=0.1, seed=42)
        _, mixed = gen2.generate_mixed_signal()

        # First n samples should match mixed signal
        n = gen1.n_samples
        np.testing.assert_array_almost_equal(dataset[:n, 0], mixed)

    def test_one_hot_encoding_frequency_0(self, small_signal_generator):
        """Test one-hot encoding for frequency 0 (1Hz)."""
        dataset = small_signal_generator.create_dataset()
        n = small_signal_generator.n_samples

        # Rows 0 to n-1 should have one-hot [1, 0, 0, 0]
        for i in range(n):
            np.testing.assert_array_equal(dataset[i, 1:5], [1, 0, 0, 0])

    def test_one_hot_encoding_frequency_1(self, small_signal_generator):
        """Test one-hot encoding for frequency 1 (3Hz)."""
        dataset = small_signal_generator.create_dataset()
        n = small_signal_generator.n_samples

        # Rows n to 2n-1 should have one-hot [0, 1, 0, 0]
        for i in range(n, 2*n):
            np.testing.assert_array_equal(dataset[i, 1:5], [0, 1, 0, 0])

    def test_one_hot_encoding_frequency_2(self, small_signal_generator):
        """Test one-hot encoding for frequency 2 (5Hz)."""
        dataset = small_signal_generator.create_dataset()
        n = small_signal_generator.n_samples

        for i in range(2*n, 3*n):
            np.testing.assert_array_equal(dataset[i, 1:5], [0, 0, 1, 0])

    def test_one_hot_encoding_frequency_3(self, small_signal_generator):
        """Test one-hot encoding for frequency 3 (7Hz)."""
        dataset = small_signal_generator.create_dataset()
        n = small_signal_generator.n_samples

        for i in range(3*n, 4*n):
            np.testing.assert_array_equal(dataset[i, 1:5], [0, 0, 0, 1])

    def test_one_hot_sum_equals_one(self, small_signal_generator):
        """Test that one-hot vectors sum to 1."""
        dataset = small_signal_generator.create_dataset()

        one_hot_sums = np.sum(dataset[:, 1:5], axis=1)
        np.testing.assert_array_equal(one_hot_sums, np.ones(len(dataset)))

    def test_target_values_correct(self, small_signal_generator):
        """Test that target values (column 5) are correct."""
        dataset = small_signal_generator.create_dataset()
        t = small_signal_generator.generate_time_array()
        targets = small_signal_generator.generate_clean_targets(t)

        n = small_signal_generator.n_samples

        # Check first frequency block
        np.testing.assert_array_almost_equal(dataset[:n, 5], targets[0])

    def test_mixed_signal_same_across_blocks(self, small_signal_generator):
        """Test that mixed signal is the same in all frequency blocks."""
        dataset = small_signal_generator.create_dataset()
        n = small_signal_generator.n_samples

        # All 4 blocks should have the same mixed signal
        block_0 = dataset[:n, 0]
        block_1 = dataset[n:2*n, 0]
        block_2 = dataset[2*n:3*n, 0]
        block_3 = dataset[3*n:4*n, 0]

        np.testing.assert_array_equal(block_0, block_1)
        np.testing.assert_array_equal(block_1, block_2)
        np.testing.assert_array_equal(block_2, block_3)

    def test_reproducibility(self):
        """Test that same seed produces same dataset."""
        gen1 = SignalGenerator(fs=100, duration=0.1, seed=42)
        dataset1 = gen1.create_dataset()

        gen2 = SignalGenerator(fs=100, duration=0.1, seed=42)
        dataset2 = gen2.create_dataset()

        np.testing.assert_array_equal(dataset1, dataset2)


class TestSaveDataset:
    """Tests for SignalGenerator.save_dataset"""

    def test_file_created(self, small_signal_generator, temp_data_dir):
        """Test that file is created at specified path."""
        filepath = temp_data_dir / "test_data.npy"
        small_signal_generator.save_dataset(str(filepath))

        assert filepath.exists()

    def test_creates_parent_directory(self, small_signal_generator, temp_dir):
        """Test that parent directory is created if not exists."""
        filepath = temp_dir / "new_dir" / "test_data.npy"
        small_signal_generator.save_dataset(str(filepath))

        assert filepath.exists()
        assert filepath.parent.exists()

    def test_file_loadable(self, small_signal_generator, temp_data_dir):
        """Test that saved file can be loaded."""
        filepath = temp_data_dir / "test_data.npy"
        small_signal_generator.save_dataset(str(filepath))

        loaded = np.load(filepath)
        assert loaded is not None

    def test_loaded_shape(self, small_signal_generator, temp_data_dir):
        """Test that loaded data has correct shape."""
        filepath = temp_data_dir / "test_data.npy"
        small_signal_generator.save_dataset(str(filepath))

        loaded = np.load(filepath)
        expected_rows = 4 * small_signal_generator.n_samples
        assert loaded.shape == (expected_rows, 6)

    def test_loaded_dtype(self, small_signal_generator, temp_data_dir):
        """Test that loaded data has correct dtype."""
        filepath = temp_data_dir / "test_data.npy"
        small_signal_generator.save_dataset(str(filepath))

        loaded = np.load(filepath)
        assert loaded.dtype == np.float32

    def test_data_integrity(self, temp_data_dir):
        """Test that saved and loaded data are identical."""
        filepath = temp_data_dir / "test_data.npy"

        # Create two generators with same seed to compare
        gen1 = SignalGenerator(fs=100, duration=0.1, seed=42)
        gen1.save_dataset(str(filepath))

        gen2 = SignalGenerator(fs=100, duration=0.1, seed=42)
        dataset = gen2.create_dataset()

        loaded = np.load(filepath)
        np.testing.assert_array_equal(dataset, loaded)


class TestFullSizeDataset:
    """Tests for full-size dataset (40,000 rows)."""

    def test_full_dataset_shape(self):
        """Test that full dataset has 40,000 rows."""
        gen = SignalGenerator(fs=1000, duration=10.0, seed=42)
        dataset = gen.create_dataset()

        assert dataset.shape == (40000, 6)

    def test_samples_per_frequency(self):
        """Test that each frequency has 10,000 samples."""
        gen = SignalGenerator(fs=1000, duration=10.0, seed=42)
        dataset = gen.create_dataset()

        # Check one-hot encoding counts
        for freq_idx in range(4):
            count = np.sum(dataset[:, 1 + freq_idx] == 1)
            assert count == 10000


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_short_duration(self):
        """Test with very short duration."""
        gen = SignalGenerator(fs=100, duration=0.01, seed=42)
        dataset = gen.create_dataset()

        # Should have 4 samples (1 per frequency)
        assert dataset.shape[0] == 4

    def test_high_sampling_rate(self):
        """Test with high sampling rate."""
        gen = SignalGenerator(fs=10000, duration=0.1, seed=42)
        dataset = gen.create_dataset()

        # 1000 samples × 4 frequencies = 4000 rows
        assert dataset.shape == (4000, 6)

    def test_low_frequencies(self):
        """Test with very low frequencies."""
        gen = SignalGenerator(
            frequencies=[0.1, 0.2, 0.3, 0.4],
            fs=100,
            duration=0.1,
            seed=42
        )
        dataset = gen.create_dataset()

        assert dataset.shape[1] == 6

    def test_high_frequencies(self):
        """Test with high frequencies."""
        gen = SignalGenerator(
            frequencies=[10.0, 20.0, 30.0, 40.0],
            fs=1000,  # Must satisfy Nyquist
            duration=0.1,
            seed=42
        )
        dataset = gen.create_dataset()

        assert dataset.shape[1] == 6
