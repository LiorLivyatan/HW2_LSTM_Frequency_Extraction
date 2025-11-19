# Phase 1: Data Generation PRD

**Phase**: 1 of 6
**Priority**: Highest
**Estimated Effort**: 2-3 hours
**Dependencies**: None
**Enables**: Phase 3 (Training), Phase 4 (Evaluation)

---

## Table of Contents
1. [Objective](#objective)
2. [Requirements](#requirements)
3. [Architecture](#architecture)
4. [Implementation Details](#implementation-details)
5. [Testing Strategy](#testing-strategy)
6. [Deliverables](#deliverables)
7. [Success Criteria](#success-criteria)
8. [Risks and Mitigation](#risks-and-mitigation)

---

## Objective

Create reproducible, high-quality training and test datasets for the LSTM frequency extraction system.

### What This Phase Delivers
- Two datasets with **completely different noise patterns** (different seeds)
- **40,000 training examples** from 10,000 time samples × 4 frequencies
- **Properly structured data** ready for PyTorch DataLoader
- **Validated signal quality** through FFT analysis

### Why This Phase is Critical
- Foundation for all subsequent phases
- Data quality directly impacts model performance
- Incorrect randomization will cause the model to fail
- Seed separation enables generalization testing

---

## Requirements

### Functional Requirements

#### FR1: Signal Parameters (EXACT SPECIFICATION)
- [ ] **Frequencies**: f₁=1Hz, f₂=3Hz, f₃=5Hz, f₄=7Hz (EXACT)
- [ ] **Time Domain**: 0 to 10 seconds
- [ ] **Sampling Rate**: Fs = 1000 Hz
- [ ] **Total Samples**: 10,000 time points

#### FR2: Noisy Signal Generation (CRITICAL)
- [ ] **Random Amplitude**: A_i(t) ~ Uniform(0.8, 1.2) **for EVERY sample t**
- [ ] **Random Phase**: φ_i(t) ~ Uniform(0, 2π) **for EVERY sample t**
- [ ] **Noisy Sinusoid Formula**:
  ```
  Sinus_i^noisy(t) = A_i(t) · sin(2π · f_i · t + φ_i(t))
  ```
- [ ] **Mixed Signal**: S(t) = (1/4) · Σ(i=1 to 4) Sinus_i^noisy(t)

#### FR3: Clean Target Generation
- [ ] **Pure Sinusoid Formula**:
  ```
  Target_i(t) = sin(2π · f_i · t)
  ```
- [ ] **No randomization**: Amplitude=1.0, Phase=0 (constant)

#### FR4: Dataset Structure
- [ ] **Training Set**: Use Seed #1, create 40,000 rows
- [ ] **Test Set**: Use Seed #2, create 40,000 rows
- [ ] **Row Format**: [S(t), C1, C2, C3, C4, Target_i(t)] (6 values)
- [ ] **One-hot Vector C**: [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]

#### FR5: Data Persistence
- [ ] Save as NumPy `.npy` files for fast loading
- [ ] File paths: `data/train_data.npy`, `data/test_data.npy`

### Non-Functional Requirements

#### NFR1: Performance
- Generation time: < 10 seconds for both datasets
- Memory usage: < 500 MB during generation
- File size: ~1.92 MB per dataset (40,000 × 6 × 8 bytes)

#### NFR2: Reproducibility
- Deterministic output given same seed
- No hidden randomness sources
- Documented seed values

#### NFR3: Code Quality
- Clean, modular code structure
- Type hints for function signatures
- Comprehensive docstrings
- Unit test coverage > 80%

---

## Architecture

### Component Overview

```
SignalGenerator
    │
    ├── __init__(frequencies, fs, duration, seed)
    │   └── Initialize parameters and random state
    │
    ├── generate_time_array()
    │   └── Returns: np.array of 10,000 time points [0, 10]
    │
    ├── generate_noisy_sinusoid(freq_idx, t_array)
    │   ├── Generate random A_i(t) for each t
    │   ├── Generate random φ_i(t) for each t
    │   └── Returns: noisy sinusoid array (10,000,)
    │
    ├── generate_mixed_signal()
    │   ├── Generate 4 noisy sinusoids
    │   ├── Sum and normalize by 1/4
    │   └── Returns: S(t) array (10,000,)
    │
    ├── generate_clean_targets()
    │   ├── Generate pure sinusoid for each frequency
    │   └── Returns: 4 target arrays (4, 10,000)
    │
    └── create_dataset()
        ├── Call generate_mixed_signal()
        ├── Call generate_clean_targets()
        ├── Build 40,000 rows with one-hot vectors
        └── Returns: np.array (40,000, 6)
```

### Data Flow Diagram

```
┌──────────────────────────────────────────────────────┐
│  Input Parameters                                    │
│  • frequencies = [1, 3, 5, 7] Hz                    │
│  • fs = 1000 Hz                                      │
│  • duration = 10 sec                                 │
│  • seed = 1 or 2                                     │
└────────────┬─────────────────────────────────────────┘
             │
             v
┌──────────────────────────────────────────────────────┐
│  Generate Time Array                                 │
│  t = linspace(0, 10, 10000)                         │
└────────────┬─────────────────────────────────────────┘
             │
             v
┌──────────────────────────────────────────────────────┐
│  For each frequency f_i (4 frequencies):             │
│  ┌────────────────────────────────────────────────┐ │
│  │ For each time sample t (10,000 samples):       │ │
│  │   A_i(t) ~ Uniform(0.8, 1.2)                   │ │
│  │   φ_i(t) ~ Uniform(0, 2π)                      │ │
│  │   noisy[t] = A_i(t) · sin(2π·f_i·t + φ_i(t))  │ │
│  └────────────────────────────────────────────────┘ │
│                                                      │
│  S(t) = (1/4) · sum(all 4 noisy sinusoids)         │
└────────────┬─────────────────────────────────────────┘
             │
             v
┌──────────────────────────────────────────────────────┐
│  Generate Clean Targets (no randomization)           │
│  For each frequency f_i:                             │
│    Target_i(t) = sin(2π · f_i · t)                  │
└────────────┬─────────────────────────────────────────┘
             │
             v
┌──────────────────────────────────────────────────────┐
│  Build Dataset (40,000 rows)                         │
│  For freq_idx in [0, 1, 2, 3]:                      │
│    one_hot = [0,0,0,0]                              │
│    one_hot[freq_idx] = 1                            │
│                                                      │
│    For t_idx in range(10000):                       │
│      row = [S[t_idx],                               │
│             one_hot[0], one_hot[1],                 │
│             one_hot[2], one_hot[3],                 │
│             Target[freq_idx][t_idx]]                │
│      dataset.append(row)                            │
└────────────┬─────────────────────────────────────────┘
             │
             v
┌──────────────────────────────────────────────────────┐
│  Save to File                                        │
│  np.save('data/train_data.npy', dataset)            │
│  Shape: (40000, 6)                                   │
│  Dtype: float32                                      │
└──────────────────────────────────────────────────────┘
```

---

## Implementation Details

### Libraries Used

| Library | Purpose | Justification |
|---------|---------|---------------|
| **NumPy** | Array operations, random generation | Efficient vectorized operations, built-in random with seed control |
| **os/pathlib** | File path management | Create directories, handle paths |
| **typing** | Type hints | Code clarity and IDE support |

### Key Class: SignalGenerator

```python
import numpy as np
from typing import Tuple
from pathlib import Path

class SignalGenerator:
    """
    Generates synthetic mixed signals with random noise and clean targets
    for LSTM frequency extraction training.

    Attributes:
        frequencies (list): List of 4 frequencies in Hz [1, 3, 5, 7]
        fs (int): Sampling frequency in Hz (1000)
        duration (float): Signal duration in seconds (10.0)
        seed (int): Random seed for reproducibility (1 or 2)
        n_samples (int): Total number of time samples (10,000)
    """

    def __init__(
        self,
        frequencies: list = [1, 3, 5, 7],
        fs: int = 1000,
        duration: float = 10.0,
        seed: int = 1
    ):
        """Initialize signal generator with parameters."""
        self.frequencies = frequencies
        self.fs = fs
        self.duration = duration
        self.seed = seed
        self.n_samples = int(fs * duration)

        # Set random seed for reproducibility
        np.random.seed(self.seed)

    def generate_time_array(self) -> np.ndarray:
        """
        Generate time array from 0 to duration.

        Returns:
            np.ndarray: Time array of shape (n_samples,)
        """
        return np.linspace(0, self.duration, self.n_samples)

    def generate_noisy_sinusoid(
        self,
        freq: float,
        t_array: np.ndarray
    ) -> np.ndarray:
        """
        Generate a single noisy sinusoid with random amplitude and phase
        at EVERY sample.

        CRITICAL: A_i(t) and φ_i(t) must be different for each t!

        Args:
            freq: Frequency in Hz
            t_array: Time array of shape (n_samples,)

        Returns:
            np.ndarray: Noisy sinusoid of shape (n_samples,)
        """
        n = len(t_array)
        noisy_signal = np.zeros(n)

        for i, t in enumerate(t_array):
            # Generate random amplitude and phase for THIS sample
            A_t = np.random.uniform(0.8, 1.2)
            phi_t = np.random.uniform(0, 2 * np.pi)

            # Compute noisy sinusoid at time t
            noisy_signal[i] = A_t * np.sin(2 * np.pi * freq * t + phi_t)

        return noisy_signal

    def generate_mixed_signal(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate mixed signal S(t) from 4 noisy sinusoids.

        Returns:
            tuple:
                - t_array: Time points (10,000,)
                - S: Mixed signal (10,000,)
        """
        t_array = self.generate_time_array()

        # Generate all 4 noisy sinusoids
        noisy_sinusoids = []
        for freq in self.frequencies:
            noisy_sin = self.generate_noisy_sinusoid(freq, t_array)
            noisy_sinusoids.append(noisy_sin)

        # Sum and normalize
        S = sum(noisy_sinusoids) / 4.0

        return t_array, S

    def generate_clean_targets(self, t_array: np.ndarray) -> np.ndarray:
        """
        Generate clean target sinusoids (no noise).

        Args:
            t_array: Time array of shape (n_samples,)

        Returns:
            np.ndarray: Clean targets of shape (4, n_samples)
        """
        targets = []
        for freq in self.frequencies:
            # Pure sinusoid: amplitude=1, phase=0
            target = np.sin(2 * np.pi * freq * t_array)
            targets.append(target)

        return np.array(targets)  # Shape: (4, 10000)

    def create_dataset(self) -> np.ndarray:
        """
        Create complete dataset with 40,000 rows.

        Each row: [S(t), C1, C2, C3, C4, Target_i(t)]
        - S(t): Noisy mixed signal value (1 value)
        - C1-C4: One-hot frequency selection (4 values)
        - Target_i(t): Clean target for selected frequency (1 value)

        Returns:
            np.ndarray: Dataset of shape (40000, 6)
        """
        # Generate signals
        t_array, S = self.generate_mixed_signal()
        targets = self.generate_clean_targets(t_array)

        # Build dataset
        dataset = []

        for freq_idx in range(4):
            # Create one-hot vector for this frequency
            one_hot = np.zeros(4)
            one_hot[freq_idx] = 1.0

            for t_idx in range(self.n_samples):
                row = np.concatenate([
                    [S[t_idx]],                    # Noisy mixed signal
                    one_hot,                       # Frequency selection (4 values)
                    [targets[freq_idx, t_idx]]     # Clean target
                ])
                dataset.append(row)

        return np.array(dataset, dtype=np.float32)  # Shape: (40000, 6)

    def save_dataset(self, filepath: str) -> None:
        """
        Generate and save dataset to file.

        Args:
            filepath: Path to save .npy file
        """
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Generate and save
        dataset = self.create_dataset()
        np.save(filepath, dataset)

        print(f"Dataset saved to {filepath}")
        print(f"  Shape: {dataset.shape}")
        print(f"  Size: {dataset.nbytes / 1024 / 1024:.2f} MB")
```

### Usage Example

```python
# Generate training dataset (Seed #1)
train_generator = SignalGenerator(
    frequencies=[1, 3, 5, 7],
    fs=1000,
    duration=10.0,
    seed=1
)
train_generator.save_dataset('data/train_data.npy')

# Generate test dataset (Seed #2)
test_generator = SignalGenerator(
    frequencies=[1, 3, 5, 7],
    fs=1000,
    duration=10.0,
    seed=2
)
test_generator.save_dataset('data/test_data.npy')
```

### Critical Implementation Notes

#### 1. Per-Sample Randomization (CRITICAL!)

```python
# WRONG - Same amplitude/phase for all samples
A = np.random.uniform(0.8, 1.2)  # Single value
phi = np.random.uniform(0, 2*np.pi)  # Single value
noisy_signal = A * np.sin(2*np.pi*freq*t_array + phi)  # ❌

# CORRECT - Different amplitude/phase per sample
for i, t in enumerate(t_array):
    A_t = np.random.uniform(0.8, 1.2)        # New value each iteration
    phi_t = np.random.uniform(0, 2*np.pi)    # New value each iteration
    noisy_signal[i] = A_t * np.sin(2*np.pi*freq*t + phi_t)  # ✓
```

#### 2. Dataset Row Structure

```python
# Example row for frequency f2 (3Hz) at time t=0.001 sec
row = [
    0.7932,    # S[1] - noisy mixed signal at t=0.001
    0,         # C1 - not selecting f1
    1,         # C2 - selecting f2 ✓
    0,         # C3 - not selecting f3
    0,         # C4 - not selecting f4
    0.0188     # Target_2[1] - sin(2π·3·0.001) = 0.0188
]
```

---

## Testing Strategy

### Unit Tests

#### Test 1: Parameter Validation
```python
def test_signal_generator_initialization():
    """Test SignalGenerator initializes with correct parameters."""
    gen = SignalGenerator(seed=1)
    assert gen.frequencies == [1, 3, 5, 7]
    assert gen.fs == 1000
    assert gen.duration == 10.0
    assert gen.n_samples == 10000
```

#### Test 2: Time Array Generation
```python
def test_time_array():
    """Test time array has correct shape and range."""
    gen = SignalGenerator(seed=1)
    t = gen.generate_time_array()
    assert t.shape == (10000,)
    assert t[0] == 0.0
    assert abs(t[-1] - 10.0) < 1e-6
```

#### Test 3: Noisy Sinusoid Randomness
```python
def test_noisy_sinusoid_randomness():
    """Test that amplitude/phase vary across samples."""
    gen = SignalGenerator(seed=1)
    t = gen.generate_time_array()

    # Generate two noisy sinusoids with same parameters
    noisy1 = gen.generate_noisy_sinusoid(1.0, t[:100])

    # Reset seed and generate again
    np.random.seed(1)
    noisy2 = gen.generate_noisy_sinusoid(1.0, t[:100])

    # Should be identical with same seed
    np.testing.assert_array_almost_equal(noisy1, noisy2)
```

#### Test 4: Dataset Shape
```python
def test_dataset_shape():
    """Test dataset has correct shape."""
    gen = SignalGenerator(seed=1)
    dataset = gen.create_dataset()
    assert dataset.shape == (40000, 6)
    assert dataset.dtype == np.float32
```

#### Test 5: One-Hot Encoding
```python
def test_one_hot_encoding():
    """Test one-hot vectors are correct."""
    gen = SignalGenerator(seed=1)
    dataset = gen.create_dataset()

    # Check first 10000 rows (frequency f1)
    assert np.all(dataset[:10000, 1:5] == [1, 0, 0, 0])

    # Check next 10000 rows (frequency f2)
    assert np.all(dataset[10000:20000, 1:5] == [0, 1, 0, 0])
```

### Integration Tests

#### Test 6: Seed Reproducibility
```python
def test_seed_reproducibility():
    """Test that same seed produces same dataset."""
    gen1 = SignalGenerator(seed=42)
    data1 = gen1.create_dataset()

    gen2 = SignalGenerator(seed=42)
    data2 = gen2.create_dataset()

    np.testing.assert_array_equal(data1, data2)
```

#### Test 7: Seed Difference
```python
def test_different_seeds_produce_different_data():
    """Test that different seeds produce different noise."""
    gen1 = SignalGenerator(seed=1)
    data1 = gen1.create_dataset()

    gen2 = SignalGenerator(seed=2)
    data2 = gen2.create_dataset()

    # S(t) values should be different (column 0)
    assert not np.allclose(data1[:, 0], data2[:, 0])

    # But targets should be same (column 5) - no randomization
    np.testing.assert_array_almost_equal(data1[:, 5], data2[:, 5])
```

### Validation Tests

#### Test 8: FFT Frequency Verification
```python
def test_fft_frequency_content():
    """Use FFT to verify correct frequencies are present."""
    from scipy import fft

    gen = SignalGenerator(seed=1)
    t, S = gen.generate_mixed_signal()

    # Compute FFT
    fft_vals = np.abs(fft.fft(S))
    freqs = fft.fftfreq(len(S), 1/gen.fs)

    # Check for peaks at 1, 3, 5, 7 Hz
    for expected_freq in [1, 3, 5, 7]:
        idx = np.argmin(np.abs(freqs - expected_freq))
        # Should have significant power at this frequency
        assert fft_vals[idx] > np.mean(fft_vals) * 10
```

#### Test 9: Signal Range Validation
```python
def test_signal_range():
    """Test that signals are in reasonable range."""
    gen = SignalGenerator(seed=1)
    dataset = gen.create_dataset()

    # S(t) should be roughly in [-1, 1] range (sum of 4 normalized sines)
    assert np.all(dataset[:, 0] >= -2.0)
    assert np.all(dataset[:, 0] <= 2.0)

    # Targets should be exactly in [-1, 1]
    assert np.all(dataset[:, 5] >= -1.0)
    assert np.all(dataset[:, 5] <= 1.0)
```

---

## Deliverables

### Code Files
- [ ] `src/data_generation.py` - SignalGenerator class implementation
- [ ] `tests/test_data_generation.py` - Unit and integration tests

### Data Files
- [ ] `data/train_data.npy` - Training dataset (Seed #1, 40,000 × 6)
- [ ] `data/test_data.npy` - Test dataset (Seed #2, 40,000 × 6)

### Documentation
- [ ] Docstrings for all functions and classes
- [ ] README section explaining data generation
- [ ] Example usage notebook (optional)

### Validation Outputs
- [ ] FFT analysis plots showing frequency content
- [ ] Sample visualization: noisy vs. clean signals
- [ ] Statistics report: mean, std, range of signals

---

## Success Criteria

### Correctness
- [ ] Dataset shape is exactly (40,000, 6)
- [ ] FFT shows clear peaks at 1, 3, 5, 7 Hz
- [ ] One-hot encoding is correct for all rows
- [ ] Targets are pure sinusoids (verified visually)
- [ ] Training and test sets have different noise (verified numerically)

### Performance
- [ ] Generation time < 10 seconds per dataset
- [ ] File size ~1.92 MB per dataset
- [ ] No memory issues during generation

### Quality
- [ ] All tests pass (>90% coverage)
- [ ] Code passes linting (flake8, black)
- [ ] Type hints complete and validated (mypy)
- [ ] Docstrings complete and clear

### Reproducibility
- [ ] Same seed produces identical dataset
- [ ] Different seeds produce different noise but same frequencies
- [ ] Random state properly managed

---

## Risks and Mitigation

### Risk 1: Incorrect Randomization Pattern
**Risk**: Using vectorized randomization instead of per-sample

**Impact**: High - Model will fail to learn

**Mitigation**:
- Use explicit loop over samples
- Add unit test verifying different values per sample
- Visual inspection of generated signals

### Risk 2: Seed Not Applied Correctly
**Risk**: Random state not properly controlled

**Impact**: Medium - Non-reproducible results

**Mitigation**:
- Set seed in `__init__`
- Test reproducibility with same seed
- Test difference with different seeds

### Risk 3: Numerical Precision Issues
**Risk**: Float precision causing subtle errors

**Impact**: Low - Should not affect learning

**Mitigation**:
- Use float32 (sufficient precision)
- Validate range of generated values
- Check for NaN/Inf values

### Risk 4: Memory Issues with Large Arrays
**Risk**: 40,000 × 6 array too large

**Impact**: Very Low - Only ~2 MB

**Mitigation**:
- Monitor memory usage during testing
- Use dtype=float32 instead of float64

---

## Dependencies

### Required For
- **Phase 3 (Training Pipeline)**: Needs training data
- **Phase 4 (Evaluation)**: Needs test data

### Depends On
- None (this is the first phase)

---

## Estimated Effort

| Activity | Time Estimate |
|----------|---------------|
| Implement SignalGenerator class | 1-1.5 hours |
| Write unit tests | 0.5-1 hour |
| Validation and FFT analysis | 0.5 hour |
| Documentation and examples | 0.5 hour |
| **Total** | **2.5-3 hours** |

---

## Next Steps

After completing Phase 1:
1. Verify all tests pass
2. Visually inspect generated signals
3. Run FFT validation
4. Proceed to [Phase 2: Model Architecture](02_MODEL_ARCHITECTURE_PRD.md)

---

**Status**: Ready for Implementation
**Last Updated**: 2025-11-16
