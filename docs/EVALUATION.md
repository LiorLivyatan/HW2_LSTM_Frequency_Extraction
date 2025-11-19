# Phase 4: Evaluation PRD

**Phase**: 4 of 6
**Priority**: High
**Estimated Effort**: 1-2 hours
**Dependencies**: Phase 1 (Data), Phase 3 (Trained Model)
**Enables**: Phase 5 (Visualization), Assignment Completion

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

Calculate performance metrics and verify generalization of the trained LSTM model.

### What This Phase Delivers
- **MSE calculations** for both training and test sets
- **Generalization analysis** (MSE_test vs MSE_train)
- **Per-frequency metrics** showing extraction quality for each frequency
- **Predictions saved** for visualization phase
- **Comprehensive metrics report** in JSON format

### Why This Phase is Critical
- Validates that the model learned successfully
- **Proves generalization** (model learned frequencies, not noise)
- Provides quantitative evidence for assignment success
- Required for assignment grading

---

## Requirements

### Functional Requirements

#### FR1: MSE Calculation on Training Set
- [ ] Load trained model from Phase 3
- [ ] Load training data (Seed #1) from Phase 1
- [ ] Generate predictions with state preservation
- [ ] Calculate MSE: `MSE_train = (1/40000) Σ(prediction - target)²`

#### FR2: MSE Calculation on Test Set
- [ ] Load test data (Seed #2) from Phase 1
- [ ] Generate predictions with state preservation
- [ ] Calculate MSE: `MSE_test = (1/40000) Σ(prediction - target)²`

#### FR3: Generalization Metrics
- [ ] Calculate absolute difference: `|MSE_test - MSE_train|`
- [ ] Calculate relative difference: `|MSE_test - MSE_train| / MSE_train`
- [ ] Flag: Is generalization good? (ratio < 0.1)

#### FR4: Per-Frequency Analysis
- [ ] Calculate MSE for each frequency separately (f₁, f₂, f₃, f₄)
- [ ] Identify if any frequency is harder to extract
- [ ] Both for train and test sets

#### FR5: Prediction Storage
- [ ] Save predictions for visualization
- [ ] Store: time, true target, prediction, frequency index
- [ ] Save as `.npy` or `.npz` for easy loading

### Non-Functional Requirements

#### NFR1: Performance
- Evaluation time: < 2 minutes total (both sets)
- Memory efficient (don't store all intermediate results)

#### NFR2: Reproducibility
- Deterministic evaluation (same model → same metrics)
- Clear logging of all metrics

#### NFR3: Reporting
- Human-readable metrics summary
- JSON format for programmatic access
- Clear pass/fail criteria

---

## Architecture

### Evaluation Pipeline

```
┌──────────────────────────────────────────────────────┐
│                 Evaluator Class                      │
│                                                      │
│  Components:                                         │
│  • model: Trained FrequencyLSTM                      │
│  • test_loader: DataLoader for test set             │
│  • train_loader: DataLoader for train set           │
│  • device: cpu or cuda                               │
│                                                      │
│  Methods:                                            │
│  • evaluate_dataset() → MSE, predictions             │
│  • calculate_per_frequency_metrics()                 │
│  • check_generalization()                            │
│  • save_metrics()                                    │
│  • save_predictions()                                │
└──────────────────────────────────────────────────────┘
```

### Evaluation Flow

```
LOAD MODEL
    ↓
EVALUATE ON TRAINING SET
    ├─→ Initialize hidden state = None
    ├─→ For each sample:
    │   ├─→ Forward pass (with state preservation)
    │   ├─→ Store prediction and target
    │   └─→ Detach state
    ├─→ Calculate MSE_train
    └─→ Calculate per-frequency MSE_train
    ↓
EVALUATE ON TEST SET
    ├─→ Initialize hidden state = None
    ├─→ For each sample:
    │   ├─→ Forward pass (with state preservation)
    │   ├─→ Store prediction and target
    │   └─→ Detach state
    ├─→ Calculate MSE_test
    └─→ Calculate per-frequency MSE_test
    ↓
GENERALIZATION ANALYSIS
    ├─→ Compare MSE_test vs MSE_train
    ├─→ Calculate relative difference
    └─→ Determine if generalization is good
    ↓
SAVE RESULTS
    ├─→ Save metrics to JSON
    ├─→ Save predictions for visualization
    └─→ Print summary report
```

---

## Implementation Details

### Libraries Used

| Library | Purpose |
|---------|---------|
| **torch** | Model inference |
| **numpy** | Metric calculations |
| **json** | Save metrics |
| **tqdm** | Progress tracking |
| **pathlib** | File handling |

### Evaluator Class

```python
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import json
from typing import Dict, Tuple, List

class Evaluator:
    """
    Evaluator for FrequencyLSTM with proper state management.

    Calculates MSE metrics and verifies generalization.

    Args:
        model: Trained FrequencyLSTM model
        device: Device for inference ('cpu' or 'cuda')
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device('cpu')
    ):
        self.model = model.to(device)
        self.model.eval()  # Set to evaluation mode
        self.device = device

    @torch.no_grad()  # Disable gradient computation for efficiency
    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        dataset_name: str = 'dataset'
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate model on a dataset with state preservation.

        Args:
            dataloader: DataLoader (batch_size=1, shuffle=False)
            dataset_name: Name for progress bar

        Returns:
            tuple:
                - mse: Mean squared error
                - predictions: Array of predictions (N,)
                - targets: Array of ground truth (N,)
        """
        # Storage
        all_predictions = []
        all_targets = []

        # Initialize state
        hidden_state = None

        # Progress bar
        pbar = tqdm(dataloader, desc=f"Evaluating {dataset_name}")

        for inputs, targets in pbar:
            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Reshape for LSTM: (batch, seq_len, features)
            inputs = inputs.unsqueeze(1)  # (1, 1, 5)

            # Forward pass with state preservation
            output, hidden_state = self.model(inputs, hidden_state)

            # Detach state (same as training)
            hidden_state = tuple(h.detach() for h in hidden_state)

            # Store results
            all_predictions.append(output.cpu().numpy().flatten())
            all_targets.append(targets.cpu().numpy().flatten())

        # Convert to numpy arrays
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)

        # Calculate MSE
        mse = np.mean((predictions - targets) ** 2)

        return mse, predictions, targets

    def calculate_per_frequency_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        dataset_size: int = 10000
    ) -> Dict[int, float]:
        """
        Calculate MSE for each frequency separately.

        Data structure: 40,000 samples = 4 frequencies × 10,000 samples
        - Rows 0-9,999: frequency f1
        - Rows 10,000-19,999: frequency f2
        - Rows 20,000-29,999: frequency f3
        - Rows 30,000-39,999: frequency f4

        Args:
            predictions: Predictions array (40,000,)
            targets: Targets array (40,000,)
            dataset_size: Samples per frequency (10,000)

        Returns:
            dict: {frequency_idx: mse} for each of 4 frequencies
        """
        per_freq_mse = {}

        for freq_idx in range(4):
            start_idx = freq_idx * dataset_size
            end_idx = (freq_idx + 1) * dataset_size

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

        Args:
            mse_train: MSE on training set
            mse_test: MSE on test set
            threshold: Maximum acceptable relative difference (default: 0.1 = 10%)

        Returns:
            dict: Generalization analysis
        """
        abs_diff = abs(mse_test - mse_train)
        rel_diff = abs_diff / mse_train if mse_train > 0 else float('inf')

        generalizes_well = rel_diff < threshold

        return {
            'mse_train': float(mse_train),
            'mse_test': float(mse_test),
            'absolute_difference': float(abs_diff),
            'relative_difference': float(rel_diff),
            'threshold': threshold,
            'generalizes_well': generalizes_well
        }

    def evaluate_all(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader
    ) -> Dict:
        """
        Comprehensive evaluation on both train and test sets.

        Args:
            train_loader: DataLoader for training set
            test_loader: DataLoader for test set

        Returns:
            dict: Complete metrics
        """
        print("=" * 60)
        print("Evaluation Report")
        print("=" * 60)

        # Evaluate training set
        print("\n1. Evaluating Training Set...")
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
        """Print human-readable summary of results."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        # Overall metrics
        print("\nOverall Performance:")
        print(f"  MSE (Training):  {results['overall']['mse_train']:.6f}")
        print(f"  MSE (Test):      {results['overall']['mse_test']:.6f}")

        # Per-frequency
        print("\nPer-Frequency MSE (Training):")
        frequencies = [1, 3, 5, 7]
        for idx, freq in enumerate(frequencies):
            mse = results['per_frequency']['train'][idx]
            print(f"  f{idx+1} = {freq}Hz:  {mse:.6f}")

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

        print("=" * 60)

    def save_metrics(self, results: Dict, save_path: str) -> None:
        """
        Save metrics to JSON file.

        Args:
            results: Evaluation results dictionary
            save_path: Path to save JSON file
        """
        # Remove predictions from saved metrics (too large)
        save_results = {
            'overall': results['overall'],
            'per_frequency': results['per_frequency'],
            'generalization': results['generalization']
        }

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(save_results, f, indent=2)

        print(f"\nMetrics saved to: {save_path}")

    def save_predictions(
        self,
        results: Dict,
        save_path: str
    ) -> None:
        """
        Save predictions and targets to numpy file.

        Args:
            results: Evaluation results with predictions
            save_path: Path to save .npz file
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            save_path,
            train_predictions=results['predictions']['train']['predictions'],
            train_targets=results['predictions']['train']['targets'],
            test_predictions=results['predictions']['test']['predictions'],
            test_targets=results['predictions']['test']['targets']
        )

        print(f"Predictions saved to: {save_path}")
```

### Evaluation Script Example

```python
import torch
from torch.utils.data import DataLoader

from src.model import FrequencyLSTM
from src.dataset import FrequencyDataset
from src.evaluation import Evaluator

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = FrequencyLSTM(hidden_size=64, num_layers=1)

    checkpoint = torch.load('models/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['loss']:.6f}")

    # Load datasets
    train_dataset = FrequencyDataset('data/train_data.npy')
    test_dataset = FrequencyDataset('data/test_data.npy')

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Create evaluator
    evaluator = Evaluator(model, device=device)

    # Evaluate
    results = evaluator.evaluate_all(train_loader, test_loader)

    # Save results
    evaluator.save_metrics(results, 'outputs/metrics.json')
    evaluator.save_predictions(results, 'outputs/predictions.npz')

    print("\nEvaluation complete!")

if __name__ == '__main__':
    main()
```

---

## Testing Strategy

### Unit Tests

#### Test 1: MSE Calculation
```python
def test_mse_calculation():
    """Test MSE calculation is correct."""
    predictions = np.array([1.0, 2.0, 3.0])
    targets = np.array([1.1, 2.2, 3.3])

    mse = np.mean((predictions - targets) ** 2)
    expected = ((0.1)**2 + (0.2)**2 + (0.3)**2) / 3

    np.testing.assert_almost_equal(mse, expected)
```

#### Test 2: Per-Frequency Split
```python
def test_per_frequency_split():
    """Test per-frequency metrics split data correctly."""
    predictions = np.random.randn(40000)
    targets = np.random.randn(40000)

    evaluator = Evaluator(model=None)
    per_freq_mse = evaluator.calculate_per_frequency_metrics(
        predictions,
        targets
    )

    assert len(per_freq_mse) == 4
    assert all(freq_idx in per_freq_mse for freq_idx in range(4))
```

### Integration Tests

#### Test 3: Full Evaluation Pipeline
```python
def test_full_evaluation():
    """Test complete evaluation pipeline."""
    # Use dummy model and data
    model = FrequencyLSTM(hidden_size=32)
    model.eval()

    dataset = FrequencyDataset('data/test_data.npy')
    # Small subset for testing
    small_dataset = torch.utils.data.Subset(dataset, range(1000))
    loader = DataLoader(small_dataset, batch_size=1, shuffle=False)

    evaluator = Evaluator(model)
    mse, predictions, targets = evaluator.evaluate_dataset(loader)

    assert mse >= 0
    assert len(predictions) == 1000
    assert len(targets) == 1000
```

---

## Deliverables

### Code Files
- [ ] `src/evaluation.py` - Evaluator class
- [ ] `evaluate.py` - Evaluation script
- [ ] `tests/test_evaluation.py` - Unit tests

### Output Files
- [ ] `outputs/metrics.json` - Numerical metrics
- [ ] `outputs/predictions.npz` - Saved predictions

### Documentation
- [ ] Metrics interpretation guide
- [ ] Success criteria explanation

---

## Success Criteria

### Required Metrics
- [ ] MSE_train calculated correctly
- [ ] MSE_test calculated correctly
- [ ] Per-frequency metrics calculated
- [ ] Generalization check performed

### Performance Targets
- [ ] MSE_train < 0.01 (ideally < 0.001)
- [ ] MSE_test ≈ MSE_train (within 10%)
- [ ] All 4 frequencies have similar MSE (no outliers)

### Quality
- [ ] All tests pass
- [ ] Metrics saved correctly
- [ ] Clear, readable output
- [ ] Predictions match targets visually (verified in Phase 5)

---

## Risks and Mitigation

### Risk 1: Poor MSE Values
**Risk**: MSE too high, indicating poor learning

**Impact**: HIGH - Assignment failure

**Mitigation**:
- Review Phase 3 training
- Check model architecture
- Verify data quality
- Try different hyperparameters

### Risk 2: Poor Generalization
**Risk**: MSE_test >> MSE_train (overfitting)

**Impact**: MEDIUM - Fails generalization requirement

**Mitigation**:
- Verify different seeds used correctly
- Check if model memorizing noise
- Review random generation in Phase 1

---

## Dependencies

### Required For
- **Phase 5 (Visualization)**: Needs predictions

### Depends On
- **Phase 1 (Data)**: Needs test dataset
- **Phase 3 (Training)**: Needs trained model

---

## Estimated Effort

| Activity | Time Estimate |
|----------|---------------|
| Implement Evaluator class | 0.5-1 hour |
| Write evaluation script | 0.25 hour |
| Testing | 0.25 hour |
| Documentation | 0.25 hour |
| **Total** | **1.25-1.75 hours** |

---

## Next Steps

After completing Phase 4:
1. Verify MSE values are acceptable
2. Confirm generalization is good
3. Check predictions are reasonable
4. Proceed to [Phase 5: Visualization](05_VISUALIZATION_PRD.md)

---

**Status**: Ready for Implementation
**Last Updated**: 2025-11-16
