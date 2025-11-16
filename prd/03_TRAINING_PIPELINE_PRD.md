# Phase 3: Training Pipeline PRD

**Phase**: 3 of 6
**Priority**: CRITICAL (Most Complex Phase)
**Estimated Effort**: 4-6 hours
**Dependencies**: Phase 1 (Data), Phase 2 (Model)
**Enables**: Phase 4 (Evaluation), Phase 5 (Visualization)

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

Implement the training pipeline with **proper LSTM state management for L=1**.

### What This Phase Delivers
- **StatefulTrainer** class with correct state preservation
- **Custom training loop** handling L=1 constraint
- **Model checkpointing** to save best model
- **Training metrics tracking** (loss curves)
- **Validated convergence** on training data

### Why This Phase is CRITICAL
- **State preservation is THE KEY** to making L=1 work
- Without correct implementation, the model cannot learn temporal patterns
- This is the pedagogical focus of the entire assignment
- Most complex phase requiring deep understanding of LSTM internals

---

## Requirements

### Functional Requirements

#### FR1: Data Loading
- [ ] Load data from Phase 1 (train_data.npy)
- [ ] Create PyTorch Dataset wrapper
- [ ] Create DataLoader with:
  - batch_size = 1 (CRITICAL for L=1)
  - shuffle = False (preserve temporal order)
  - num_workers = 0 (avoid multiprocessing complications)

#### FR2: State Management (CRITICAL!)
- [ ] **Initialize state ONCE** at epoch start (or None)
- [ ] **Preserve state** across all 40,000 samples in epoch
- [ ] **Detach state** from computation graph after each backward pass
- [ ] **Never reset state** between consecutive samples
- [ ] **Reset state** at start of each new epoch

#### FR3: Training Loop
- [ ] Standard supervised learning loop
- [ ] Loss function: MSE (Mean Squared Error)
- [ ] Optimizer: Adam (recommended)
- [ ] Gradient clipping: Optional but recommended
- [ ] Progress tracking: Loss per epoch

#### FR4: Model Checkpointing
- [ ] Save best model based on training loss
- [ ] Save model state_dict
- [ ] Save optimizer state_dict (for resuming)
- [ ] Save training history

#### FR5: Configuration
- [ ] Configurable hyperparameters:
  - Learning rate
  - Number of epochs
  - Hidden size
  - Gradient clip value
- [ ] Device selection (CPU/GPU)

### Non-Functional Requirements

#### NFR1: Performance
- Training time: < 5 minutes per epoch (on CPU)
- Memory usage: Stable (no leaks from state accumulation)
- GPU utilization: > 80% if using GPU

#### NFR2: Robustness
- Handle NaN/Inf in loss
- Graceful handling of keyboard interrupt
- Resumable training from checkpoint

#### NFR3: Monitoring
- Real-time loss display
- Progress bar (tqdm)
- Memory usage tracking
- State statistics logging

---

## Architecture

### Training Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  StatefulTrainer                            │
│                                                             │
│  Components:                                                │
│  • model: FrequencyLSTM                                     │
│  • train_loader: DataLoader (batch_size=1, shuffle=False)  │
│  • criterion: MSELoss                                       │
│  • optimizer: Adam                                          │
│  • device: cpu or cuda                                      │
└─────────────────────────────────────────────────────────────┘
```

### Critical State Management Flow

```
EPOCH START
    │
    ├─→ hidden_state = None  (Initialize once)
    │
    ├─→ FOR sample_idx in range(40,000):
    │       │
    │       ├─→ Get batch: (input, target) from DataLoader
    │       │   • input shape: (1, 1, 5)
    │       │   • target shape: (1, 1)
    │       │
    │       ├─→ Forward pass:
    │       │   output, hidden_state = model(input, hidden_state)
    │       │   • hidden_state contains (h_t, c_t)
    │       │   • State flows from previous sample
    │       │
    │       ├─→ Compute loss:
    │       │   loss = MSE(output, target)
    │       │
    │       ├─→ Backward pass:
    │       │   optimizer.zero_grad()
    │       │   loss.backward()
    │       │   [optional: clip_grad_norm]
    │       │   optimizer.step()
    │       │
    │       └─→ CRITICAL: Detach state!
    │           hidden_state = tuple(h.detach() for h in hidden_state)
    │           • Prevents backprop through entire history
    │           • Avoids memory explosion
    │           • Keeps state values but breaks gradient connection
    │
    └─→ EPOCH END
        • Log average loss
        • Save checkpoint if best
        • (State discarded, will reinitialize next epoch)
```

### Memory Management Pattern

```
Without Detachment (WRONG):
────────────────────────────
Sample 1: x₁ → LSTM → y₁ ──┐
                            │
Sample 2: x₂ → LSTM → y₂ ──┤
                            ├─→ Computation graph spans ALL samples
Sample 3: x₃ → LSTM → y₃ ──┤   Memory: O(n) where n = 40,000
                            │   EXPLOSION!
...                         │
                            │
Sample 40000: x₄₀₀₀₀ → LSTM → y₄₀₀₀₀ ──┘


With Detachment (CORRECT):
──────────────────────────────
Sample 1: x₁ → LSTM → y₁
          ↓ (detach)
Sample 2: x₂ → LSTM → y₂  ← Computation graph ONLY this sample
          ↓ (detach)        Memory: O(1) - constant!
Sample 3: x₃ → LSTM → y₃
          ↓ (detach)
...

State values preserved, but gradient connections severed!
```

---

## Implementation Details

### Libraries Used

| Library | Purpose |
|---------|---------|
| **torch** | PyTorch framework |
| **torch.nn** | Loss functions |
| **torch.optim** | Optimizers |
| **torch.utils.data** | Dataset and DataLoader |
| **tqdm** | Progress bars |
| **pathlib** | File path handling |
| **json** | Save training history |

### Dataset Wrapper

```python
import torch
from torch.utils.data import Dataset
import numpy as np

class FrequencyDataset(Dataset):
    """
    PyTorch Dataset wrapper for frequency extraction data.

    Data structure: Each row is [S(t), C1, C2, C3, C4, Target]
    - Input: [S(t), C1, C2, C3, C4] (5 values)
    - Target: [Target] (1 value)

    Args:
        data_path (str): Path to .npy file (40,000 × 6 array)
    """

    def __init__(self, data_path: str):
        # Load data
        self.data = np.load(data_path).astype(np.float32)

        # Split into inputs and targets
        self.inputs = self.data[:, :5]   # [S(t), C1, C2, C3, C4]
        self.targets = self.data[:, 5:6]  # [Target] - keep 2D

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Get a single sample.

        Returns:
            tuple: (input, target)
                - input: tensor of shape (5,)
                - target: tensor of shape (1,)
        """
        input_vec = torch.from_numpy(self.inputs[idx])    # (5,)
        target_val = torch.from_numpy(self.targets[idx])  # (1,)

        return input_vec, target_val
```

### StatefulTrainer Class

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import json
from typing import Dict, List, Optional

class StatefulTrainer:
    """
    Trainer for FrequencyLSTM with proper state management for L=1.

    This trainer implements the critical state preservation pattern:
    - State initialized once per epoch
    - State preserved across all samples within epoch
    - State detached after each backward pass to prevent memory explosion

    Args:
        model: FrequencyLSTM model
        train_loader: DataLoader with batch_size=1, shuffle=False
        criterion: Loss function (typically MSELoss)
        optimizer: Optimizer (typically Adam)
        device: Device to train on ('cpu' or 'cuda')
        clip_grad_norm: Maximum gradient norm (None for no clipping)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device = torch.device('cpu'),
        clip_grad_norm: Optional[float] = 1.0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.clip_grad_norm = clip_grad_norm

        self.history = {
            'train_loss': [],
            'epoch_times': []
        }

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch with state preservation.

        This is the CRITICAL function implementing L=1 with state management.

        Args:
            epoch: Current epoch number (for logging)

        Returns:
            float: Average loss for the epoch
        """
        self.model.train()

        # CRITICAL: Initialize state once at epoch start
        hidden_state = None  # Will be initialized to zeros by LSTM

        total_loss = 0.0
        num_samples = 0

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            leave=True
        )

        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move to device
            inputs = inputs.to(self.device)    # (batch=1, features=5)
            targets = targets.to(self.device)  # (batch=1, 1)

            # Reshape for LSTM: (batch, seq_len, features)
            inputs = inputs.unsqueeze(1)  # (1, 1, 5)

            # ============================================================
            # CRITICAL SECTION: State-preserving forward pass
            # ============================================================

            # Forward pass with previous state
            output, hidden_state = self.model(inputs, hidden_state)

            # Compute loss
            loss = self.criterion(output, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Optional gradient clipping
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.clip_grad_norm
                )

            self.optimizer.step()

            # ============================================================
            # CRITICAL: Detach state from computation graph
            # ============================================================
            # This prevents backpropagation through entire sequence history
            # State values are preserved, but gradient connections are severed
            hidden_state = tuple(h.detach() for h in hidden_state)

            # ============================================================
            # END CRITICAL SECTION
            # ============================================================

            # Track loss
            total_loss += loss.item()
            num_samples += 1

            # Update progress bar
            avg_loss = total_loss / num_samples
            pbar.set_postfix({'loss': f'{avg_loss:.6f}'})

        # Calculate epoch average loss
        epoch_loss = total_loss / num_samples

        return epoch_loss

    def train(
        self,
        num_epochs: int,
        save_dir: str = 'models',
        save_best: bool = True
    ) -> Dict:
        """
        Train for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            save_best: Whether to save best model

        Returns:
            dict: Training history
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        best_loss = float('inf')

        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Total samples per epoch: {len(self.train_loader)}")

        for epoch in range(1, num_epochs + 1):
            # Train one epoch
            epoch_loss = self.train_epoch(epoch)

            # Update history
            self.history['train_loss'].append(epoch_loss)

            print(f"\nEpoch {epoch}/{num_epochs} - Loss: {epoch_loss:.6f}")

            # Save best model
            if save_best and epoch_loss < best_loss:
                best_loss = epoch_loss
                self.save_checkpoint(
                    save_dir=save_dir,
                    filename='best_model.pth',
                    epoch=epoch,
                    loss=epoch_loss
                )
                print(f"  → Saved best model (loss: {best_loss:.6f})")

        print("\nTraining complete!")
        print(f"Best loss: {best_loss:.6f}")

        return self.history

    def save_checkpoint(
        self,
        save_dir: str,
        filename: str = 'checkpoint.pth',
        epoch: int = 0,
        loss: float = 0.0
    ) -> None:
        """
        Save model checkpoint.

        Args:
            save_dir: Directory to save checkpoint
            filename: Filename for checkpoint
            epoch: Current epoch
            loss: Current loss
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'history': self.history
        }

        save_path = Path(save_dir) / filename
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            dict: Checkpoint information
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)

        return checkpoint

    def save_history(self, save_path: str) -> None:
        """Save training history to JSON."""
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
```

### Training Script Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.model import FrequencyLSTM
from src.dataset import FrequencyDataset
from src.training import StatefulTrainer

# Configuration
CONFIG = {
    'data_path': 'data/train_data.npy',
    'hidden_size': 64,
    'num_layers': 1,
    'learning_rate': 0.001,
    'num_epochs': 50,
    'batch_size': 1,  # CRITICAL: Must be 1 for L=1
    'clip_grad_norm': 1.0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def main():
    # Load dataset
    train_dataset = FrequencyDataset(CONFIG['data_path'])

    # Create DataLoader
    # CRITICAL: batch_size=1, shuffle=False
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,  # Preserve temporal order!
        num_workers=0   # Avoid multiprocessing complications
    )

    # Create model
    model = FrequencyLSTM(
        input_size=5,
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers']
    )

    print(model.get_model_summary())

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    # Create trainer
    trainer = StatefulTrainer(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device(CONFIG['device']),
        clip_grad_norm=CONFIG['clip_grad_norm']
    )

    # Train
    history = trainer.train(
        num_epochs=CONFIG['num_epochs'],
        save_dir='models',
        save_best=True
    )

    # Save history
    trainer.save_history('outputs/training_history.json')

    print("\nTraining complete!")

if __name__ == '__main__':
    main()
```

---

## Testing Strategy

### Unit Tests

#### Test 1: Dataset Loading
```python
def test_dataset_loading():
    """Test dataset loads correctly."""
    dataset = FrequencyDataset('data/train_data.npy')

    assert len(dataset) == 40000

    input_vec, target_val = dataset[0]
    assert input_vec.shape == (5,)
    assert target_val.shape == (1,)
```

#### Test 2: DataLoader Configuration
```python
def test_dataloader_config():
    """Test DataLoader has correct configuration for L=1."""
    dataset = FrequencyDataset('data/train_data.npy')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Check first batch
    inputs, targets = next(iter(loader))
    assert inputs.shape == (1, 5)
    assert targets.shape == (1, 1)
```

#### Test 3: State Detachment
```python
def test_state_detachment():
    """Test that state detachment works correctly."""
    h = torch.randn(1, 1, 64, requires_grad=True)
    c = torch.randn(1, 1, 64, requires_grad=True)

    hidden = (h, c)

    # Detach
    hidden_detached = tuple(h.detach() for h in hidden)

    # Original should require grad
    assert hidden[0].requires_grad
    assert hidden[1].requires_grad

    # Detached should NOT require grad
    assert not hidden_detached[0].requires_grad
    assert not hidden_detached[1].requires_grad

    # Values should be same
    torch.testing.assert_close(hidden[0], hidden_detached[0])
    torch.testing.assert_close(hidden[1], hidden_detached[1])
```

### Integration Tests

#### Test 4: Single Epoch Training
```python
def test_single_epoch_training():
    """Test that one epoch of training works."""
    # Small subset for testing
    dataset = FrequencyDataset('data/train_data.npy')
    small_dataset = torch.utils.data.Subset(dataset, range(1000))

    loader = DataLoader(small_dataset, batch_size=1, shuffle=False)

    model = FrequencyLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainer = StatefulTrainer(model, loader, criterion, optimizer)

    initial_loss = trainer.train_epoch(epoch=1)

    assert initial_loss > 0
    assert not np.isnan(initial_loss)
    assert not np.isinf(initial_loss)
```

#### Test 5: Loss Decreases Over Epochs
```python
def test_loss_decreases():
    """Test that loss decreases over multiple epochs."""
    dataset = FrequencyDataset('data/train_data.npy')
    small_dataset = torch.utils.data.Subset(dataset, range(5000))

    loader = DataLoader(small_dataset, batch_size=1, shuffle=False)

    model = FrequencyLSTM(hidden_size=32)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainer = StatefulTrainer(model, loader, criterion, optimizer)

    history = trainer.train(num_epochs=5, save_best=False)

    # Loss should generally decrease
    first_loss = history['train_loss'][0]
    last_loss = history['train_loss'][-1]

    assert last_loss < first_loss
```

### Validation Tests

#### Test 6: Memory Stability
```python
def test_memory_stability():
    """Test that memory doesn't explode during training."""
    import psutil
    import os

    process = psutil.Process(os.getpid())

    dataset = FrequencyDataset('data/train_data.npy')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = FrequencyLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    trainer = StatefulTrainer(model, loader, criterion, optimizer)

    # Measure memory before
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    # Train one epoch
    trainer.train_epoch(epoch=1)

    # Measure memory after
    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    # Memory increase should be reasonable (< 500 MB)
    mem_increase = mem_after - mem_before
    assert mem_increase < 500, f"Memory increased by {mem_increase:.2f} MB"
```

---

## Deliverables

### Code Files
- [ ] `src/dataset.py` - FrequencyDataset class
- [ ] `src/training.py` - StatefulTrainer class
- [ ] `train.py` - Training script
- [ ] `tests/test_training.py` - Unit and integration tests

### Model Outputs
- [ ] `models/best_model.pth` - Trained model checkpoint
- [ ] `outputs/training_history.json` - Loss curves

### Documentation
- [ ] Training configuration guide
- [ ] State management explanation
- [ ] Troubleshooting guide

---

## Success Criteria

### Correctness
- [ ] Loss decreases over epochs
- [ ] No NaN/Inf in loss
- [ ] State preserved correctly (verified by monitoring state values)
- [ ] Model converges (loss < 0.01 ideally)

### Performance
- [ ] Training time < 5 min/epoch on CPU
- [ ] Memory usage stable (no leaks)
- [ ] No warnings or errors

### Quality
- [ ] All tests pass
- [ ] Code clean and well-documented
- [ ] Checkpoint saving/loading works
- [ ] Training resumable from checkpoint

---

## Risks and Mitigation

### Risk 1: Memory Explosion from State Accumulation
**Risk**: Forgetting to detach state causes memory explosion

**Impact**: CRITICAL - Training will crash

**Mitigation**:
- Mandatory state detachment after each step
- Unit test for detachment
- Memory monitoring during training
- Clear documentation with examples

### Risk 2: Incorrect State Reset Logic
**Risk**: Resetting state between samples within epoch

**Impact**: HIGH - Model cannot learn temporal patterns

**Mitigation**:
- Clear documentation of when to reset
- Visual inspection of state values
- Test that state changes across samples

### Risk 3: Gradient Explosion
**Risk**: Unstable gradients in LSTM

**Impact**: MEDIUM - Training instability

**Mitigation**:
- Gradient clipping (norm = 1.0)
- Monitor gradient statistics
- Adjust learning rate if needed

### Risk 4: Poor Convergence
**Risk**: Model doesn't learn effectively

**Impact**: MEDIUM - Poor results

**Mitigation**:
- Try different learning rates
- Adjust hidden size
- Ensure data quality (Phase 1)
- Verify model architecture (Phase 2)

---

## Dependencies

### Required For
- **Phase 4 (Evaluation)**: Needs trained model
- **Phase 5 (Visualization)**: Needs predictions

### Depends On
- **Phase 1 (Data)**: Needs training data
- **Phase 2 (Model)**: Needs model architecture

---

## Estimated Effort

| Activity | Time Estimate |
|----------|---------------|
| Implement FrequencyDataset | 0.5 hour |
| Implement StatefulTrainer | 2-3 hours |
| Write training script | 0.5 hour |
| Testing and debugging | 1-2 hours |
| Documentation | 0.5 hour |
| **Total** | **4.5-6.5 hours** |

---

## Next Steps

After completing Phase 3:
1. Verify training converges
2. Check memory stability
3. Validate state management
4. Proceed to [Phase 4: Evaluation](04_EVALUATION_PRD.md)

---

**Status**: Ready for Implementation
**Last Updated**: 2025-11-16
