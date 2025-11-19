# Phase 2: Model Architecture PRD

**Phase**: 2 of 6
**Priority**: High
**Estimated Effort**: 1-2 hours
**Dependencies**: None (can develop in parallel with Phase 1)
**Enables**: Phase 3 (Training Pipeline)

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

Design and implement the LSTM neural network architecture for conditional frequency extraction.

**IMPORTANT CLARIFICATION**:
- **L=1** refers to `sequence_length=1` (each forward pass processes one time point)
- **L=1 does NOT constrain `num_layers`** (number of stacked LSTM layers in the model)
- The `num_layers` parameter is **fully experimentally tunable** (try 1, 2, 3, 4, etc.)

### What This Phase Delivers
- **FrequencyLSTM model** with proper input/output structure
- **State management** capabilities (critical for L=1)
- **Configurable architecture** (hidden size, num layers)
- **Validated model** tested with dummy data

### Why This Phase is Critical
- The model architecture must support explicit state management
- Input/output dimensions must match data structure exactly
- Foundation for the training pipeline (Phase 3)
- PyTorch implementation enables manual state control

---

## Requirements

### Functional Requirements

#### FR1: Input Structure
- [ ] **Input Dimension**: 5 (S(t) + 4-dimensional one-hot vector C)
- [ ] **Input Shape**: (batch_size, sequence_length, input_size)
  - batch_size: 1 (for L=1)
  - sequence_length: 1 (critical constraint)
  - input_size: 5
- [ ] **Input Type**: torch.FloatTensor

#### FR2: Output Structure
- [ ] **Output Dimension**: 1 (scalar prediction for Target_i(t))
- [ ] **Output Shape**: (batch_size, 1)
- [ ] **Output Type**: torch.FloatTensor
- [ ] **No activation function** (regression task)

#### FR3: LSTM Layer
- [ ] **Configurable hidden size**: Default 128, tunable [32, 64, 128, 256]
- [ ] **Configurable num layers**: Default 1, tunable [1, 2, 3, 4] (experimentally adjust as needed)
- [ ] **State outputs**: Must return (h_n, c_n)
- [ ] **Batch first**: Set batch_first=True for convenience
- [ ] **Dropout**: Only applied between layers if num_layers > 1

#### FR4: State Management
- [ ] **State initialization**: Method to create zero states
- [ ] **State passing**: Accept optional state input
- [ ] **State return**: Always return updated state
- [ ] **State shapes**:
  - h_n: (num_layers, batch_size, hidden_size)
  - c_n: (num_layers, batch_size, hidden_size)

#### FR5: Output Layer
- [ ] **Fully connected layer**: hidden_size → 1
- [ ] **Linear activation**: No sigmoid/tanh (regression)
- [ ] **Bias enabled**: True

### Non-Functional Requirements

#### NFR1: Performance
- Forward pass: < 1ms per sample
- Model size: < 10 MB
- GPU compatible (if available)

#### NFR2: Code Quality
- Clean PyTorch nn.Module subclass
- Type hints for all methods
- Comprehensive docstrings
- Follows PyTorch conventions

#### NFR3: Configurability
- Hyperparameters configurable via constructor
- Easy to experiment with different architectures
- Clear defaults based on problem requirements

---

## Architecture

### Model Overview

```
┌─────────────────────────────────────────────────────────┐
│                    FrequencyLSTM                        │
│                                                         │
│  Input: [S(t), C1, C2, C3, C4]                         │
│         Shape: (batch=1, seq=1, features=5)             │
│                                                         │
│         ↓                                               │
│  ┌──────────────────────────────────────────┐          │
│  │        LSTM Layer                        │          │
│  │                                          │          │
│  │  • input_size: 5                         │          │
│  │  • hidden_size: 64 (default)             │          │
│  │  • num_layers: 1 (default)               │          │
│  │  • batch_first: True                     │          │
│  │                                          │          │
│  │  Maintains:                              │          │
│  │  • Hidden state (h_t): (1, 1, 64)        │          │
│  │  • Cell state (c_t):   (1, 1, 64)        │          │
│  └────────────┬─────────────────────────────┘          │
│               │                                         │
│               ↓                                         │
│  ┌──────────────────────────────────────────┐          │
│  │     Fully Connected Layer                │          │
│  │                                          │          │
│  │  • Input: 64 features                    │          │
│  │  • Output: 1 (regression)                │          │
│  │  • Activation: None (linear)             │          │
│  └────────────┬─────────────────────────────┘          │
│               │                                         │
│               ↓                                         │
│  Output: Target_i(t)                                    │
│         Shape: (batch=1, 1)                             │
│                                                         │
│  State: (h_n, c_n)                                      │
│         Shapes: (1, 1, 64) each                         │
└─────────────────────────────────────────────────────────┘
```

### State Flow Diagram

```
Training Step t:
┌───────────────────────────────────────────────────────┐
│                                                       │
│  Previous State (t-1):                                │
│    h_{t-1}: (1, 1, 64)                               │
│    c_{t-1}: (1, 1, 64)                               │
│                                                       │
│         ↓                                             │
│  ┌──────────────────────────────────┐                │
│  │  LSTM Cell                       │                │
│  │                                  │                │
│  │  Input: x_t (1, 1, 5)           │                │
│  │  State: (h_{t-1}, c_{t-1})      │                │
│  │                                  │                │
│  │  → Computes new state            │                │
│  │  → Outputs hidden representation │                │
│  └──────────┬───────────────────────┘                │
│             │                                         │
│             ↓                                         │
│  New State (t):                                       │
│    h_t: (1, 1, 64)  ───┐                            │
│    c_t: (1, 1, 64)  ───┼─→ Pass to next step (t+1)  │
│                        │   (after detachment!)       │
│             ↓          │                             │
│  ┌──────────────────┐  │                             │
│  │  FC Layer        │  │                             │
│  │  Output: y_t     │  │                             │
│  └──────────────────┘  │                             │
│                        │                             │
└────────────────────────┼─────────────────────────────┘
                         │
                         ↓
                   Next Training Step (t+1)
```

---

## Implementation Details

### Libraries Used

| Library | Purpose |
|---------|---------|
| **torch** | Neural network framework |
| **torch.nn** | Neural network modules |
| **typing** | Type hints |

### Key Class: FrequencyLSTM

```python
import torch
import torch.nn as nn
from typing import Tuple, Optional

class FrequencyLSTM(nn.Module):
    """
    LSTM model for extracting individual frequency components from
    a mixed noisy signal via conditional regression.

    Architecture:
        Input (5) → LSTM (hidden_size) → FC (1) → Output

    The model accepts a selection vector C (one-hot encoded) that
    conditions the extraction on a specific frequency.

    Args:
        input_size (int): Size of input features (default: 5)
            [S(t), C1, C2, C3, C4]
        hidden_size (int): Number of LSTM hidden units (default: 64)
        num_layers (int): Number of LSTM layers (default: 1)
        dropout (float): Dropout probability between LSTM layers
            (default: 0.0, not used for single layer)
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super(FrequencyLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input shape: (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
                For L=1: (1, 1, 5)
            hidden: Optional tuple of (h_0, c_0) LSTM states
                Each of shape (num_layers, batch_size, hidden_size)
                If None, initialized to zeros

        Returns:
            tuple:
                - output: Predicted target of shape (batch_size, 1)
                - (h_n, c_n): Updated LSTM states, each of shape
                    (num_layers, batch_size, hidden_size)
        """
        # LSTM forward pass
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        # h_n, c_n shapes: (num_layers, batch_size, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)

        # Take the last timestep output
        # For L=1, this is just lstm_out[:, 0, :]
        last_output = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)

        # Pass through fully connected layer
        output = self.fc(last_output)  # Shape: (batch_size, 1)

        return output, (h_n, c_n)

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device = torch.device('cpu')
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden and cell states to zeros.

        Args:
            batch_size: Batch size (typically 1 for L=1)
            device: Device to create tensors on (cpu or cuda)

        Returns:
            tuple: (h_0, c_0) initialized to zeros
                Each of shape (num_layers, batch_size, hidden_size)
        """
        h_0 = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=device
        )
        c_0 = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=device
        )
        return (h_0, c_0)

    def count_parameters(self) -> int:
        """
        Count total number of trainable parameters.

        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_summary(self) -> str:
        """
        Get a formatted string summary of model architecture.

        Returns:
            str: Model summary
        """
        summary = []
        summary.append("=" * 60)
        summary.append("FrequencyLSTM Model Summary")
        summary.append("=" * 60)
        summary.append(f"Input size:      {self.input_size}")
        summary.append(f"Hidden size:     {self.hidden_size}")
        summary.append(f"Num layers:      {self.num_layers}")
        summary.append(f"Output size:     1")
        summary.append("-" * 60)
        summary.append(f"Total parameters: {self.count_parameters():,}")
        summary.append("=" * 60)

        return "\n".join(summary)
```

### Usage Example

```python
import torch

# Create model
model = FrequencyLSTM(
    input_size=5,
    hidden_size=64,
    num_layers=1
)

print(model.get_model_summary())

# Dummy input (batch=1, seq=1, features=5)
x = torch.randn(1, 1, 5)

# Initial forward pass (no hidden state)
output, hidden = model(x)
print(f"Output shape: {output.shape}")  # (1, 1)
print(f"h_n shape: {hidden[0].shape}")  # (1, 1, 64)
print(f"c_n shape: {hidden[1].shape}")  # (1, 1, 64)

# Next forward pass (with previous hidden state)
x_next = torch.randn(1, 1, 5)
output_next, hidden_next = model(x_next, hidden)

# State has been updated
assert not torch.equal(hidden[0], hidden_next[0])
```

### Model Hyperparameters

| Hyperparameter | Default | Options | Impact |
|----------------|---------|---------|--------|
| **input_size** | 5 | Fixed | Matches data structure |
| **hidden_size** | 128 | [32, 64, 128, 256] | Capacity vs. speed tradeoff |
| **num_layers** | 1 | [1, 2, 3, 4, ...] | Depth vs. complexity (fully tunable) |
| **dropout** | 0.0 | [0.0, 0.1, 0.2, 0.3] | Regularization (only used if num_layers > 1) |

#### Recommended Starting Point
- **hidden_size**: 128
  - Good capacity for 4-frequency separation
  - Balanced between underfitting and overfitting
- **num_layers**: 1
  - Simpler architecture to start
  - Easier to debug state management
  - **Feel free to experiment with 2, 3, or 4 layers** for potentially better performance

---

## Testing Strategy

### Unit Tests

#### Test 1: Model Initialization
```python
def test_model_initialization():
    """Test model initializes with correct architecture."""
    model = FrequencyLSTM(hidden_size=64, num_layers=1)

    assert model.input_size == 5
    assert model.hidden_size == 64
    assert model.num_layers == 1
    assert isinstance(model.lstm, nn.LSTM)
    assert isinstance(model.fc, nn.Linear)
```

#### Test 2: Forward Pass Shape
```python
def test_forward_pass_shape():
    """Test forward pass produces correct output shape."""
    model = FrequencyLSTM()

    # Input: (batch=1, seq=1, features=5)
    x = torch.randn(1, 1, 5)

    output, (h_n, c_n) = model(x)

    # Check shapes
    assert output.shape == (1, 1)
    assert h_n.shape == (1, 1, 64)
    assert c_n.shape == (1, 1, 64)
```

#### Test 3: State Initialization
```python
def test_state_initialization():
    """Test init_hidden creates correct zero states."""
    model = FrequencyLSTM(hidden_size=64, num_layers=1)

    h_0, c_0 = model.init_hidden(batch_size=1)

    assert h_0.shape == (1, 1, 64)
    assert c_0.shape == (1, 1, 64)
    assert torch.all(h_0 == 0)
    assert torch.all(c_0 == 0)
```

#### Test 4: State Propagation
```python
def test_state_propagation():
    """Test that states are updated across forward passes."""
    model = FrequencyLSTM()
    model.eval()  # Disable dropout if any

    x1 = torch.randn(1, 1, 5)
    x2 = torch.randn(1, 1, 5)

    # First forward pass
    output1, hidden1 = model(x1)

    # Second forward pass with previous state
    output2, hidden2 = model(x2, hidden1)

    # States should be different
    assert not torch.equal(hidden1[0], hidden2[0])
    assert not torch.equal(hidden1[1], hidden2[1])

    # Outputs should be different (different inputs)
    assert not torch.equal(output1, output2)
```

#### Test 5: Gradient Flow
```python
def test_gradient_flow():
    """Test that gradients flow through the network."""
    model = FrequencyLSTM()

    x = torch.randn(1, 1, 5, requires_grad=True)
    target = torch.randn(1, 1)

    output, _ = model(x)
    loss = nn.MSELoss()(output, target)
    loss.backward()

    # Check that LSTM has gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.all(param.grad == 0), f"Zero gradient for {name}"
```

### Integration Tests

#### Test 6: Batch Processing
```python
def test_batch_processing():
    """Test model handles batch size > 1 (even though we use 1)."""
    model = FrequencyLSTM()

    # Batch of 4
    x = torch.randn(4, 1, 5)

    output, (h_n, c_n) = model(x)

    assert output.shape == (4, 1)
    assert h_n.shape == (1, 4, 64)
    assert c_n.shape == (1, 4, 64)
```

#### Test 7: GPU Compatibility
```python
def test_gpu_compatibility():
    """Test model works on GPU if available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")

    model = FrequencyLSTM().cuda()
    x = torch.randn(1, 1, 5).cuda()

    output, (h_n, c_n) = model(x)

    assert output.device.type == 'cuda'
    assert h_n.device.type == 'cuda'
    assert c_n.device.type == 'cuda'
```

### Validation Tests

#### Test 8: Parameter Count
```python
def test_parameter_count():
    """Test parameter count is reasonable."""
    model = FrequencyLSTM(hidden_size=64, num_layers=1)
    param_count = model.count_parameters()

    # LSTM: 4 * (input_size + hidden_size + 1) * hidden_size
    # FC: (hidden_size + 1) * output_size
    expected_lstm = 4 * (5 + 64 + 1) * 64  # ~17,920
    expected_fc = (64 + 1) * 1  # 65
    expected_total = expected_lstm + expected_fc

    assert abs(param_count - expected_total) < 100  # Allow some tolerance
```

#### Test 9: Model Determinism
```python
def test_model_determinism():
    """Test that same input produces same output with same seed."""
    torch.manual_seed(42)
    model1 = FrequencyLSTM()
    x = torch.randn(1, 1, 5)

    output1, _ = model1(x)

    torch.manual_seed(42)
    model2 = FrequencyLSTM()

    output2, _ = model2(x)

    torch.testing.assert_close(output1, output2)
```

---

## Deliverables

### Code Files
- [ ] `src/model.py` - FrequencyLSTM implementation
- [ ] `tests/test_model.py` - Unit and integration tests

### Documentation
- [ ] Class and method docstrings
- [ ] Architecture diagram (in this PRD)
- [ ] Usage examples
- [ ] Hyperparameter tuning guide

### Validation Outputs
- [ ] Model summary printout
- [ ] Parameter count
- [ ] Sample forward pass results

---

## Success Criteria

### Correctness
- [ ] All unit tests pass
- [ ] Forward pass produces correct shapes
- [ ] State propagation works correctly
- [ ] Gradients flow through all parameters

### Performance
- [ ] Forward pass < 1ms per sample
- [ ] Model loads on GPU (if available)
- [ ] Parameter count < 100K

### Quality
- [ ] Code follows PyTorch conventions
- [ ] Type hints complete
- [ ] Docstrings clear and comprehensive
- [ ] No warnings during model creation

---

## Risks and Mitigation

### Risk 1: State Management Complexity
**Risk**: Difficulty properly handling LSTM states

**Impact**: Medium - Critical for Phase 3

**Mitigation**:
- Clear documentation of state flow
- Explicit state return in forward()
- Unit tests for state propagation
- Reference examples

### Risk 2: Shape Mismatches
**Risk**: Input/output shapes don't match data

**Impact**: High - Will break training

**Mitigation**:
- Extensive shape testing
- Clear documentation of expected shapes
- Assertions in forward() if needed

### Risk 3: Over-parameterization
**Risk**: Model too large for simple task

**Impact**: Low - May overfit

**Mitigation**:
- Start with small hidden_size (64)
- Test with different sizes
- Monitor validation performance

---

## Dependencies

### Required For
- **Phase 3 (Training Pipeline)**: Needs model definition

### Depends On
- None (can develop independently)

---

## Estimated Effort

| Activity | Time Estimate |
|----------|---------------|
| Implement FrequencyLSTM class | 0.5-1 hour |
| Write unit tests | 0.5 hour |
| Documentation and examples | 0.5 hour |
| **Total** | **1.5-2 hours** |

---

## Next Steps

After completing Phase 2:
1. Verify all tests pass
2. Test with dummy data from Phase 1
3. Validate state management behavior
4. Proceed to [Phase 3: Training Pipeline](03_TRAINING_PIPELINE_PRD.md)

---

**Status**: Ready for Implementation
**Last Updated**: 2025-11-16
