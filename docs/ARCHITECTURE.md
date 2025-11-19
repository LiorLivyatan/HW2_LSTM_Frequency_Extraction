# Architecture Documentation
## LSTM Frequency Extraction System

**Version**: 2.0
**Last Updated**: November 2025
**Document Owner**: Asif Amar
**Status**: Production Ready

---

## Table of Contents

1. [Overview](#1-overview)
2. [C4 Model Diagrams](#2-c4-model-diagrams)
3. [Architectural Decision Records (ADRs)](#3-architectural-decision-records-adrs)
4. [Component Specifications](#4-component-specifications)
5. [Data Architecture](#5-data-architecture)
6. [API Documentation](#6-api-documentation)
7. [Deployment Architecture](#7-deployment-architecture)
8. [Security Architecture](#8-security-architecture)
9. [Performance Architecture](#9-performance-architecture)
10. [Testing Architecture](#10-testing-architecture)

---

## 1. Overview

### 1.1 System Purpose

The LSTM Frequency Extraction System is a deep learning application that extracts individual frequency components from mixed noisy signals using conditional regression. The system demonstrates advanced LSTM state management through the L=1 sequence length constraint.

### 1.2 Architectural Principles

| Principle | Description | Implementation |
|-----------|-------------|----------------|
| **Modularity** | Separate concerns into distinct modules | 7 source modules |
| **Configurability** | Externalize all parameters | YAML + env vars |
| **Testability** | Design for comprehensive testing | 97% coverage |
| **Reproducibility** | Ensure consistent results | Seed control |
| **Observability** | Enable monitoring and debugging | Comprehensive logging |

### 1.3 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LSTM Frequency Extraction System                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    Presentation Layer                        │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │   │
│   │  │     CLI     │  │   Logging   │  │   Visualization     │  │   │
│   │  │  (main.py)  │  │   Output    │  │   (matplotlib)      │  │   │
│   │  └─────────────┘  └─────────────┘  └─────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                │                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    Application Layer                         │   │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │   │
│   │  │   Training   │  │  Evaluation  │  │  Visualization   │   │   │
│   │  │   Pipeline   │  │    Module    │  │     Module       │   │   │
│   │  └──────────────┘  └──────────────┘  └──────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                │                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                      Core Layer                              │   │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │   │
│   │  │  FrequencyLSTM │  │   Dataset    │  │ SignalGenerator  │   │   │
│   │  │    (model)     │  │   Wrapper    │  │                  │   │   │
│   │  └──────────────┘  └──────────────┘  └──────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                │                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                 Infrastructure Layer                         │   │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │   │
│   │  │   PyTorch    │  │    NumPy     │  │   File System    │   │   │
│   │  │   Runtime    │  │   Arrays     │  │    Storage       │   │   │
│   │  └──────────────┘  └──────────────┘  └──────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. C4 Model Diagrams

### 2.1 Level 1: System Context Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         System Context                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                          ┌─────────────┐                             │
│                          │ Researcher  │                             │
│                          │   (User)    │                             │
│                          └──────┬──────┘                             │
│                                 │                                    │
│                                 │ Uses CLI to                        │
│                                 │ train/evaluate/visualize           │
│                                 ▼                                    │
│           ┌─────────────────────────────────────────┐                │
│           │                                         │                │
│           │     LSTM Frequency Extraction System    │                │
│           │                                         │                │
│           │  Extracts individual frequency          │                │
│           │  components from mixed noisy signals    │                │
│           │  using conditional LSTM regression      │                │
│           │                                         │                │
│           └─────────────────┬───────────────────────┘                │
│                             │                                        │
│                             │ Reads/Writes                           │
│                             ▼                                        │
│                    ┌─────────────────┐                               │
│                    │   File System   │                               │
│                    │                 │                               │
│                    │ - Datasets      │                               │
│                    │ - Model ckpts   │                               │
│                    │ - Results       │                               │
│                    └─────────────────┘                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Level 2: Container Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Container Diagram                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                  CLI Application (main.py)                   │    │
│  │                                                              │    │
│  │  Orchestrates all phases: data, train, eval, viz             │    │
│  │  Python + argparse                                           │    │
│  └───────────────────────────┬──────────────────────────────────┘    │
│                              │                                       │
│              ┌───────────────┼───────────────┐                       │
│              │               │               │                       │
│              ▼               ▼               ▼                       │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐             │
│  │  Data Layer   │  │ Training Layer│  │ Output Layer  │             │
│  │               │  │               │  │               │             │
│  │ - Generation  │  │ - LSTM Model  │  │ - Evaluation  │             │
│  │ - Dataset     │  │ - Trainer     │  │ - Visualizer  │             │
│  │               │  │               │  │               │             │
│  │ Python/NumPy  │  │ Python/PyTorch│  │ Python/Mpl    │             │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘             │
│          │                  │                  │                     │
│          ▼                  ▼                  ▼                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      Data Storage                            │    │
│  │                                                              │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │    │
│  │  │  data/   │  │ models/  │  │ outputs/ │  │  logs/   │     │    │
│  │  │  .npy    │  │  .pth    │  │  .png    │  │  .log    │     │    │
│  │  │  files   │  │  .json   │  │  .json   │  │          │     │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │    │
│  │                                                              │    │
│  │  File System Storage                                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Level 3: Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Component Diagram                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  CLI Container                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                              │    │
│  │  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐ │    │
│  │  │    main.py   │────▶│   Config     │────▶│   Logger     │ │    │
│  │  │ Orchestrator │     │   Loader     │     │   Setup      │ │    │
│  │  └──────┬───────┘     └──────────────┘     └──────────────┘ │    │
│  │         │                                                    │    │
│  │         │ Coordinates                                        │    │
│  │         ▼                                                    │    │
│  │  ┌──────────────────────────────────────────────────────┐   │    │
│  │  │              Phase Execution Engine                   │   │    │
│  │  │                                                       │   │    │
│  │  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐   │   │    │
│  │  │  │Phase1│  │Phase2│  │Phase3│  │Phase4│  │Phase5│   │   │    │
│  │  │  │ Data │──│Model │──│Train │──│ Eval │──│ Viz  │   │   │    │
│  │  │  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘   │   │    │
│  │  └──────────────────────────────────────────────────────┘   │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Core Components                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                              │    │
│  │  ┌─────────────────────┐      ┌─────────────────────┐       │    │
│  │  │  SignalGenerator    │      │  FrequencyDataset   │       │    │
│  │  │                     │      │                     │       │    │
│  │  │ - generate_signals  │      │ - __getitem__       │       │    │
│  │  │ - generate_targets  │      │ - __len__           │       │    │
│  │  │ - create_dataset    │──────│ - load_data         │       │    │
│  │  │ - save_data         │      │                     │       │    │
│  │  └─────────────────────┘      └─────────────────────┘       │    │
│  │                                                              │    │
│  │  ┌─────────────────────┐      ┌─────────────────────┐       │    │
│  │  │  FrequencyLSTM      │      │  StatefulTrainer    │       │    │
│  │  │                     │      │                     │       │    │
│  │  │ - __init__          │      │ - train_epoch       │       │    │
│  │  │ - forward           │◀─────│ - train             │       │    │
│  │  │ - get_summary       │      │ - save_checkpoint   │       │    │
│  │  └─────────────────────┘      │ - state_detach      │       │    │
│  │                               └─────────────────────┘       │    │
│  │                                                              │    │
│  │  ┌─────────────────────┐      ┌─────────────────────┐       │    │
│  │  │  Evaluator          │      │  Visualizer         │       │    │
│  │  │                     │      │                     │       │    │
│  │  │ - evaluate          │      │ - plot_comparison   │       │    │
│  │  │ - get_predictions   │──────│ - plot_all_freqs    │       │    │
│  │  │ - calc_metrics      │      │ - plot_loss_curve   │       │    │
│  │  └─────────────────────┘      │ - plot_analysis     │       │    │
│  │                               └─────────────────────┘       │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.4 Deployment Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Deployment Diagram                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Developer Workstation (macOS/Linux/Windows)                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                              │    │
│  │  Python Runtime (3.8+)                                       │    │
│  │  ┌─────────────────────────────────────────────────────┐    │    │
│  │  │                                                      │    │    │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐     │    │    │
│  │  │  │  PyTorch   │  │   NumPy    │  │ Matplotlib │     │    │    │
│  │  │  │   2.0+     │  │  1.24+     │  │   3.7+     │     │    │    │
│  │  │  └────────────┘  └────────────┘  └────────────┘     │    │    │
│  │  │                                                      │    │    │
│  │  │  ┌────────────────────────────────────────────┐     │    │    │
│  │  │  │     LSTM Frequency Extraction App          │     │    │    │
│  │  │  │                                            │     │    │    │
│  │  │  │  main.py + src/*.py                        │     │    │    │
│  │  │  └────────────────────────────────────────────┘     │    │    │
│  │  │                                                      │    │    │
│  │  └─────────────────────────────────────────────────────┘    │    │
│  │                                                              │    │
│  │  Storage                                                     │    │
│  │  ┌─────────────────────────────────────────────────────┐    │    │
│  │  │  Local Filesystem                                    │    │    │
│  │  │  /HW2_LSTM_Frequency_Extraction/                     │    │    │
│  │  │    ├── data/        (datasets)                       │    │    │
│  │  │    ├── models/      (checkpoints)                    │    │    │
│  │  │    ├── outputs/     (results)                        │    │    │
│  │  │    └── logs/        (training logs)                  │    │    │
│  │  └─────────────────────────────────────────────────────┘    │    │
│  │                                                              │    │
│  │  Optional: GPU Acceleration                                  │    │
│  │  ┌─────────────────────────────────────────────────────┐    │    │
│  │  │  CUDA-enabled GPU (if available)                     │    │    │
│  │  │  - Training acceleration                             │    │    │
│  │  │  - Inference acceleration                            │    │    │
│  │  └─────────────────────────────────────────────────────┘    │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Architectural Decision Records (ADRs)

### ADR-001: PyTorch over TensorFlow/Keras

**Status**: Accepted

**Context**: Need to implement explicit LSTM state management for L=1 constraint.

**Decision**: Use PyTorch as the deep learning framework.

**Rationale**:
- PyTorch provides explicit state control via `output, (h_n, c_n) = lstm(input, (h_0, c_0))`
- Keras abstracts state management, making manual control difficult
- PyTorch's imperative programming model aligns with pedagogical goals
- Better debugging capabilities with eager execution

**Consequences**:
- Positive: Full control over hidden state management
- Positive: Clear gradient computation and detachment
- Negative: Less high-level abstraction (more boilerplate)
- Negative: Requires explicit device management

**Alternatives Considered**:
| Framework | Pros | Cons | Verdict |
|-----------|------|------|---------|
| TensorFlow 2.x | Large ecosystem | State management opaque | Rejected |
| Keras | Simple API | Cannot access internal state | Rejected |
| JAX | Functional, fast | Steep learning curve | Rejected |

---

### ADR-002: Sequence Length L=1 with Manual State

**Status**: Accepted

**Context**: Assignment requires processing one time point at a time to demonstrate LSTM understanding.

**Decision**: Implement L=1 sequence length with explicit state passing between samples.

**Rationale**:
- Pedagogical requirement to understand LSTM internals
- Demonstrates temporal learning without batched sequences
- Forces understanding of state detachment to prevent memory leaks

**Implementation**:
```python
hidden_state = None
for sample in dataloader:
    output, hidden_state = model(input, hidden_state)
    loss.backward()
    hidden_state = tuple(h.detach() for h in hidden_state)  # Critical
```

**Consequences**:
- Positive: Deep understanding of LSTM mechanics
- Positive: Demonstrates temporal learning concept
- Negative: More complex implementation
- Negative: Potential memory issues if not properly detached

---

### ADR-003: Per-Sample Randomization

**Status**: Accepted

**Context**: Need to prevent model from memorizing noise patterns.

**Decision**: Apply random amplitude and phase at each time sample individually.

**Rationale**:
- Forces model to learn frequency structure
- Prevents overfitting to specific noise patterns
- Ensures generalization to unseen noise

**Implementation**:
```python
for i, t in enumerate(t_array):
    A_t = np.random.uniform(0.8, 1.2)      # Per-sample
    phi_t = np.random.uniform(0, 0.02*np.pi)  # Per-sample
    signal[i] = A_t * np.sin(2*np.pi*freq*t + phi_t)
```

**Consequences**:
- Positive: Better generalization
- Positive: More realistic signal processing scenario
- Negative: Slower data generation (no vectorization)
- Negative: More complex data generation code

---

### ADR-004: YAML-Based Configuration

**Status**: Accepted

**Context**: Need flexible hyperparameter management without code changes.

**Decision**: Use YAML files for all configurable parameters with environment variable overrides.

**Rationale**:
- Human-readable format
- Easy to version control
- Supports complex nested structures
- Environment variables allow runtime overrides

**Configuration Hierarchy**:
```
config.yaml (base) < environment variables < CLI arguments
```

**Consequences**:
- Positive: Easy experimentation
- Positive: Reproducible configurations
- Positive: No code changes for hyperparameter tuning
- Negative: Additional dependency (PyYAML)

---

### ADR-005: Modular Architecture

**Status**: Accepted

**Context**: Need maintainable, testable code for academic submission.

**Decision**: Organize code into single-responsibility modules.

**Module Structure**:
```
src/
├── data_generation.py   # SignalGenerator
├── dataset.py           # FrequencyDataset
├── model.py             # FrequencyLSTM
├── training.py          # StatefulTrainer
├── evaluation.py        # Evaluator
├── visualization.py     # Visualizer
└── table_generator.py   # TableGenerator
```

**Consequences**:
- Positive: Easy to test individual components
- Positive: Clear separation of concerns
- Positive: Easier to understand and maintain
- Negative: More files to manage

---

### ADR-006: Test-Driven Quality

**Status**: Accepted

**Context**: Need confidence in correctness for academic submission.

**Decision**: Achieve >90% test coverage with comprehensive test suite.

**Testing Strategy**:
- Unit tests for each module
- Integration tests for phase transitions
- Edge case coverage
- Fixture-based test setup

**Current Achievement**: 97% coverage with 400+ tests

**Consequences**:
- Positive: High confidence in correctness
- Positive: Safe refactoring
- Positive: Living documentation
- Negative: Development time investment

---

### ADR-007: Batch Size Flexibility

**Status**: Accepted

**Context**: Need to balance training speed with memory constraints.

**Decision**: Support configurable batch sizes while maintaining state preservation.

**Implementation**:
- Hidden state shape: `(num_layers, batch_size, hidden_size)`
- Each position in batch tracks independent temporal sequence
- State preserved across batches, detached after backward pass

**Default**: batch_size=32

**Consequences**:
- Positive: Faster training with larger batches
- Positive: Works on machines with limited memory (small batches)
- Negative: More complex state management

---

### ADR-008: Comprehensive Logging

**Status**: Accepted

**Context**: Need visibility into training progress and debugging.

**Decision**: Implement multi-level logging with file and console output.

**Log Levels**:
- INFO: Progress updates, metrics
- DEBUG: Detailed state information
- WARNING: Potential issues
- ERROR: Failures

**Consequences**:
- Positive: Easy debugging
- Positive: Training progress visibility
- Positive: Audit trail for experiments
- Negative: Log file management needed

---

## 4. Component Specifications

### 4.1 SignalGenerator Component

**Location**: `src/data_generation.py`

**Purpose**: Generate synthetic noisy signals and clean targets for training/testing.

**Interface**:
```python
class SignalGenerator:
    def __init__(
        self,
        frequencies: List[float] = [1, 3, 5, 7],
        sampling_rate: int = 1000,
        duration: float = 10.0,
        seed: int = None
    )

    def generate_noisy_signal(self, freq: float) -> np.ndarray:
        """Generate single noisy sinusoid with per-sample randomization."""

    def generate_mixed_signal(self) -> np.ndarray:
        """Generate mixed signal from all frequencies."""

    def generate_target(self, freq: float) -> np.ndarray:
        """Generate clean target sinusoid."""

    def create_dataset(self) -> np.ndarray:
        """Create complete dataset with shape (40000, 6)."""

    def save_data(self, data: np.ndarray, filepath: str) -> None:
        """Save dataset to .npy file."""
```

**Data Contract**:
- Output shape: `(40000, 6)`
- Columns: `[S(t), C1, C2, C3, C4, Target]`
- Row organization: 10000 samples per frequency

---

### 4.2 FrequencyDataset Component

**Location**: `src/dataset.py`

**Purpose**: PyTorch Dataset wrapper for training data.

**Interface**:
```python
class FrequencyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str)

    def __len__(self) -> int:
        """Return number of samples (40000)."""

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (input_features, target) for sample idx.

        Returns:
            input: (5,) tensor [S(t), C1, C2, C3, C4]
            target: scalar tensor
        """

    def get_frequency_data(self, freq_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get all data for a specific frequency."""
```

---

### 4.3 FrequencyLSTM Component

**Location**: `src/model.py`

**Purpose**: LSTM neural network for frequency extraction.

**Interface**:
```python
class FrequencyLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0
    )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with explicit state management.

        Args:
            x: Input tensor (batch, seq_len=1, features=5)
            hidden: Optional (h_0, c_0) from previous sample

        Returns:
            output: Prediction (batch, 1)
            hidden: (h_n, c_n) for next sample
        """

    def get_model_summary(self) -> Dict[str, Any]:
        """Return model architecture summary."""
```

**State Shapes**:
- h_n: `(num_layers, batch_size, hidden_size)`
- c_n: `(num_layers, batch_size, hidden_size)`

---

### 4.4 StatefulTrainer Component

**Location**: `src/training.py`

**Purpose**: Training loop with L=1 state preservation pattern.

**Interface**:
```python
class StatefulTrainer:
    def __init__(
        self,
        model: FrequencyLSTM,
        config: Dict[str, Any],
        device: torch.device
    )

    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int
    ) -> Dict[str, List[float]]:
        """
        Train model with state preservation.

        Returns:
            history: Dict with 'train_loss' list
        """

    def train_epoch(
        self,
        dataloader: DataLoader
    ) -> float:
        """
        Single epoch with L=1 state management.

        CRITICAL: Implements state detachment pattern.
        """

    def save_checkpoint(self, filepath: str) -> None:
        """Save model state dict."""

    def load_checkpoint(self, filepath: str) -> None:
        """Load model state dict."""
```

**Critical Implementation** (train_epoch):
```python
def train_epoch(self, dataloader):
    hidden_state = None  # Initialize once per epoch

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Add sequence dimension
        inputs = inputs.unsqueeze(1)  # (batch, 1, 5)

        # Forward pass
        outputs, hidden_state = self.model(inputs, hidden_state)

        # Compute loss
        loss = self.criterion(outputs.squeeze(), targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.clip_grad_norm
        )

        self.optimizer.step()

        # CRITICAL: Detach state from computation graph
        hidden_state = tuple(h.detach() for h in hidden_state)
```

---

### 4.5 Evaluator Component

**Location**: `src/evaluation.py`

**Purpose**: Calculate performance metrics and store predictions.

**Interface**:
```python
class Evaluator:
    def __init__(
        self,
        model: FrequencyLSTM,
        device: torch.device
    )

    def evaluate(
        self,
        dataloader: DataLoader
    ) -> Tuple[float, np.ndarray]:
        """
        Evaluate model with state preservation.

        Returns:
            mse: Mean squared error
            predictions: All predictions (40000,)
        """

    def calculate_metrics(
        self,
        train_mse: float,
        test_mse: float,
        train_preds: np.ndarray,
        test_preds: np.ndarray,
        train_data: np.ndarray,
        test_data: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate comprehensive metrics including per-frequency."""

    def save_metrics(self, metrics: Dict, filepath: str) -> None:
        """Save metrics to JSON file."""

    def save_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        filepath: str
    ) -> None:
        """Save predictions to .npz file."""
```

---

### 4.6 Visualizer Component

**Location**: `src/visualization.py`

**Purpose**: Generate all visualization graphs.

**Interface**:
```python
class Visualizer:
    def __init__(self, config: Dict[str, Any])

    def plot_frequency_comparison(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        mixed_signal: np.ndarray,
        freq_idx: int = 1,  # 3Hz
        time_window: Tuple[int, int] = (0, 1000),
        save_path: str = None
    ) -> None:
        """
        Graph 1: Single frequency comparison.
        Shows target, prediction, and noisy input.
        """

    def plot_all_frequencies(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        time_window: Tuple[int, int] = (0, 1000),
        save_path: str = None
    ) -> None:
        """
        Graph 2: All 4 frequencies in 2x2 grid.
        """

    def plot_training_loss(
        self,
        history: Dict[str, List[float]],
        save_path: str = None
    ) -> None:
        """Plot training loss curve."""

    def plot_fft_analysis(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        save_path: str = None
    ) -> None:
        """FFT comparison of predictions vs targets."""

    def plot_error_distributions(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        save_path: str = None
    ) -> None:
        """Error histogram per frequency."""
```

---

## 5. Data Architecture

### 5.1 Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                        Data Flow Diagram                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────┐     ┌─────────────────┐                     │
│  │ SignalGenerator │────▶│  train_data.npy │                     │
│  │   (Seed #1)     │     │   (40000, 6)    │                     │
│  └─────────────────┘     └────────┬────────┘                     │
│                                   │                               │
│  ┌─────────────────┐              │     ┌─────────────────┐      │
│  │ SignalGenerator │──────────────┤     │ FrequencyDataset│      │
│  │   (Seed #2)     │              │     │                 │      │
│  └─────────────────┘              │     └────────┬────────┘      │
│          │                        │              │                │
│          ▼                        │              ▼                │
│  ┌─────────────────┐              │     ┌─────────────────┐      │
│  │  test_data.npy  │              │     │   DataLoader    │      │
│  │   (40000, 6)    │              │     │  (batch=32)     │      │
│  └────────┬────────┘              │     └────────┬────────┘      │
│           │                       │              │                │
│           │                       │              ▼                │
│           │                       │     ┌─────────────────┐      │
│           │                       └────▶│ StatefulTrainer │      │
│           │                             │                 │      │
│           │                             └────────┬────────┘      │
│           │                                      │                │
│           │                                      ▼                │
│           │                             ┌─────────────────┐      │
│           │                             │   best_model    │      │
│           │                             │     .pth        │      │
│           │                             └────────┬────────┘      │
│           │                                      │                │
│           ▼                                      ▼                │
│  ┌─────────────────┐                    ┌─────────────────┐      │
│  │   Evaluator     │◀───────────────────│   Evaluator     │      │
│  │  (test data)    │                    │  (train data)   │      │
│  └────────┬────────┘                    └────────┬────────┘      │
│           │                                      │                │
│           └──────────────┬───────────────────────┘                │
│                          │                                        │
│                          ▼                                        │
│                 ┌─────────────────┐                               │
│                 │  metrics.json   │                               │
│                 │ predictions.npz │                               │
│                 └────────┬────────┘                               │
│                          │                                        │
│                          ▼                                        │
│                 ┌─────────────────┐                               │
│                 │   Visualizer    │                               │
│                 │                 │                               │
│                 └────────┬────────┘                               │
│                          │                                        │
│                          ▼                                        │
│                 ┌─────────────────┐                               │
│                 │   graphs/*.png  │                               │
│                 └─────────────────┘                               │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 Data Schemas

#### Dataset Schema (.npy)

```
Shape: (40000, 6)

Columns:
┌─────┬────┬────┬────┬────┬────────┐
│ Col │  0 │  1 │  2 │  3 │   4    │   5    │
├─────┼────┼────┼────┼────┼────────┤────────┤
│Name │S(t)│ C1 │ C2 │ C3 │   C4   │ Target │
│Type │f32 │f32 │f32 │f32 │  f32   │  f32   │
│Range│±2  │0/1 │0/1 │0/1 │  0/1   │  ±1    │
└─────┴────┴────┴────┴────┴────────┴────────┘

Row Organization:
- Rows 0-9999:      f₁ = 1Hz (C=[1,0,0,0])
- Rows 10000-19999: f₂ = 3Hz (C=[0,1,0,0])
- Rows 20000-29999: f₃ = 5Hz (C=[0,0,1,0])
- Rows 30000-39999: f₄ = 7Hz (C=[0,0,0,1])
```

#### Model Checkpoint Schema (.pth)

```python
{
    'model_state_dict': OrderedDict,  # LSTM + FC weights
    'optimizer_state_dict': OrderedDict,  # Adam state
    'epoch': int,  # Checkpoint epoch
    'loss': float,  # Best loss value
    'config': dict  # Model configuration
}
```

#### Metrics Schema (.json)

```json
{
    "mse_train": 0.0993,
    "mse_test": 0.0994,
    "generalization_gap_absolute": 0.0001,
    "generalization_gap_percent": 0.13,
    "generalization_status": "PASS",
    "per_frequency": {
        "1Hz": {"mse_train": 0.00983, "mse_test": 0.01030},
        "3Hz": {"mse_train": 0.2226, "mse_test": 0.2237},
        "5Hz": {"mse_train": 0.0636, "mse_test": 0.0624},
        "7Hz": {"mse_train": 0.1011, "mse_test": 0.1013}
    },
    "timestamp": "2025-11-19T10:30:00"
}
```

#### Predictions Schema (.npz)

```python
{
    'train_predictions': np.ndarray,  # (40000,)
    'test_predictions': np.ndarray,   # (40000,)
    'train_targets': np.ndarray,      # (40000,)
    'test_targets': np.ndarray        # (40000,)
}
```

#### Training History Schema (.json)

```json
{
    "train_loss": [0.5, 0.3, 0.2, ...],  // Per-epoch loss
    "epochs": [1, 2, 3, ...],
    "best_epoch": 100,
    "best_loss": 0.0993,
    "training_time_seconds": 3000
}
```

### 5.3 Storage Layout

```
HW2_LSTM_Frequency_Extraction/
│
├── data/                           # Raw datasets
│   ├── train_data.npy             # 2.4 MB
│   └── test_data.npy              # 2.4 MB
│
├── models/                         # Model artifacts
│   ├── best_model.pth             # ~2 MB
│   └── training_history.json      # ~10 KB
│
├── outputs/                        # Results
│   ├── metrics.json               # ~2 KB
│   ├── predictions.npz            # ~2 MB
│   ├── run_config.yaml            # ~1 KB
│   │
│   ├── graphs/                    # Visualizations
│   │   ├── frequency_comparison.png    # ~500 KB
│   │   ├── all_frequencies.png         # ~800 KB
│   │   ├── training_loss_curve.png     # ~200 KB
│   │   └── ... (7 more graphs)
│   │
│   └── tables/                    # Markdown reports
│       ├── performance_summary.md
│       └── per_frequency_metrics.md
│
└── logs/                          # Training logs
    └── lstm_extraction.log        # Variable size
```

---

## 6. API Documentation

### 6.1 Public Module APIs

#### data_generation.py

```python
# Main entry point
def generate_and_save_datasets(config: Dict) -> Tuple[str, str]:
    """Generate and save train/test datasets.

    Args:
        config: Configuration dict with 'data' section

    Returns:
        Tuple of (train_path, test_path)
    """
```

#### model.py

```python
# Model factory
def create_model(config: Dict, device: torch.device) -> FrequencyLSTM:
    """Create model from configuration.

    Args:
        config: Configuration dict with 'model' section
        device: Target device (CPU/GPU)

    Returns:
        Initialized FrequencyLSTM model
    """
```

#### training.py

```python
# Training entry point
def train_model(
    model: FrequencyLSTM,
    train_path: str,
    config: Dict,
    device: torch.device
) -> Dict[str, List[float]]:
    """Train model on dataset.

    Args:
        model: FrequencyLSTM model
        train_path: Path to training data
        config: Configuration dict
        device: Target device

    Returns:
        Training history dict
    """
```

#### evaluation.py

```python
# Evaluation entry point
def evaluate_model(
    model: FrequencyLSTM,
    train_path: str,
    test_path: str,
    config: Dict,
    device: torch.device
) -> Dict[str, Any]:
    """Evaluate model on train and test sets.

    Args:
        model: Trained FrequencyLSTM model
        train_path: Path to training data
        test_path: Path to test data
        config: Configuration dict
        device: Target device

    Returns:
        Comprehensive metrics dict
    """
```

#### visualization.py

```python
# Visualization entry point
def generate_visualizations(
    metrics_path: str,
    predictions_path: str,
    history_path: str,
    data_paths: Dict[str, str],
    output_dir: str,
    config: Dict
) -> List[str]:
    """Generate all visualization graphs.

    Args:
        metrics_path: Path to metrics.json
        predictions_path: Path to predictions.npz
        history_path: Path to training_history.json
        data_paths: Dict with 'train' and 'test' paths
        output_dir: Output directory for graphs
        config: Configuration dict

    Returns:
        List of generated graph paths
    """
```

### 6.2 Configuration API

```yaml
# config.yaml structure

# Model hyperparameters
model:
  input_size: 5        # Fixed: S(t) + 4D one-hot
  hidden_size: 128     # Tunable: 64-512
  num_layers: 1        # Tunable: 1-4
  dropout: 0.0         # Tunable: 0.0-0.5

# Training hyperparameters
training:
  learning_rate: 0.0001  # Tunable: 1e-5 to 1e-2
  num_epochs: 100        # Tunable: 50-200
  batch_size: 32         # Tunable: 1-128
  clip_grad_norm: 1.0    # Tunable: 0.5-5.0
  device: "auto"         # Options: "auto", "cpu", "cuda"

# Data generation parameters
data:
  frequencies: [1, 3, 5, 7]  # Hz
  sampling_rate: 1000        # Samples/second
  duration: 10.0             # Seconds
  train_seed: 1              # RNG seed for training
  test_seed: 2               # RNG seed for testing
  amplitude_range: [0.8, 1.2]  # Per-sample amplitude
  phase_range: [0, 0.0628]     # Per-sample phase (0.02π)

# File paths
paths:
  data_dir: "data"
  model_dir: "models"
  output_dir: "outputs"
  log_dir: "logs"

# Visualization settings
visualization:
  dpi: 300
  figsize: [12, 8]
  time_window: [0, 1000]  # Samples to show
  freq_for_comparison: 1  # Index (0-3) for Graph 1
```

### 6.3 CLI Interface

```bash
# Main entry point
python main.py [OPTIONS]

Options:
  --mode {all,data,train,eval,viz}
      Execution mode (default: all)

  --config PATH
      Configuration file path (default: config.yaml)

  --verbose
      Enable debug logging

  --help
      Show help message

# Environment variable overrides
LSTM_LEARNING_RATE=0.001 python main.py
LSTM_HIDDEN_SIZE=256 python main.py
LSTM_NUM_EPOCHS=200 python main.py
```

---

## 7. Deployment Architecture

### 7.1 Local Deployment

```bash
# Prerequisites
- Python 3.8+
- pip

# Installation
git clone <repository>
cd HW2_LSTM_Frequency_Extraction
pip install -r requirements.txt

# Execution
python main.py --mode all
```

### 7.2 Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install torch numpy matplotlib pyyaml pytest pytest-cov

# Verify installation
python -c "import torch; print(torch.__version__)"
```

### 7.3 GPU Support

```python
# Automatic GPU detection
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Manual override
device = torch.device("cpu")  # Force CPU
```

---

## 8. Security Architecture

### 8.1 Security Principles

| Principle | Implementation |
|-----------|----------------|
| No hardcoded secrets | All config externalized |
| Input validation | All inputs checked |
| Safe file operations | Path validation |
| No network access | Self-contained system |

### 8.2 Input Validation

```python
# Configuration validation
def validate_config(config: Dict) -> None:
    assert config['model']['input_size'] == 5
    assert config['model']['hidden_size'] > 0
    assert config['model']['num_layers'] > 0
    assert 0.0 <= config['model']['dropout'] < 1.0
    assert config['training']['learning_rate'] > 0
    assert config['training']['num_epochs'] > 0
    assert config['training']['batch_size'] > 0
```

### 8.3 File Path Security

```python
# Path validation
def validate_path(path: str, must_exist: bool = True) -> str:
    path = os.path.abspath(path)
    if must_exist and not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")
    return path
```

---

## 9. Performance Architecture

### 9.1 Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Training time/epoch | < 60s | ~30s |
| Inference time/sample | < 1ms | < 0.1ms |
| Memory usage | < 8GB | < 2GB |
| Model size | < 100MB | ~2MB |

### 9.2 Optimization Strategies

#### Training Optimization

```python
# Gradient accumulation alternative for memory
accumulation_steps = 4
for i, (input, target) in enumerate(dataloader):
    output = model(input)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### Inference Optimization

```python
# Disable gradient computation
with torch.no_grad():
    output, hidden = model(input, hidden)
```

#### Data Loading Optimization

```python
# Efficient data loading
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,  # Required for state preservation
    num_workers=0,  # Avoid multiprocessing issues
    pin_memory=True if device.type == 'cuda' else False
)
```

### 9.3 Memory Management

```python
# Prevent memory leaks
hidden_state = tuple(h.detach() for h in hidden_state)

# Clear cache periodically (GPU)
if device.type == 'cuda':
    torch.cuda.empty_cache()
```

---

## 10. Testing Architecture

### 10.1 Test Organization

```
tests/
├── conftest.py              # Shared fixtures
├── test_data_generation.py  # Unit tests for SignalGenerator
├── test_dataset.py          # Unit tests for FrequencyDataset
├── test_model.py            # Unit tests for FrequencyLSTM
├── test_training.py         # Unit tests for StatefulTrainer
├── test_evaluation.py       # Unit tests for Evaluator
├── test_visualization.py    # Unit tests for Visualizer
├── test_main.py             # Integration tests
└── test_table_generator.py  # Unit tests for TableGenerator
```

### 10.2 Test Categories

| Category | Purpose | Examples |
|----------|---------|----------|
| Unit | Test individual components | State detachment, output shapes |
| Integration | Test component interactions | Training pipeline, evaluation |
| Edge case | Test boundary conditions | Empty data, extreme values |
| Regression | Prevent bug recurrence | Known issues |

### 10.3 Key Test Cases

#### State Management Tests

```python
def test_state_detachment():
    """Verify hidden state is detached after backward pass."""
    trainer = StatefulTrainer(model, config, device)

    # Run one training step
    hidden_before = trainer._hidden_state
    trainer.train_step(input, target)
    hidden_after = trainer._hidden_state

    # Verify detachment
    assert not hidden_after[0].requires_grad
    assert not hidden_after[1].requires_grad

def test_state_preservation():
    """Verify hidden state is preserved between samples."""
    trainer = StatefulTrainer(model, config, device)

    # Run multiple steps
    trainer.train_step(input1, target1)
    hidden_after_1 = trainer._hidden_state

    trainer.train_step(input2, target2)
    hidden_after_2 = trainer._hidden_state

    # State should have changed (learning happened)
    assert not torch.equal(hidden_after_1[0], hidden_after_2[0])
```

#### Data Generation Tests

```python
def test_per_sample_randomization():
    """Verify amplitude varies at each time point."""
    generator = SignalGenerator(seed=42)
    signal = generator.generate_noisy_signal(1.0)

    # Extract amplitudes (peaks)
    peaks = np.abs(signal[signal > 0.7])

    # Should have variation
    assert peaks.std() > 0.01

def test_reproducibility():
    """Verify same seed produces same data."""
    gen1 = SignalGenerator(seed=42)
    gen2 = SignalGenerator(seed=42)

    data1 = gen1.create_dataset()
    data2 = gen2.create_dataset()

    np.testing.assert_array_equal(data1, data2)
```

### 10.4 Coverage Requirements

| Module | Minimum Coverage | Current |
|--------|------------------|---------|
| data_generation.py | 100% | 100% |
| dataset.py | 100% | 100% |
| model.py | 100% | 100% |
| training.py | 95% | 98% |
| evaluation.py | 100% | 100% |
| visualization.py | 90% | 94% |

### 10.5 Test Execution

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_training.py -v

# Run specific test
pytest tests/test_training.py::test_state_detachment -v

# Run with warnings
pytest tests/ -v --tb=short -W default
```

---

## Appendix A: Sequence Diagrams

### Training Sequence

```
┌─────┐          ┌──────────┐        ┌───────┐        ┌─────────┐
│User │          │StatefulTrainer│    │ Model │        │DataLoader│
└──┬──┘          └─────┬──────┘      └───┬───┘        └────┬────┘
   │                   │                 │                  │
   │ train(epochs)     │                 │                  │
   │──────────────────▶│                 │                  │
   │                   │                 │                  │
   │                   │ for epoch in epochs:              │
   │                   │─┐               │                  │
   │                   │ │ hidden=None   │                  │
   │                   │◀┘               │                  │
   │                   │                 │                  │
   │                   │ for batch in dataloader:          │
   │                   │────────────────────────────────────▶│
   │                   │                 │    (inputs, targets)
   │                   │◀────────────────────────────────────│
   │                   │                 │                  │
   │                   │ forward(x, h)   │                  │
   │                   │────────────────▶│                  │
   │                   │    (output, h)  │                  │
   │                   │◀────────────────│                  │
   │                   │                 │                  │
   │                   │ loss.backward() │                  │
   │                   │─┐               │                  │
   │                   │ │               │                  │
   │                   │◀┘               │                  │
   │                   │                 │                  │
   │                   │ h = detach(h)   │                  │
   │                   │─┐               │                  │
   │                   │ │ CRITICAL      │                  │
   │                   │◀┘               │                  │
   │                   │                 │                  │
   │  history          │                 │                  │
   │◀──────────────────│                 │                  │
   │                   │                 │                  │
```

### Evaluation Sequence

```
┌─────┐          ┌─────────┐        ┌───────┐        ┌─────────┐
│User │          │Evaluator│        │ Model │        │DataLoader│
└──┬──┘          └────┬────┘        └───┬───┘        └────┬────┘
   │                  │                 │                  │
   │ evaluate()       │                 │                  │
   │─────────────────▶│                 │                  │
   │                  │                 │                  │
   │                  │ model.eval()    │                  │
   │                  │────────────────▶│                  │
   │                  │                 │                  │
   │                  │ with no_grad(): │                  │
   │                  │─┐               │                  │
   │                  │ │ hidden=None   │                  │
   │                  │◀┘               │                  │
   │                  │                 │                  │
   │                  │ for batch:      │                  │
   │                  │────────────────────────────────────▶│
   │                  │◀────────────────────────────────────│
   │                  │                 │                  │
   │                  │ forward(x, h)   │                  │
   │                  │────────────────▶│                  │
   │                  │◀────────────────│                  │
   │                  │                 │                  │
   │                  │ detach(h)       │                  │
   │                  │─┐               │                  │
   │                  │◀┘               │                  │
   │                  │                 │                  │
   │ (mse, preds)     │                 │                  │
   │◀─────────────────│                 │                  │
   │                  │                 │                  │
```

---

## Appendix B: State Transition Diagram

```
┌──────────────────────────────────────────────────────────────┐
│              LSTM Hidden State Lifecycle                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│   ┌─────────────┐                                             │
│   │    None     │ ◀──── Epoch Start                           │
│   └──────┬──────┘                                             │
│          │                                                    │
│          │ First sample                                       │
│          ▼                                                    │
│   ┌─────────────┐                                             │
│   │  Initialized │                                            │
│   │  (zeros)     │                                            │
│   └──────┬──────┘                                             │
│          │                                                    │
│          │ Forward pass                                       │
│          ▼                                                    │
│   ┌─────────────┐                                             │
│   │  Updated     │ ◀───┐                                      │
│   │ (has grads)  │     │                                      │
│   └──────┬──────┘     │                                      │
│          │             │                                      │
│          │ Backward    │                                      │
│          ▼             │                                      │
│   ┌─────────────┐      │                                      │
│   │  Detached    │     │                                      │
│   │ (no grads)   │─────┘ Next sample                          │
│   └──────┬──────┘                                             │
│          │                                                    │
│          │ Epoch End                                          │
│          ▼                                                    │
│   ┌─────────────┐                                             │
│   │    None     │ ──── Reset for next epoch                   │
│   └─────────────┘                                             │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Appendix C: Error Handling

### Error Categories

| Category | Examples | Handling |
|----------|----------|----------|
| Configuration | Invalid parameters | Validate and raise |
| File I/O | Missing files | Clear error messages |
| Memory | OOM errors | Reduce batch size |
| Numerical | NaN/Inf | Gradient clipping |

### Error Messages

```python
# Configuration errors
raise ValueError(f"Invalid hidden_size: {hidden_size}. Must be positive.")
raise ValueError(f"Invalid dropout: {dropout}. Must be in [0, 1).")

# File errors
raise FileNotFoundError(f"Data file not found: {path}")
raise IOError(f"Cannot write to directory: {output_dir}")

# Training errors
raise RuntimeError("Training diverged: loss is NaN")
raise RuntimeError("Gradient explosion detected")
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Nov 2025 | Developer | Initial architecture |
| 2.0 | Nov 2025 | Developer | Production-ready version |

---

**End of Architecture Documentation**
