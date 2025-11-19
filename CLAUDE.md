# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

This is an M.Sc. assignment for developing an LSTM system to extract individual frequency components from mixed noisy signals. The task is **conditional regression**: given a noisy mixed signal S(t) containing 4 frequencies (1Hz, 3Hz, 5Hz, 7Hz) and a one-hot selection vector C, the LSTM must output the clean sinusoid for the selected frequency.

### How to Use This Repository

**IMPORTANT: This codebase follows a documentation-driven development approach.**

The `docs/` folder contains comprehensive documentation:
- **`docs/PRD.md`** - Main Product Requirements Document with goals, metrics, and acceptance criteria
- **`docs/ARCHITECTURE.md`** - System architecture with C4 diagrams and ADRs
- **`docs/DATA_GENERATION.md`** through **`docs/INTEGRATION.md`** - Implementation guides
- **`docs/RESEARCH_AND_EXPERIMENTS.md`** - Experimental results and research notes

**Before implementing any phase, read the corresponding guide:**
- Starting? Read `docs/ARCHITECTURE.md` for system architecture overview
- See `tests/` folder for comprehensive test suite (97% coverage)

This CLAUDE.md provides quick reference for critical concepts, but the implementation guides contain the complete details.

---

## Critical Architecture Concepts

### The L=1 State Preservation Pattern (MOST CRITICAL)

This assignment has a **pedagogical constraint**: **Sequence Length L=1**, meaning each sample is processed individually (one time point per forward pass).

**IMPORTANT**: L=1 refers to `sequence_length=1`, NOT `num_layers=1`. The number of stacked LSTM layers (`num_layers`) is **experimentally tunable** (try 1, 2, 3, etc.).

**The Pedagogical "Trick":**

With L=1, PyTorch's LSTM would **normally reset the hidden state** between each sample, destroying temporal continuity. The assignment's challenge is to **manually preserve the state** across all 10,000 samples, creating an "effective temporal window" through explicit state management rather than batched sequences.

**Correct Implementation Pattern:**
```python
# In training loop (src/training.py)
hidden_state = None  # Initialize ONCE per epoch

for sample in dataloader:
    # Forward pass with previous state (MANUAL state passing)
    output, hidden_state = model(input, hidden_state)

    # Backward pass
    loss.backward()
    optimizer.step()

    # CRITICAL: Detach state from computation graph
    # This preserves state VALUES but breaks gradient connections
    # Prevents memory explosion from 40,000-step backprop chain
    hidden_state = tuple(h.detach() for h in hidden_state)
```

**Why this matters:**
- **PyTorch default behavior at L=1**: Would reset state between samples (no temporal learning)
- **Manual state preservation**: Creates effective window of 10,000 consecutive samples
- **State detachment**: Prevents memory explosion from 40,000-step backprop chain
- **Pedagogical goal**: Understanding LSTM internal state by manually managing it
- This is THE KEY to making L=1 work and is the assignment's core challenge

### Per-Sample Randomization (CRITICAL)

Amplitude and phase must vary randomly at EVERY sample t, not per signal:

**WRONG:**
```python
A = np.random.uniform(0.8, 1.2)  # Single value
noisy = A * np.sin(2*np.pi*freq*t_array)  # ❌
```

**CORRECT:**
```python
for i, t in enumerate(t_array):
    A_t = np.random.uniform(0.8, 1.2)      # New value each iteration
    phi_t = np.random.uniform(0, 2*np.pi)  # New value each iteration
    noisy[i] = A_t * np.sin(2*np.pi*freq*t + phi_t)  # ✓
```

This forces the network to learn frequency structure, not memorize noise patterns.

### Data Structure

- **40,000 rows total**: 4 frequencies × 10,000 time samples
- **Row format**: `[S(t), C1, C2, C3, C4, Target_i(t)]` (6 values)
  - S(t): Noisy mixed signal (1 scalar)
  - C1-C4: One-hot frequency selector (4 values)
  - Target_i(t): Clean target sinusoid (1 scalar)
- **Training set**: Seed #1
- **Test set**: Seed #2 (different noise, same frequencies)

Rows 0-9,999: frequency f₁ (1Hz)
Rows 10,000-19,999: frequency f₂ (3Hz)
Rows 20,000-29,999: frequency f₃ (5Hz)
Rows 30,000-39,999: frequency f₄ (7Hz)

---

## Development Workflow

### Phase-Based Development with PRD Guidance

**IMPORTANT**: The project is organized into 6 phases. Each phase has a detailed guide in the `docs/` folder.

**When implementing each phase, ALWAYS read the corresponding guide first.**

1. **Phase 1: Data Generation** - Read `docs/DATA_GENERATION.md`
2. **Phase 2: Model Architecture** - Read `docs/MODEL_ARCHITECTURE.md`
3. **Phase 3: Training Pipeline** ⭐ **CRITICAL** - Read `docs/TRAINING_PIPELINE.md`
4. **Phase 4: Evaluation** - Read `docs/EVALUATION.md`
5. **Phase 5: Visualization** - Read `docs/VISUALIZATION.md`
6. **Phase 6: Integration** - Read `docs/INTEGRATION.md`

**Architecture Overview**: Before starting, read `docs/ARCHITECTURE.md` for complete system architecture.

### Recommended Development Order

```bash
# 1. Generate and validate data
python -c "from src.data_generation import SignalGenerator; ..."

# 2. Test model architecture
python -c "from src.model import FrequencyLSTM; ..."

# 3. Train model (when Phases 1-2 complete)
python main.py --mode train

# 4. Evaluate
python main.py --mode eval

# 5. Visualize
python main.py --mode viz

# 6. Full pipeline
python main.py --mode all
```

---

## Technology Stack Rationale

- **PyTorch** (NOT TensorFlow/Keras): Explicit LSTM state control required for L=1
  - `output, (h_n, c_n) = lstm(input, (h_0, c_0))` enables manual state passing
  - Keras abstracts state management too much for this assignment

- **NumPy**: Signal generation and data arrays
  - Efficient vectorized operations
  - Built-in seed control for reproducibility

- **Matplotlib**: Scientific visualization (no need for Plotly/Seaborn)

- **No Pandas**: Unnecessary overhead for this task

---

## Key Implementation Details

### DataLoader Configuration
```python
# IMPORTANT: shuffle=False to preserve temporal order
train_loader = DataLoader(
    dataset,
    batch_size=32,     # Can be any size (1, 32, 64, etc.)
    shuffle=False,     # CRITICAL: Preserve temporal order
    num_workers=0      # Avoid multiprocessing complications
)
```

**Batch Size Notes:**
- `batch_size=1`: Sequential processing (simplest, but slower)
- `batch_size=32`: 32 parallel sequences (faster training)
- Hidden state shape adapts: `(num_layers, batch_size, hidden_size)`
- Each batch position tracks its own temporal sequence
- Variable batch sizes (e.g., last batch) handled automatically

### Model Input/Output Shapes
- **Input**: `(batch, seq_len=1, features=5)` - e.g., (32, 1, 5) for batch_size=32
- **Output**: `(batch, 1)` - e.g., (32, 1) for batch_size=32
- **Hidden state**: `(num_layers, batch, hidden_size)` - e.g., (1, 32, 128) for single-layer with batch_size=32

### Evaluation Requirements
- MSE_train: Performance on Seed #1 data
- MSE_test: Performance on Seed #2 data
- **Success**: MSE_test ≈ MSE_train (within 10%)
- Target: MSE < 0.01 (ideally < 0.001)

### Visualization Requirements
- **Graph 1**: Use frequency f₂ (3Hz), test set, first 1 second (1000 samples)
- **Graph 2**: All 4 frequencies, 2×2 subplot layout
- **Data source**: Always use test set to demonstrate generalization
- **Format**: PNG, 300 DPI

---

## Common Pitfalls to Avoid

1. **State Reset Between Samples**: Never reset hidden_state within an epoch
2. **Forgetting State Detachment**: Will cause memory explosion
3. **Vectorized Randomization**: Must loop over samples for A_i(t) and φ_i(t)
4. **Wrong Batch Size**: Must be 1 for L=1 constraint
5. **Shuffled DataLoader**: Must preserve temporal order (shuffle=False)
6. **Same Seed for Train/Test**: Defeats generalization testing

---

## File Organization

```
HW2/
├── docs/                       # All documentation
│   ├── PRD.md                  # Main Product Requirements Document
│   ├── ARCHITECTURE.md         # System architecture and ADRs
│   ├── DATA_GENERATION.md      # Data generation guide
│   ├── MODEL_ARCHITECTURE.md   # Model design guide
│   ├── TRAINING_PIPELINE.md    # Training guide (CRITICAL)
│   ├── EVALUATION.md           # Evaluation guide
│   ├── VISUALIZATION.md        # Visualization guide
│   ├── INTEGRATION.md          # Integration guide
│   └── RESEARCH_AND_EXPERIMENTS.md  # Experiments and research
│
├── src/                    # Implementation
│   ├── data_generation.py  # SignalGenerator class
│   ├── dataset.py          # PyTorch Dataset wrapper
│   ├── model.py            # FrequencyLSTM class
│   ├── training.py         # StatefulTrainer class (CRITICAL)
│   ├── evaluation.py       # Evaluator class
│   └── visualization.py    # Visualizer class
│
├── tests/                  # Unit tests (97% coverage)
├── data/                   # Generated datasets
├── models/                 # Saved checkpoints
├── outputs/                # Results (metrics, graphs, predictions)
│
├── main.py                 # Orchestration script
├── run_experiments.py      # Hyperparameter experiments
├── config.yaml             # Hyperparameters
├── ASSIGNMENT_REQUIREMENTS.md  # Assignment specs in English
└── CLAUDE.md               # This file
```

---

## Assignment Success Criteria

### Required Deliverables
1. Two datasets (train/test) with different noise (Seed #1 vs #2)
2. Trained LSTM model with proper state management
3. MSE metrics showing generalization (MSE_test ≈ MSE_train)
4. Two visualization graphs (comparison + all frequencies)

### Critical for Success
- **State preservation pattern** implemented correctly
- **Per-sample randomization** in data generation
- **Generalization demonstrated** through test set performance
- **Clean extraction** visible in graphs

### Documentation References

**Main Documentation** (start here):
- `docs/PRD.md` - Complete PRD with goals, metrics, requirements
- `docs/ARCHITECTURE.md` - System architecture, C4 diagrams, ADRs

**Implementation Guides** (use these when coding):
- `docs/DATA_GENERATION.md` - Data generation specification
- `docs/MODEL_ARCHITECTURE.md` - LSTM model design
- `docs/TRAINING_PIPELINE.md` - Training with state management (CRITICAL!)
- `docs/EVALUATION.md` - Metrics and generalization testing
- `docs/VISUALIZATION.md` - Graph creation specifications
- `docs/INTEGRATION.md` - End-to-end pipeline
- `docs/RESEARCH_AND_EXPERIMENTS.md` - Experimental results

**Assignment Context**:
- `ASSIGNMENT_REQUIREMENTS.md` - Original assignment translated from Hebrew

---

## Quick Reference: Key Formulas

**Noisy Signal:**
```
Sinus_i^noisy(t) = A_i(t) · sin(2π · f_i · t + φ_i(t))
where A_i(t) ~ Uniform(0.8, 1.2), φ_i(t) ~ Uniform(0, 2π)

S(t) = (1/4) · Σ(i=1 to 4) Sinus_i^noisy(t)
```

**Clean Target:**
```
Target_i(t) = sin(2π · f_i · t)
```

**MSE Calculation:**
```
MSE = (1/40000) · Σ(prediction - target)²
```
