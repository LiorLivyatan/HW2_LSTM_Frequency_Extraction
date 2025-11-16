# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

This is an M.Sc. assignment for developing an LSTM system to extract individual frequency components from mixed noisy signals. The task is **conditional regression**: given a noisy mixed signal S(t) containing 4 frequencies (1Hz, 3Hz, 5Hz, 7Hz) and a one-hot selection vector C, the LSTM must output the clean sinusoid for the selected frequency.

### How to Use This Repository

**IMPORTANT: This codebase follows a PRD-driven development approach.**

The `prd/` folder contains detailed Product Requirements Documents for each phase. **These PRDs are your primary implementation guides** - they contain:
- Complete architectural specifications with code examples
- Testing strategies and validation methods
- Success criteria and common pitfalls
- Estimated time for each phase

**Before implementing any phase, read its corresponding PRD first:**
- Starting? Read `prd/00_MASTER_PRD.md` for architecture overview
- Implementing Phase X? Read `prd/0X_*_PRD.md` for detailed specs

This CLAUDE.md provides quick reference for critical concepts, but the PRDs contain the complete implementation details.

---

## Critical Architecture Concepts

### The L=1 State Preservation Pattern (MOST CRITICAL)

This assignment has a **pedagogical constraint**: Sequence Length L=1, meaning each sample is processed individually. The LSTM must learn temporal patterns through internal state preservation, NOT through batched sequences.

**Correct Implementation Pattern:**
```python
# In training loop (src/training.py)
hidden_state = None  # Initialize ONCE per epoch

for sample in dataloader:
    # Forward pass with previous state
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
- Without state preservation: Model cannot learn temporal patterns
- Without state detachment: Memory explosion after thousands of samples
- This is THE KEY to making L=1 work and is the assignment's pedagogical focus

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

**IMPORTANT**: The project is organized into 6 phases. Each phase has a detailed PRD (Product Requirements Document) in the `prd/` folder that contains:
- Complete implementation specifications
- Architecture diagrams and code examples
- Testing strategies
- Success criteria
- Common pitfalls and mitigation strategies

**When implementing each phase, ALWAYS read the corresponding PRD first.**

1. **Phase 1: Data Generation** (2-3 hours)
   - **Read**: `prd/01_DATA_GENERATION_PRD.md`
   - Generate `data/train_data.npy` and `data/test_data.npy`
   - Validate with FFT to ensure correct frequencies
   - Follow SignalGenerator class specification exactly

2. **Phase 2: Model Architecture** (1-2 hours)
   - **Read**: `prd/02_MODEL_ARCHITECTURE_PRD.md`
   - Build `FrequencyLSTM` class in `src/model.py`
   - Test with dummy data before training
   - Verify state shapes and parameter count

3. **Phase 3: Training Pipeline** (4-6 hours) ⭐ **MOST CRITICAL**
   - **Read**: `prd/03_TRAINING_PIPELINE_PRD.md` (READ CAREFULLY!)
   - Implement `StatefulTrainer` in `src/training.py`
   - Get state preservation pattern exactly right
   - Monitor for memory leaks
   - This PRD contains the critical state detachment pattern

4. **Phase 4: Evaluation** (1-2 hours)
   - **Read**: `prd/04_EVALUATION_PRD.md`
   - Calculate MSE on train and test sets
   - Verify generalization: MSE_test ≈ MSE_train
   - Follow Evaluator class specification

5. **Phase 5: Visualization** (2-3 hours)
   - **Read**: `prd/05_VISUALIZATION_PRD.md`
   - Create Graph 1: Single frequency comparison (Target vs LSTM vs Noisy)
   - Create Graph 2: All 4 frequencies (2×2 grid)
   - Use exact graph specifications from PRD

6. **Phase 6: Integration** (1-2 hours)
   - **Read**: `prd/06_INTEGRATION_PRD.md`
   - Build `main.py` orchestration script
   - Add CLI with argparse
   - Follow configuration structure from PRD

**Architecture Overview**: Before starting, read `prd/00_MASTER_PRD.md` for the complete system architecture and technology stack decisions.

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
# CRITICAL: batch_size=1, shuffle=False
train_loader = DataLoader(
    dataset,
    batch_size=1,      # Required for L=1
    shuffle=False,     # Preserve temporal order
    num_workers=0      # Avoid multiprocessing complications
)
```

### Model Input/Output Shapes
- **Input**: `(batch=1, seq_len=1, features=5)` - reshaped from (1, 5)
- **Output**: `(batch=1, 1)` - scalar prediction
- **Hidden state**: `(num_layers, batch=1, hidden_size)` - typically (1, 1, 64)

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
├── prd/                    # Product Requirements Documents (design specs)
│   ├── 00_MASTER_PRD.md    # Start here for architecture overview
│   ├── 01-06_*.md          # Phase-specific detailed specs
│
├── src/                    # Implementation (to be created)
│   ├── data_generation.py  # SignalGenerator class
│   ├── dataset.py          # PyTorch Dataset wrapper
│   ├── model.py            # FrequencyLSTM class
│   ├── training.py         # StatefulTrainer class (CRITICAL)
│   ├── evaluation.py       # Evaluator class
│   └── visualization.py    # Visualizer class
│
├── data/                   # Generated datasets
├── models/                 # Saved checkpoints
├── outputs/                # Results (metrics, graphs, predictions)
├── tests/                  # Unit tests
│
├── main.py                 # Orchestration script
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

**Primary Implementation Guides** (use these when coding):
- `prd/00_MASTER_PRD.md` - System architecture overview and technology decisions
- `prd/01_DATA_GENERATION_PRD.md` - Complete data generation specification
- `prd/02_MODEL_ARCHITECTURE_PRD.md` - LSTM model design and structure
- `prd/03_TRAINING_PIPELINE_PRD.md` - Training loop with state management (CRITICAL!)
- `prd/04_EVALUATION_PRD.md` - Metrics calculation and generalization testing
- `prd/05_VISUALIZATION_PRD.md` - Graph creation specifications
- `prd/06_INTEGRATION_PRD.md` - End-to-end pipeline orchestration

**Assignment Context**:
- `ASSIGNMENT_REQUIREMENTS.md` - Original assignment translated from Hebrew
- `L2-homework.pdf` - Original Hebrew assignment document

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
