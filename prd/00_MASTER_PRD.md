# LSTM Frequency Extraction System - Master PRD

**Project**: M.Sc. LLM Course - HW2
**Instructor**: Dr. Segal Yoram
**Date**: November 2025
**Version**: 1.0

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Technology Stack](#technology-stack)
4. [Development Phases](#development-phases)
5. [Critical Success Factors](#critical-success-factors)
6. [Risk Management](#risk-management)
7. [Timeline and Milestones](#timeline-and-milestones)
8. [Dependencies Between Phases](#dependencies-between-phases)

---

## Project Overview

### Mission
Develop an LSTM (Long Short-Term Memory) neural network system capable of extracting individual pure frequency components from a mixed, noisy signal containing 4 different sinusoidal frequencies.

### The Challenge
Given a noisy mixed signal S(t) composed of 4 ideal sinusoidal frequencies (1Hz, 3Hz, 5Hz, 7Hz) with varying random amplitude and phase at EVERY sample, the LSTM must learn to:
- Extract each pure frequency component separately
- Ignore random noise and variations
- Be conditioned on a one-hot selection vector indicating which frequency to extract

### Success Criteria
1. **Low MSE**: Achieve low Mean Squared Error on both training and test sets
2. **Generalization**: MSE_test ≈ MSE_train (within 10%)
3. **State Management**: Correctly implement L=1 with internal state preservation
4. **Visualization**: Clear graphs demonstrating clean extraction from noise

### Assignment Context
This is a **conditional regression** problem with a critical pedagogical constraint:
- **Sequence Length L = 1** (default): Each sample processed individually
- **State Preservation**: LSTM internal state (h_t, c_t) must be preserved across samples
- **Purpose**: Demonstrate LSTM's temporal learning capability through internal memory

---

## System Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1: Data Generation                  │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │ Seed #1      │         │ Seed #2      │                  │
│  │ (Training)   │         │ (Test)       │                  │
│  └──────┬───────┘         └──────┬───────┘                  │
│         │                        │                           │
│         v                        v                           │
│  [Generate Noisy Mixed Signal S(t)]                         │
│  [Generate Clean Targets for each frequency]                │
│         │                        │                           │
│         v                        v                           │
│  train_data.npy           test_data.npy                     │
│  (40,000 x 6)             (40,000 x 6)                      │
└─────────┬───────────────────────┬───────────────────────────┘
          │                       │
          v                       │
┌─────────────────────────────────┼───────────────────────────┐
│         PHASE 2: Model Architecture                         │
│                                 │                            │
│  ┌────────────────────────────────────────────┐             │
│  │        FrequencyLSTM Model                 │             │
│  │                                            │             │
│  │  Input: [S(t), C1, C2, C3, C4] (5-dim)   │             │
│  │         ↓                                  │             │
│  │  LSTM Layer (hidden_size=64-128)          │             │
│  │         ↓                                  │             │
│  │  Fully Connected Output Layer             │             │
│  │         ↓                                  │             │
│  │  Output: Target_i(t) (scalar)             │             │
│  │                                            │             │
│  │  State: (h_t, c_t) preserved across       │             │
│  │         samples                            │             │
│  └────────────────────────────────────────────┘             │
└─────────┬───────────────────────┬───────────────────────────┘
          │                       │
          v                       │
┌─────────────────────────────────┼───────────────────────────┐
│         PHASE 3: Training Pipeline              │            │
│                                 │                            │
│  ┌────────────────────────────────────────────┐             │
│  │     StatefulTrainer                        │             │
│  │                                            │             │
│  │  For each epoch:                           │             │
│  │    hidden_state = None  (init once)        │             │
│  │                                            │             │
│  │    For each sample in 40,000:              │             │
│  │      output, hidden_state =                │             │
│  │        model(input, hidden_state)          │             │
│  │                                            │             │
│  │      loss = MSE(output, target)            │             │
│  │      loss.backward()                       │             │
│  │      optimizer.step()                      │             │
│  │                                            │             │
│  │      hidden_state = tuple(h.detach()       │             │
│  │                      for h in hidden_state)│             │
│  │                      ↑                     │             │
│  │                 CRITICAL!                  │             │
│  └────────────────────────────────────────────┘             │
│                                 │                            │
│                   [Trained Model: best_model.pth]           │
└─────────┬───────────────────────┬───────────────────────────┘
          │                       │
          v                       v
┌─────────────────────────────────────────────────────────────┐
│    PHASE 4: Evaluation          PHASE 5: Visualization      │
│                                                              │
│  • MSE_train calculation        • Graph 1: Frequency         │
│  • MSE_test calculation           Comparison (1 freq)       │
│  • Generalization check         • Graph 2: All 4             │
│  • Per-frequency metrics          Frequencies (2x2 grid)    │
│                                                              │
│  Output: metrics.json           Output: PNG graphs          │
└──────────────────────────────────────────────────────────────┘
          │
          v
┌─────────────────────────────────────────────────────────────┐
│         PHASE 6: Integration & Orchestration                │
│                                                              │
│  main.py: End-to-end pipeline with CLI and configuration    │
└──────────────────────────────────────────────────────────────┘
```

### Component Architecture

```
HW2/
├── prd/                              # This folder - PRD documents
│   ├── 00_MASTER_PRD.md              # This file
│   ├── 01_DATA_GENERATION_PRD.md
│   ├── 02_MODEL_ARCHITECTURE_PRD.md
│   ├── 03_TRAINING_PIPELINE_PRD.md
│   ├── 04_EVALUATION_PRD.md
│   ├── 05_VISUALIZATION_PRD.md
│   └── 06_INTEGRATION_PRD.md
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── data_generation.py            # SignalGenerator class
│   ├── dataset.py                    # PyTorch Dataset wrapper
│   ├── model.py                      # FrequencyLSTM class
│   ├── training.py                   # StatefulTrainer class
│   ├── evaluation.py                 # Evaluator class
│   └── visualization.py              # Visualizer class
│
├── data/                             # Generated datasets
│   ├── train_data.npy               # 40,000 x 6 array (Seed #1)
│   └── test_data.npy                # 40,000 x 6 array (Seed #2)
│
├── models/                           # Saved models
│   └── best_model.pth               # Trained LSTM checkpoint
│
├── outputs/                          # Results
│   ├── graphs/
│   │   ├── frequency_comparison.png # Graph 1
│   │   └── all_frequencies.png      # Graph 2
│   └── metrics.json                 # MSE and evaluation metrics
│
├── main.py                           # Main orchestration script
├── config.yaml                       # Configuration parameters
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## Technology Stack

### Core Libraries

| Library | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| **Python** | 3.8+ | Programming language | Standard for ML/DL |
| **PyTorch** | 2.0+ | LSTM implementation | **CRITICAL**: Explicit state control for L=1 |
| **NumPy** | 1.24+ | Signal generation & arrays | Efficient vectorized operations |
| **Matplotlib** | 3.7+ | Visualization | Standard scientific plotting |

### Why PyTorch Over TensorFlow/Keras?

**Critical Requirement**: Manual LSTM state management for L=1

PyTorch LSTM allows explicit state handling:
```python
output, (h_n, c_n) = lstm(input, (h_0, c_0))
```

This is ESSENTIAL for:
1. Preserving state across individual samples (L=1)
2. State detachment to prevent memory explosion
3. Understanding state propagation (pedagogical goal)

Keras/TensorFlow abstract state management, making the L=1 constraint harder to implement correctly.

### Development Tools
- **Git**: Version control
- **pytest**: Unit testing
- **black**: Code formatting
- **mypy**: Type checking (optional)

---

## Development Phases

### Phase Breakdown

| Phase | PRD Document | Priority | Est. Time | Dependencies |
|-------|--------------|----------|-----------|--------------|
| **Phase 1** | [01_DATA_GENERATION_PRD.md](01_DATA_GENERATION_PRD.md) | Highest | 2-3 hours | None |
| **Phase 2** | [02_MODEL_ARCHITECTURE_PRD.md](02_MODEL_ARCHITECTURE_PRD.md) | High | 1-2 hours | None |
| **Phase 3** | [03_TRAINING_PIPELINE_PRD.md](03_TRAINING_PIPELINE_PRD.md) | Critical | 4-6 hours | Phase 1, 2 |
| **Phase 4** | [04_EVALUATION_PRD.md](04_EVALUATION_PRD.md) | High | 1-2 hours | Phase 3 |
| **Phase 5** | [05_VISUALIZATION_PRD.md](05_VISUALIZATION_PRD.md) | Medium | 2-3 hours | Phase 4 |
| **Phase 6** | [06_INTEGRATION_PRD.md](06_INTEGRATION_PRD.md) | Medium | 1-2 hours | All phases |

**Total Estimated Development Time**: 11-18 hours

### Phase Descriptions

#### Phase 1: Data Generation
**Objective**: Create reproducible, high-quality training and test datasets

**Key Deliverables**:
- `SignalGenerator` class with proper randomization
- `train_data.npy` (Seed #1)
- `test_data.npy` (Seed #2)

**Critical Requirement**: Random amplitude A_i(t) and phase φ_i(t) at EVERY sample t

[→ Full PRD: 01_DATA_GENERATION_PRD.md](01_DATA_GENERATION_PRD.md)

---

#### Phase 2: Model Architecture
**Objective**: Design LSTM model with proper I/O structure and state management

**Key Deliverables**:
- `FrequencyLSTM` class
- State initialization and forward pass methods
- Model architecture validated with dummy data

**Critical Requirement**: Support stateful operation with explicit state returns

[→ Full PRD: 02_MODEL_ARCHITECTURE_PRD.md](02_MODEL_ARCHITECTURE_PRD.md)

---

#### Phase 3: Training Pipeline
**Objective**: Implement L=1 training loop with proper state preservation

**Key Deliverables**:
- `StatefulTrainer` class
- Custom training loop with state management
- Model checkpointing

**Critical Requirement**: State detachment pattern to prevent memory explosion

[→ Full PRD: 03_TRAINING_PIPELINE_PRD.md](03_TRAINING_PIPELINE_PRD.md)

---

#### Phase 4: Evaluation
**Objective**: Calculate performance metrics and verify generalization

**Key Deliverables**:
- MSE calculations (train and test)
- Generalization analysis
- Per-frequency metrics

**Critical Requirement**: MSE_test ≈ MSE_train to prove learning

[→ Full PRD: 04_EVALUATION_PRD.md](04_EVALUATION_PRD.md)

---

#### Phase 5: Visualization
**Objective**: Create required graphs demonstrating extraction quality

**Key Deliverables**:
- Graph 1: Single frequency comparison (3 overlaid signals)
- Graph 2: All 4 frequencies (2x2 subplot grid)

**Critical Requirement**: Use test set (Seed #2) for all visualizations

[→ Full PRD: 05_VISUALIZATION_PRD.md](05_VISUALIZATION_PRD.md)

---

#### Phase 6: Integration
**Objective**: Assemble end-to-end pipeline with proper orchestration

**Key Deliverables**:
- `main.py` orchestration script
- Configuration management
- Complete documentation

**Critical Requirement**: Reproducibility across all components

[→ Full PRD: 06_INTEGRATION_PRD.md](06_INTEGRATION_PRD.md)

---

## Critical Success Factors

### 1. Proper State Management (THE KEY FACTOR)

**The Challenge**: With L=1, each sample is processed individually, but temporal dependencies must be learned through internal state.

**The Solution**:
```python
# Correct implementation pattern
hidden_state = None  # Initialize once per epoch

for sample_idx, (input, target) in enumerate(dataloader):
    # Forward pass with previous state
    output, hidden_state = model(input, hidden_state)

    # Compute loss and update weights
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    # CRITICAL: Detach state from computation graph
    hidden_state = tuple(h.detach() for h in hidden_state)
```

**Why This Matters**:
- Without state preservation: No temporal learning possible
- Without state detachment: Memory explosion after thousands of samples
- This is the pedagogical focus of the assignment

### 2. Per-Sample Randomization

**Requirement**: Amplitude A_i(t) and phase φ_i(t) must be random at EVERY sample t

**Why**: Forces the network to learn underlying frequency structure, not memorize noise patterns

**Implementation**:
```python
for t in time_samples:
    A_i = np.random.uniform(0.8, 1.2)  # New random value each sample
    phi_i = np.random.uniform(0, 2*np.pi)  # New random value each sample
    noisy_sinus[t] = A_i * np.sin(2*np.pi*f_i*t + phi_i)
```

### 3. Seed Separation for Generalization

**Training Set**: Seed #1
**Test Set**: Seed #2

**Why**: Completely different noise patterns verify the network learned frequency structure, not specific noise

**Success Metric**: MSE_test ≈ MSE_train (within 10%)

### 4. Correct Data Structure

**Each row** in 40,000-row dataset:
```
[S(t), C1, C2, C3, C4, Target_i(t)]
  1     4 values      1

Total: 6 values per row
```

**Structure**:
- 10,000 time samples × 4 frequencies = 40,000 rows
- Input to LSTM: [S(t), C1, C2, C3, C4] (5-dimensional)
- Output from LSTM: Target_i(t) (scalar)

### 5. Proper Evaluation Methodology

**Required Metrics**:
1. MSE_train: Performance on training data (Seed #1)
2. MSE_test: Performance on test data (Seed #2)
3. Generalization gap: |MSE_test - MSE_train|

**Required Visualizations**:
1. Graph 1: Target vs. LSTM Output vs. Noisy Input (one frequency)
2. Graph 2: All 4 frequencies extracted (2x2 grid)

**Critical**: Use test set for visualizations to demonstrate generalization

---

## Risk Management

### Risk Matrix

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| **State management complexity** | High | Critical | Extensive testing, clear documentation, reference implementation |
| **Memory explosion from long backprop** | Medium | High | Mandatory state detachment after each step |
| **Overfitting to noise** | Medium | High | Test set validation with different seed |
| **Poor convergence** | Medium | Medium | Hyperparameter tuning, gradient clipping |
| **Data generation bugs** | Low | High | FFT validation, visual inspection |
| **Incorrect L=1 implementation** | Medium | Critical | Unit tests, state flow verification |

### Mitigation Details

#### State Management Complexity
**Prevention**:
- Follow reference implementation pattern exactly
- Add assertions to verify state shapes
- Test with small dataset first (100 samples)
- Log state statistics during training

**Detection**:
- Monitor memory usage
- Check state values for NaN/Inf
- Verify state changes across samples

#### Overfitting Prevention
**Strategies**:
- Different seeds for train/test
- Monitor MSE_test - MSE_train gap
- Early stopping if gap increases
- No dropout (not applicable for this regression task)

---

## Timeline and Milestones

### Development Timeline

```
Week 1: Data and Model Foundation
├─ Days 1-2: Phase 1 - Data Generation
│  └─ Milestone: Validated datasets created
├─ Day 3: Phase 2 - Model Architecture
│  └─ Milestone: LSTM model tested with dummy data
└─ Days 4-7: Phase 3 - Training Pipeline
   └─ Milestone: Model trained with proper state management

Week 2: Evaluation and Integration
├─ Days 8-9: Phase 4 - Evaluation
│  └─ Milestone: MSE metrics calculated, generalization verified
├─ Days 10-11: Phase 5 - Visualization
│  └─ Milestone: Required graphs generated
├─ Day 12: Phase 6 - Integration
│  └─ Milestone: End-to-end pipeline working
└─ Days 13-14: Testing, Documentation, Refinement
   └─ Final Milestone: Complete submission ready
```

### Milestones and Checkpoints

| Milestone | Completion Criteria | Verification Method |
|-----------|---------------------|---------------------|
| **M1: Data Ready** | Train and test datasets created | FFT shows correct frequencies, visual inspection |
| **M2: Model Ready** | LSTM forward pass works | Dummy data test, state shapes verified |
| **M3: Training Works** | Model converges | Loss decreasing, no memory errors |
| **M4: Generalization** | MSE_test ≈ MSE_train | Numerical comparison (< 10% difference) |
| **M5: Visualization** | Graphs show clean extraction | Visual quality check |
| **M6: Integration** | End-to-end runs without errors | Full pipeline test |

---

## Dependencies Between Phases

### Dependency Graph

```
Phase 1: Data Generation
    │
    ├──→ Phase 3: Training (needs data)
    │
    └──→ Phase 4: Evaluation (needs test data)

Phase 2: Model Architecture
    │
    └──→ Phase 3: Training (needs model)

Phase 3: Training
    │
    ├──→ Phase 4: Evaluation (needs trained model)
    │
    └──→ Phase 5: Visualization (needs predictions)

Phase 4: Evaluation
    │
    └──→ Phase 5: Visualization (needs metrics)

Phase 5: Visualization
    │
    └──→ Phase 6: Integration

All Phases
    │
    └──→ Phase 6: Integration (orchestrates everything)
```

### Parallel Development Opportunities

**Can be developed in parallel**:
- Phase 1 (Data Generation) and Phase 2 (Model Architecture)
- Phase 4 (Evaluation) and Phase 5 (Visualization) - if using same interface

**Must be sequential**:
- Phase 3 requires Phase 1 and 2 complete
- Phase 4 and 5 require Phase 3 complete
- Phase 6 requires all phases complete

### Development Sequence Recommendation

**Recommended Order**:
1. **Phase 1** first (validate data quality early)
2. **Phase 2** second (can start while validating Phase 1)
3. **Phase 3** third (most complex, needs focused effort)
4. **Phase 4** fourth (straightforward once training works)
5. **Phase 5** fifth (depends on predictions from Phase 4)
6. **Phase 6** last (ties everything together)

---

## Testing Strategy

### Unit Testing
- Each module has dedicated test file
- Test data generation with known frequencies
- Test model forward pass with dummy data
- Test state preservation logic

### Integration Testing
- Test data → model pipeline
- Test training → evaluation pipeline
- Test full end-to-end workflow

### Validation Testing
- FFT analysis of generated signals
- Visual inspection of noisy vs. clean
- MSE calculation verification
- Graph quality checks

---

## Success Criteria Summary

### Technical Success
- [ ] MSE_train < 0.01 (ideally < 0.001)
- [ ] MSE_test ≈ MSE_train (within 10%)
- [ ] All 4 frequencies extracted cleanly
- [ ] Visualizations clearly show extraction quality
- [ ] State management correctly implemented (L=1)

### Code Quality
- [ ] Modular, well-organized code structure
- [ ] Clear documentation and comments
- [ ] Reproducible results (seed control)
- [ ] No memory leaks or errors
- [ ] Clean git history

### Documentation
- [ ] All PRDs complete and accurate
- [ ] README with usage instructions
- [ ] Code comments explaining critical sections
- [ ] Results documented with metrics and graphs

---

## References

- **Assignment Document**: `ASSIGNMENT_REQUIREMENTS.md`
- **Original PDF**: `L2-homework.pdf`
- **PyTorch LSTM Docs**: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- **NumPy Random Docs**: https://numpy.org/doc/stable/reference/random/index.html

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-16 | System | Initial PRD creation |

---

**Next Steps**: Review each phase-specific PRD before starting implementation.
