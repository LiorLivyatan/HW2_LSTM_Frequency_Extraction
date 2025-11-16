# LSTM Frequency Extraction Assignment - Complete Requirements Guide

**Course:** M.Sc. LLM Course
**Instructor:** Dr. Segal Yoram
**Date:** November 2025

---

## Table of Contents
1. [Assignment Overview](#assignment-overview)
2. [Problem Statement](#problem-statement)
3. [Dataset Creation Requirements](#dataset-creation-requirements)
4. [LSTM Model Architecture](#lstm-model-architecture)
5. [Critical Implementation Requirements](#critical-implementation-requirements)
6. [Performance Evaluation](#performance-evaluation)
7. [Deliverables Checklist](#deliverables-checklist)
8. [Grading Criteria for A+/100](#grading-criteria-for-a100)

---

## Assignment Overview

### Goal
Develop an LSTM (Long Short-Term Memory) neural network system capable of extracting individual pure frequency components from a mixed, noisy signal containing 4 different sinusoidal frequencies.

### The Challenge
Given a noisy mixed signal S(t) composed of 4 ideal sinusoidal frequencies with varying random noise, the LSTM must learn to extract each pure frequency separately while ignoring the random noise.

---

## Problem Statement

### 1.1 The Task

**Input:**
- A noisy mixed signal `S[t]` composed of 4 frequencies: f₁=1Hz, f₂=3Hz, f₃=5Hz, f₄=7Hz
- A one-hot selection vector `C` indicating which frequency to extract

**Output:**
- A clean (pure) sinusoidal signal at the selected frequency: `Target_i[t]`

### 1.2 The Principle

This is a **Conditional Regression** problem:
- The system must learn to separate and extract the correct frequency component
- The extraction is conditioned on the selection vector C
- The LSTM must ignore random noise and variations in amplitude/phase

### 1.3 Usage Example

If selection vector C = [0, 1, 0, 0], the system should extract the clean f₂ (3Hz) frequency:

```
Input:  (S[t], C) → LSTM → Output: Pure_Sinus₂[t]

Step 1: S[0] + C → LSTM → Pure_Sinus₂[0]  (noisy input → clean output)
Step 2: S[1] + C → LSTM → Pure_Sinus₂[1]  (noisy input → clean output)
...
```

---

## Dataset Creation Requirements

### 2.1 General Parameters (MUST FOLLOW EXACTLY)

| Parameter | Value | Critical? |
|-----------|-------|-----------|
| **Frequencies** | f₁=1Hz, f₂=3Hz, f₃=5Hz, f₄=7Hz | ✓ YES |
| **Time Domain** | 0-10 seconds | ✓ YES |
| **Sampling Rate (Fs)** | 1000 Hz | ✓ YES |
| **Total Samples** | 10,000 samples | ✓ YES |

### 2.2 Noisy Signal Creation (CRITICAL POINT!)

**⚠️ CRITICAL:** The amplitude `A_i(t)` and phase `φ_i(t)` MUST vary randomly at EVERY sample `t`:

For each noisy sinusoid at time t:
```
A_i(t) ~ Uniform(0.8, 1.2)     # Random amplitude per sample
φ_i(t) ~ Uniform(0, 2π)         # Random phase per sample

Sinus_i^noisy(t) = A_i(t) · sin(2π · f_i · t + φ_i(t))
```

**Mixed Signal (System Input):**
```
S(t) = (1/4) · Σ(i=1 to 4) Sinus_i^noisy(t)
```

This is normalized by dividing by 4.

### 2.3 Ground Truth Targets (NO NOISE!)

For each frequency f_i, the clean target is:
```
Target_i(t) = sin(2π · f_i · t)
```

Pure sinusoid with:
- Amplitude = 1.0 (constant)
- Phase = 0 (constant)
- NO random variations

### 2.4 Train vs. Test Sets (ESSENTIAL)

| Dataset | Seed | Purpose | Samples |
|---------|------|---------|---------|
| **Training Set** | Seed #1 | Train the LSTM | 40,000 rows |
| **Test Set** | Seed #2 | Evaluate generalization | 40,000 rows |

**Why 40,000 rows?**
- 10,000 samples × 4 frequencies = 40,000 training examples

**CRITICAL:** Test set uses completely different random noise (Seed #2) to verify the LSTM learned frequency structure, not memorized noise patterns!

---

## LSTM Model Architecture

### 3.1 Input Structure

Each row in the training dataset contains:

| Column | Description | Shape |
|--------|-------------|-------|
| `S[t]` | Noisy mixed signal sample | Scalar (1,) |
| `C1, C2, C3, C4` | One-hot selection vector | Vector (4,) |

**Example row format:**
```
Row: [S[t], C1, C2, C3, C4]
```

The input to LSTM at each timestep is a vector of size 5.

### 3.2 Output Structure

- Single scalar value: `Target_i[t]`
- The pure sinusoid sample for the selected frequency

### 3.3 Training Dataset Table Structure

Example (Training Set):

| Row | t (sec) | S[t] (noisy) | C (selection) | Target (clean) |
|-----|---------|--------------|---------------|----------------|
| 1 | 0.000 | 0.8124 | [1,0,0,0] | 0.0000 |
| ... | ... | ... | [1,0,0,0] | ... |
| 10001 | 0.000 | 0.8124 | [0,1,0,0] | 0.0000 |
| 10002 | 0.001 | 0.7932 | [0,1,0,0] | 0.0188 |
| ... | ... | ... | ... | ... |
| 40000 | 9.999 | 0.6543 | [0,0,0,1] | 0.0440 |

---

## Critical Implementation Requirements

### 4.1 Sequence Length = 1 (PEDAGOGICAL CONSTRAINT)

**⚠️ MOST CRITICAL REQUIREMENT FOR A+ GRADE:**

The default sequence length MUST be **L = 1**, meaning:
- Each training sample is processed individually
- The LSTM processes ONE sample at a time
- NO sequences/batches of multiple consecutive samples

### 4.2 Internal State Management (CRITICAL!)

**The key to success:** Proper management of LSTM's internal state.

LSTM has two internal state components:
- **h_t** (Hidden State)
- **c_t** (Cell State)

These allow the network to learn **temporal dependencies** between samples.

#### State Management Requirements:

**✓ YOU MUST:** Ensure internal state (h_t, c_t) is **NOT reset** between consecutive samples

**✗ DO NOT:** Reset state between samples when working with L=1

#### State Management Comparison Table:

| Scenario | State Management | Required Action | Enables Learning |
|----------|------------------|-----------------|------------------|
| **Regular LSTM (L>1)** | Resets state at each sequence | Assumes no sequential relationship | NO temporal learning needed |
| **This Assignment (L=1)** | Preserves state across samples | State passed as input to next step | YES - learns sequential patterns via state |

**Why this matters:**
- With L=1 and preserved state, the LSTM must use its internal memory to learn the frequency structure
- This demonstrates LSTM's temporal advantage
- The network learns long-term dependencies through state propagation

### 4.3 Alternative Approach (OPTIONAL - For Extra Credit)

**Students are invited** to implement longer sequence lengths (L ≠ 1):

**Options:**
- L = 10 or L = 50 (Sliding Window approach)

**If you choose L ≠ 1, you MUST include in your report:**
1. Detailed justification for the choice
2. Explanation of how this leverages LSTM's temporal advantage
3. Description of how you handle the output (single value vs. sequence output)

---

## Performance Evaluation

### 5.1 Success Metrics (REQUIRED)

You MUST calculate and report:

#### 1. MSE on Training Set (Seed #1):
```
MSE_train = (1/40000) · Σ(j=1 to 40000) [LSTM(S_train[t], C) - Target[t]]²
```

#### 2. MSE on Test Set (Seed #2):
```
MSE_test = (1/40000) · Σ(j=1 to 40000) [LSTM(S_test[t], C) - Target[t]]²
```

#### 3. Generalization Check:
```
If MSE_test ≈ MSE_train → System generalizes well!
```

This proves the LSTM learned the underlying frequency structure, not just memorized noise patterns.

### 5.2 Required Visualizations

**Graph 1: Frequency Comparison (for one frequency, e.g., f₂)**

On the same plot, show all three components:
1. Target₂ (clean ground truth) - as a line
2. LSTM Output (green dots)
3. S (noisy mixed signal) - as background (chaos)

This demonstrates the LSTM successfully extracted the clean frequency from noise.

**Graph 2: All 4 Extracted Frequencies**

Create 4 sub-plots, one for each frequency f_i, each showing the extraction quality separately.

**⚠️ CRITICAL:** Use Test Set (Seed #2) for all visualizations to demonstrate generalization!

---

## Deliverables Checklist

### ✓ Required Deliverables:

- [ ] **Data Generation:**
  - [ ] Create 2 datasets (Train with Seed #1, Test with Seed #2)
  - [ ] Implement random amplitude and phase variation at EVERY sample
  - [ ] Verify 40,000 rows per dataset (10,000 samples × 4 frequencies)

- [ ] **LSTM Model:**
  - [ ] Build LSTM accepting input (S[t], C)
  - [ ] Output clean sample Target_i[t]
  - [ ] Implement proper internal state management (preserved across samples for L=1)

- [ ] **State Management:**
  - [ ] Ensure internal state (h_t, c_t) is preserved between consecutive samples when L=1
  - [ ] This is required for sequential learning

- [ ] **Evaluation:**
  - [ ] Calculate MSE_train
  - [ ] Calculate MSE_test
  - [ ] Verify generalization: MSE_test ≈ MSE_train
  - [ ] Create Graph 1: Single frequency comparison (Target vs LSTM vs Noisy)
  - [ ] Create Graph 2: All 4 frequencies extracted separately

- [ ] **Report/Documentation:**
  - [ ] Explain methodology
  - [ ] Show results with metrics
  - [ ] Include visualizations
  - [ ] Discuss why state management enables learning from noise

---

## Critical Success Factors

### Essential Requirements (From Assignment):

1. **Correct Data Generation**
   - Random A_i(t) and φ_i(t) at EVERY sample t
   - Proper normalization of mixed signal
   - Separate seeds for train/test (Seed #1 for training, Seed #2 for testing)

2. **Proper LSTM Implementation**
   - Correct input structure (S[t], C)
   - Proper output (single clean sample)
   - Working conditional regression

3. **Critical State Management**
   - Internal state preserved across samples when L=1
   - Correct understanding and implementation
   - This is THE KEY to success!

4. **Performance Metrics**
   - MSE_train and MSE_test calculated correctly
   - Good generalization demonstrated (MSE_test ≈ MSE_train)
   - Low MSE values showing successful extraction

5. **Required Visualizations**
   - Graph 1: Single frequency comparison (Target vs LSTM Output vs Noisy Input)
   - Graph 2: All 4 extracted frequencies shown separately
   - Uses test set (Seed #2) for visualization

### Optional Enhancement (Mentioned in Assignment):

- **Alternative Sequence Length (L ≠ 1)**
  - Students are invited to implement L > 1 (e.g., L=10 or L=50) with sliding window
  - MUST provide detailed justification if chosen
  - MUST explain how this leverages LSTM's temporal advantage
  - MUST describe how output is handled

---

## Key Success Factors

**What makes this assignment successful:**

1. **Understand the internal state management** - This is what makes LSTM work for L=1!
   - The network uses its memory (h_t, c_t) to track frequency structure over time
   - Without proper state preservation, the network cannot learn sequential patterns

2. **Proper noise implementation** - Random amplitude/phase at EVERY sample
   - This forces the LSTM to learn frequency structure, not memorize patterns

3. **Generalization** - Test set must show similar performance to training
   - Proves the network learned the underlying frequencies
   - Not just memorizing noise patterns

4. **Clear visualizations** - Show the dramatic difference between:
   - Chaotic noisy input
   - Clean extracted output
   - Perfect ground truth

5. **Understanding the "why"** - Explain in your report:
   - Why L=1 with state preservation works
   - How LSTM learns from sequential data
   - Why random noise per sample is critical

---

## Summary

This assignment tests your understanding of:
- LSTM temporal learning capabilities
- Conditional regression
- Signal processing fundamentals
- Proper evaluation and generalization testing

**The key insight:** By preserving internal state across individual samples (L=1), the LSTM can learn the periodic structure of frequencies despite random noise at each timestep. This demonstrates the power of recurrent networks for sequential data.

**Critical for success:** Proper internal state management when L=1 is the pedagogical focus and the key to making this work!
