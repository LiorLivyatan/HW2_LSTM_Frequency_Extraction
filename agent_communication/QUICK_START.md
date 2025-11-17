# Agent Communication System - Quick Start Guide

## Overview

This system tracks agent activities, maintains handoff documentation, and ensures traceability across all 6 phases of the LSTM Frequency Extraction project.

---

## Quick Reference: Available Agents

### Custom Project Agents

| Agent | Phase | Use When | Key Validation |
|-------|-------|----------|----------------|
| **signal-validation-expert** | 1 | After data generation | Per-sample randomization, FFT analysis |
| **lstm-state-debugger** | 3 | Before/during training | State detachment, memory leaks |
| **lstm-training-monitor** | 3 | During/after training | Loss curves, convergence |
| **lstm-freq-evaluator** | 4 | After training | MSE analysis, generalization |

### System Agents

| Agent | Use When |
|-------|----------|
| **general-purpose** | Default for most coding tasks |
| **Explore** | Quick codebase searches, file patterns |
| **Plan** | Research without execution |

---

## How to Use This System

### 1. Before Starting a Phase

```bash
# Check phase assignment
cat agent_communication/phase_assignments.json | grep -A 20 "phase_X"

# Review PRD
cat prd/0X_*_PRD.md

# Check for previous handoff
cat agent_communication/handoffs/phase(X-1)_to_phaseX.md
```

### 2. During Agent Execution

Agents should create:

1. **JSON Summary**: `reports/[agent-name]/summary.json`
2. **Markdown Report**: `reports/[agent-name]/YYYY-MM-DD_description.md`
3. **Log Entry**: Append to `logs/agent_activity.log`

**Templates available**:
- `reports/signal-validation-expert/TEMPLATE_*.json|.md`
- `reports/lstm-state-debugger/TEMPLATE_*.json`
- `handoffs/TEMPLATE_phase_handoff.md`

### 3. After Phase Completion

```bash
# Create handoff document
cp agent_communication/handoffs/TEMPLATE_phase_handoff.md \
   agent_communication/handoffs/phaseX_to_phaseY.md

# Update phase status
# Edit: agent_communication/phase_assignments.json
# Change: "phase_X": "not_started" → "completed"
```

---

## Phase-Specific Workflows

### Phase 1: Data Generation

```
1. Implement src/data_generation.py
2. Generate data/train_data.npy and data/test_data.npy
3. Invoke: signal-validation-expert
4. Review: reports/signal-validation-expert/[date]_validation.md
5. If PASS: Create handoff to Phase 2
6. If FAIL: Fix issues and re-validate
```

**Critical Checks**:
- ✓ Per-sample randomization (loop-based)
- ✓ FFT shows 1Hz, 3Hz, 5Hz, 7Hz
- ✓ Train/test seeds differ

### Phase 3: Training Pipeline (CRITICAL)

```
1. Implement src/training.py
2. Invoke: lstm-state-debugger (BEFORE training!)
3. Review: reports/lstm-state-debugger/[date]_state_check.md
4. Fix any state management issues
5. Start training (1 epoch)
6. Invoke: lstm-training-monitor
7. Review: reports/lstm-training-monitor/[date]_training.md
8. Apply recommendations
9. Complete training
10. Re-validate with both agents
```

**Critical Checks**:
- ✓ State detachment after backward()
- ✓ States preserved between samples
- ✓ No memory leaks
- ✓ Loss decreasing

### Phase 4: Evaluation

```
1. Implement src/evaluation.py
2. Calculate MSE_train and MSE_test
3. Invoke: lstm-freq-evaluator
4. Review: reports/lstm-freq-evaluator/[date]_evaluation.md
5. Check: Generalization ratio 0.90 ≤ (MSE_test/MSE_train) ≤ 1.10
6. If FAIL: Return to Phase 3 with recommendations
7. If PASS: Create handoff to Phase 5
```

**Critical Checks**:
- ✓ MSE_test ≈ MSE_train (within 10%)
- ✓ Overall MSE < 0.01
- ✓ All frequencies extracted

---

## Agent Invocation Examples

### Example 1: Validate Data Generation

**Scenario**: Just generated datasets, need validation

**Command** (conceptual):
```
Invoke: signal-validation-expert
Task: "Validate generated datasets data/train_data.npy and data/test_data.npy.
       Check per-sample randomization, FFT analysis, and seed separation."
```

**Expected Output**:
- `reports/signal-validation-expert/2025-11-16_phase1_validation.md`
- `reports/signal-validation-expert/summary.json`
- Log entry in `logs/agent_activity.log`

### Example 2: Debug State Management

**Scenario**: Training crashes with OOM after 5000 samples

**Command** (conceptual):
```
Invoke: lstm-state-debugger
Task: "Analyze state management in src/training.py:150-200.
       Memory consumption growing during training.
       Suspect missing state detachment."
```

**Expected Output**:
- Identifies missing `.detach()` calls
- Provides exact line numbers
- Shows corrected code
- Explains memory implications

### Example 3: Analyze Training Performance

**Scenario**: Loss plateau at 0.05 after epoch 20

**Command** (conceptual):
```
Invoke: lstm-training-monitor
Task: "Training plateaued at epoch 20 with loss=0.05.
       Current: lr=0.001, hidden_size=64, batch_size=1.
       Need hyperparameter tuning recommendations."
```

**Expected Output**:
- Loss curve analysis
- Specific recommendations (e.g., "reduce lr to 0.0005")
- Expected improvement estimates

---

## File Naming Conventions

### Agent Reports

**Format**: `YYYY-MM-DD_brief-description.md`

**Examples**:
- `2025-11-16_phase1_validation.md`
- `2025-11-16_state_management_check.md`
- `2025-11-16_training_epoch10_analysis.md`
- `2025-11-16_final_evaluation.md`

### Handoff Documents

**Format**: `phaseX_to_phaseY.md`

**Examples**:
- `phase1_to_phase2.md`
- `phase2_to_phase3.md`
- `phase3_to_phase4.md`

---

## Common Workflows

### Workflow A: "Happy Path" - Phase Completion

```
1. Implement phase code
2. Invoke primary agent for validation
3. Agent reports PASS
4. Create handoff document
5. Update phase_assignments.json status to "completed"
6. Move to next phase
```

### Workflow B: "Issues Found" - Iterative Fix

```
1. Implement phase code
2. Invoke primary agent for validation
3. Agent reports FAIL with issues
4. Review agent report for specific fixes
5. Apply fixes
6. Re-invoke agent
7. Repeat until PASS
8. Create handoff document
```

### Workflow C: "Phase 3 Multi-Agent" - Training

```
1. Implement training code
2. Invoke lstm-state-debugger (validate setup)
3. Fix any state issues
4. Run 1 training epoch
5. Invoke lstm-training-monitor (check convergence)
6. Apply hyperparameter recommendations
7. Complete training
8. Re-invoke both agents for final validation
9. Create handoff document
```

---

## Troubleshooting

### Q: Agent report not created

**A**: Check:
1. Agent completed successfully?
2. Write permissions in `agent_communication/`?
3. Review `logs/agent_activity.log` for errors

### Q: Missing handoff from previous phase

**A**:
1. Check `handoffs/` directory
2. If missing, create from template
3. Review previous phase agent reports to fill in details

### Q: Conflicting agent recommendations

**A**:
1. Check report timestamps (latest = most relevant)
2. Prioritize specialized agents over general-purpose
3. For Phase 3: lstm-state-debugger takes priority on state issues

---

## Integration with CLAUDE.md

This communication system enforces CLAUDE.md principles:

| CLAUDE.md Principle | Enforced By |
|---------------------|-------------|
| L=1 State Preservation | lstm-state-debugger mandatory validation |
| Per-Sample Randomization | signal-validation-expert FFT checks |
| PRD-Driven Development | All agents reference phase PRDs |
| Generalization Validation | lstm-freq-evaluator threshold checks |

---

## Directory Structure

```
agent_communication/
├── README.md                    # Full documentation
├── QUICK_START.md               # This file
├── active_agents.json           # Agent registry
├── phase_assignments.json       # Phase mappings
│
├── reports/                     # Agent reports
│   ├── signal-validation-expert/
│   │   ├── TEMPLATE_*.json|.md
│   │   └── [dated reports]
│   ├── lstm-state-debugger/
│   ├── lstm-training-monitor/
│   └── lstm-freq-evaluator/
│
├── handoffs/                    # Phase handoffs
│   ├── TEMPLATE_phase_handoff.md
│   └── [actual handoffs]
│
└── logs/                        # Activity logs
    └── agent_activity.log
```

---

## Key Files to Review

1. **Start Here**: `README.md` - Full system documentation
2. **Agent Details**: `active_agents.json` - All agent capabilities
3. **Phase Planning**: `phase_assignments.json` - What to use when
4. **Templates**: `reports/*/TEMPLATE_*` - Report formats

---

## Success Metrics

Track these across phases:

| Metric | Target | Check |
|--------|--------|-------|
| Phase 1 validation | All PASS | signal-validation-expert report |
| Phase 3 state mgmt | No memory leaks | lstm-state-debugger report |
| Phase 3 training | Loss < 0.01 | lstm-training-monitor report |
| Phase 4 generalization | Ratio 0.9-1.1 | lstm-freq-evaluator report |

---

## Tips for Success

1. **Always read the PRD first** before invoking agents
2. **Use lstm-state-debugger BEFORE training** to catch issues early
3. **Create handoffs immediately** after phase completion
4. **Log everything** to maintain traceability
5. **Review templates** before creating reports

---

**Need Help?**
- Full docs: `agent_communication/README.md`
- PRD reference: `prd/00_MASTER_PRD.md`
- Project guide: `CLAUDE.md`

---

**Last Updated**: 2025-11-16
**Version**: 1.0
