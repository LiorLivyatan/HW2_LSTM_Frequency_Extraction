# Agent Communication System

## Overview

This directory contains a structured communication system for tracking agent activities during the LSTM Frequency Extraction project implementation. The system ensures traceability, maintains handoff documentation between phases, and provides visibility into what each agent has accomplished.

---

## Directory Structure

```
agent_communication/
├── README.md                    # This file - system documentation
├── active_agents.json           # Registry of all available agents
├── phase_assignments.json       # Agent-to-phase mappings
│
├── reports/                     # Agent execution reports (organized by agent)
│   ├── signal-validation-expert/
│   ├── lstm-state-debugger/
│   ├── lstm-training-monitor/
│   └── lstm-freq-evaluator/
│
├── handoffs/                    # Inter-phase handoff documents
│   ├── phase1_to_phase2.md
│   ├── phase2_to_phase3.md
│   ├── phase3_to_phase4.md
│   ├── phase4_to_phase5.md
│   └── phase5_to_phase6.md
│
└── logs/                        # Execution logs
    └── agent_activity.log
```

---

## Agent Types

### Built-in System Agents

| Agent Type | Purpose | When to Use |
|------------|---------|-------------|
| **general-purpose** | Multi-step tasks, complex research | Default for most coding tasks |
| **Explore** | Codebase exploration | Quick searches, pattern matching |
| **Plan** | Planning and analysis | Research without execution |

### Custom Project Agents

| Agent Name | Phase | Specialization | When to Invoke |
|------------|-------|----------------|----------------|
| **signal-validation-expert** | Phase 1 | Data generation validation | After implementing/modifying signal generation |
| **lstm-state-debugger** | Phase 3 | LSTM state management | When debugging state preservation, memory leaks |
| **lstm-training-monitor** | Phase 3 | Training convergence | Analyzing training performance, hyperparameter tuning |
| **lstm-freq-evaluator** | Phase 4 | Model evaluation | Interpreting metrics, generalization analysis |

---

## Communication Protocol

### Agent Report Format

Each agent execution creates TWO files in its designated `reports/` subdirectory:

#### 1. Summary JSON (`summary.json`)

Machine-readable execution summary:

```json
{
  "agent_name": "signal-validation-expert",
  "phase": "Phase 1",
  "execution_timestamp": "2025-11-16T10:30:00Z",
  "task_description": "Validate generated train/test datasets",
  "status": "COMPLETED|IN_PROGRESS|FAILED|BLOCKED",
  "findings": {
    "per_sample_randomization": "PASS|FAIL",
    "frequency_content": "PASS|FAIL",
    "seed_separation": "PASS|FAIL",
    "dataset_structure": "PASS|FAIL"
  },
  "critical_issues": [
    "List of blocking issues (empty if none)"
  ],
  "recommendations": [
    "Actionable next steps"
  ],
  "artifacts_created": [
    "List of files created/modified"
  ],
  "next_agent": "Name of next agent to invoke (or 'None')",
  "execution_duration": "Estimated time spent"
}
```

#### 2. Detailed Markdown Report (`YYYY-MM-DD_description.md`)

Human-readable detailed analysis:

```markdown
# Agent Report: [Agent Name]

**Phase**: Phase X
**Date**: YYYY-MM-DD
**Status**: COMPLETED/IN_PROGRESS/FAILED/BLOCKED

---

## Task Summary

[Brief description of what the agent was asked to do]

## Methodology

[How the agent approached the task]

## Findings

### Critical Findings
[Any blocking issues or major discoveries]

### Detailed Analysis
[In-depth analysis with code references, line numbers, etc.]

## Issues Identified

| Issue | Severity | Location | Description |
|-------|----------|----------|-------------|
| ... | HIGH/MEDIUM/LOW | file:line | ... |

## Recommendations

1. **[Priority]** [Specific action item]
   - Rationale: [Why this is needed]
   - Expected outcome: [What this will achieve]

## Artifacts Created/Modified

- `path/to/file1.py` - [Description]
- `path/to/file2.npy` - [Description]

## Handoff Notes

[Information for the next agent or phase]

## Verification Steps

[How to verify the agent's work was successful]

---

**Next Agent**: [Name or "None"]
**Execution Duration**: [Time spent]
```

---

## Phase-to-Agent Assignments

Refer to `phase_assignments.json` for the complete mapping. Quick reference:

| Phase | Description | Primary Agent(s) | Expected Output |
|-------|-------------|------------------|-----------------|
| **Phase 1** | Data Generation | signal-validation-expert | Validated train/test datasets |
| **Phase 2** | Model Architecture | general-purpose | Tested FrequencyLSTM class |
| **Phase 3** | Training Pipeline | lstm-state-debugger + lstm-training-monitor | Trained model with state validation |
| **Phase 4** | Evaluation | lstm-freq-evaluator | Performance metrics analysis |
| **Phase 5** | Visualization | general-purpose | Publication-quality graphs |
| **Phase 6** | Integration | general-purpose | End-to-end pipeline |

---

## Handoff Documents

After completing each phase, create a handoff document in `handoffs/`:

**Template**: `phaseX_to_phaseY.md`

```markdown
# Phase X → Phase Y Handoff

**Date**: YYYY-MM-DD
**Status**: READY|BLOCKED|IN_PROGRESS

---

## Phase X Completion Status

- [ ] All deliverables created
- [ ] Validation passed
- [ ] Tests passing
- [ ] Documentation updated

## Artifacts Delivered

1. **File**: `path/to/artifact`
   - **Purpose**: [What this file does]
   - **Validation**: [How it was validated]

## Known Issues / Technical Debt

[Any issues that weren't blocking but should be addressed]

## Phase Y Prerequisites

- [ ] Prerequisite 1
- [ ] Prerequisite 2

## Notes for Phase Y Team

[Any context, warnings, or recommendations]

---

**Phase X Lead Agent**: [Agent name]
**Phase Y Assignee**: [Agent name]
```

---

## Logging

All agent activities are logged to `logs/agent_activity.log`:

```
[2025-11-16 10:30:15] [INFO] signal-validation-expert: Started validation of train_data.npy
[2025-11-16 10:32:48] [INFO] signal-validation-expert: FFT analysis complete - all 4 frequencies detected
[2025-11-16 10:35:22] [SUCCESS] signal-validation-expert: Validation PASSED - datasets ready for Phase 2
```

---

## Usage Workflow

### 1. Before Starting a Phase

1. Check `phase_assignments.json` for the assigned agent
2. Review any handoff documents from previous phase
3. Prepare task description for the agent

### 2. During Agent Execution

1. Agent creates timestamped report in `reports/[agent-name]/`
2. Agent updates `summary.json` with findings
3. Agent logs activities to `logs/agent_activity.log`

### 3. After Agent Completion

1. Review agent report for critical issues
2. Create handoff document if phase is complete
3. Update `active_agents.json` status if needed
4. Invoke next agent based on recommendations

### 4. Cross-Phase Coordination

- Multiple agents can work on the same phase (e.g., Phase 3)
- Check existing reports before re-running validation
- Use handoff documents to maintain context

---

## Best Practices

### For Claude Code Orchestrator

1. **Read before writing**: Check existing reports before invoking agents
2. **One agent per task**: Don't overload agents with multiple responsibilities
3. **Preserve context**: Include relevant PRD references in agent prompts
4. **Track handoffs**: Always create handoff documents between phases
5. **Log everything**: Update activity log for traceability

### For Report Creation

1. **Be specific**: Include file paths, line numbers, exact error messages
2. **Actionable recommendations**: Each recommendation should be immediately implementable
3. **Evidence-based**: Support findings with data, code snippets, or test results
4. **Status clarity**: Make PASS/FAIL determinations explicit
5. **Next steps**: Always specify what should happen next

### For Handoffs

1. **Complete checklists**: Don't hand off incomplete work
2. **Document blockers**: If blocked, explain exactly what's needed
3. **Validation proof**: Include evidence that deliverables are correct
4. **Context preservation**: Explain decisions that affect future phases

---

## Integration with CLAUDE.md Principles

This communication system enforces the key principles from `CLAUDE.md`:

### 1. L=1 State Preservation (Phase 3)
- **lstm-state-debugger** validates state detachment pattern
- Reports include memory leak checks
- Verification of state flow across 40,000 samples

### 2. Per-Sample Randomization (Phase 1)
- **signal-validation-expert** checks loop-based randomization
- FFT validation confirms frequency content
- Rejects vectorized amplitude/phase generation

### 3. PRD-Driven Development
- All agents reference their phase-specific PRD
- Reports align with PRD success criteria
- Handoffs verify PRD deliverables

### 4. Generalization Validation (Phase 4)
- **lstm-freq-evaluator** checks MSE_test ≈ MSE_train
- Reports include per-frequency analysis
- Validation of Seed #1 vs Seed #2 performance

---

## Troubleshooting

### Agent Report Not Created
- Check if agent completed successfully
- Verify write permissions in `agent_communication/`
- Review `logs/agent_activity.log` for errors

### Missing Handoff Document
- Ensure previous phase is fully complete
- Check phase completion checklist
- Verify all deliverables exist

### Conflicting Agent Recommendations
- Review both reports carefully
- Check timestamps to determine sequence
- Prioritize recommendations from specialized agents over general-purpose

---

## File Naming Conventions

### Reports
- Format: `YYYY-MM-DD_brief-description.md`
- Example: `2025-11-16_phase1_validation.md`

### Handoffs
- Format: `phaseX_to_phaseY.md`
- Example: `phase1_to_phase2.md`

### Logs
- Single file: `agent_activity.log` (append-only)

---

## Version Control

- **Commit**: All agent reports and handoff documents
- **Ignore**: Logs directory (`logs/*.log` in `.gitignore`)
- **Track**: JSON registry files for reproducibility

---

## Maintenance

### Weekly
- Review `agent_activity.log` for patterns
- Archive old reports if directory becomes large

### Per Phase
- Create handoff document
- Update `active_agents.json` status
- Clean up temporary files

### End of Project
- Archive entire `agent_communication/` directory
- Create final summary report
- Document lessons learned for future projects

---

## Contact / Issues

For issues with this communication system:
1. Check this README first
2. Review example reports in `reports/` subdirectories
3. Consult `CLAUDE.md` for project-specific guidance
4. Refer to phase PRDs in `prd/` directory

---

**Last Updated**: 2025-11-16
**System Version**: 1.0
**Project**: HW2_LSTM_Frequency_Extraction
