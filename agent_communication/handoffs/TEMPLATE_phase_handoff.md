# Phase X → Phase Y Handoff

**Date**: YYYY-MM-DD
**From Phase**: Phase X - [Phase Name]
**To Phase**: Phase Y - [Phase Name]
**Status**: READY|BLOCKED|IN_PROGRESS

---

## Phase X Completion Checklist

- [ ] All deliverables created and validated
- [ ] All tests passing
- [ ] Agent validation reports reviewed
- [ ] Critical issues resolved
- [ ] Documentation updated
- [ ] Code committed to git

**Completion Status**: XX% complete

---

## Artifacts Delivered

### 1. [Artifact Name]

- **File**: `path/to/file`
- **Purpose**: What this file does and why it's needed for Phase Y
- **Validation**: How it was validated (agent report, tests, manual review)
- **Status**: ✓ READY | ⚠ ISSUES | ✗ BLOCKED

### 2. [Additional artifacts following same format]

---

## Agent Reports Summary

### [Agent Name] Report

- **Date**: YYYY-MM-DD
- **Status**: PASS|FAIL
- **Report Location**: `agent_communication/reports/[agent-name]/YYYY-MM-DD_*.md`
- **Key Findings**:
  - [Summary of critical findings]
  - [Any warnings or notes]
- **Critical Issues**: [None|List of issues]

### [Additional agent reports if applicable]

---

## Known Issues / Technical Debt

### Issue 1: [Description]

- **Severity**: HIGH|MEDIUM|LOW
- **Location**: file.py:line
- **Impact on Phase Y**: [How this affects next phase]
- **Workaround**: [If applicable]
- **Resolution Plan**: [When/how this will be fixed]

### [Additional issues]

**Note**: If no issues, state "No known issues - clean handoff"

---

## Phase Y Prerequisites

Check that these prerequisites are met before starting Phase Y:

- [ ] **Prerequisite 1**: [Specific requirement]
  - Status: ✓ MET | ✗ NOT MET
  - Evidence: [How verified]

- [ ] **Prerequisite 2**: [Specific requirement]
  - Status: ✓ MET | ✗ NOT MET
  - Evidence: [How verified]

- [ ] **Prerequisite 3**: [Additional as needed]

**All Prerequisites Met**: YES|NO

---

## Critical Information for Phase Y Team

### What Phase Y Needs to Know

1. **[Critical Point 1]**
   - Context: [Background]
   - Implication: [What this means for Phase Y]
   - Action: [What Phase Y should do]

2. **[Critical Point 2]**
   - [Same format]

### Assumptions & Constraints

- **Assumption 1**: [What was assumed in Phase X that Phase Y should maintain]
- **Constraint 1**: [Technical constraints Phase Y must work within]

### Warnings & Cautions

- ⚠ **Warning 1**: [Potential pitfall Phase Y should avoid]
- ⚠ **Warning 2**: [Additional warnings]

---

## Configuration & Hyperparameters

**Key Settings from Phase X**:

```yaml
# Settings that Phase Y should be aware of
setting1: value1
setting2: value2
```

**Recommendations for Phase Y**:
- [Any suggested configuration changes]
- [Hyperparameters that may need tuning]

---

## Testing & Validation Notes

### Tests Performed in Phase X

1. **[Test Type]**: [What was tested and results]
2. **[Test Type]**: [What was tested and results]

### Recommended Tests for Phase Y

1. **[Test Type]**: [What Phase Y should test to ensure integration]
2. **[Test Type]**: [Additional recommendations]

---

## Code Quality Notes

**Code Review Status**: REVIEWED|NOT REVIEWED

**Code Style**: Follows project standards|Has issues

**Documentation**: COMPLETE|INCOMPLETE

**Test Coverage**: XX%

**Notes**:
- [Any code quality observations]
- [Areas that need improvement]

---

## Timeline & Effort

**Phase X Actual Duration**: X hours (estimated: Y hours)

**Variance Analysis**:
- [Why actual differed from estimate if applicable]
- [Lessons learned]

**Phase Y Estimated Duration**: X hours (from PRD)

**Recommended Adjustments**:
- [Any timeline adjustments based on Phase X experience]

---

## Dependencies Resolved

List dependencies from Phase X that are now resolved:

- ✓ **Dependency 1**: [Description] - Resolved via [artifact/action]
- ✓ **Dependency 2**: [Description] - Resolved via [artifact/action]

---

## File Structure After Phase X

```
project/
├── src/
│   ├── [files created in Phase X]
├── data/
│   ├── [data files created in Phase X]
├── [other directories]
```

**New Files**: X files added
**Modified Files**: Y files modified
**Deleted Files**: Z files removed

---

## Next Steps for Phase Y

### Immediate Actions

1. **[Action 1]**: [What to do first]
   - Priority: HIGH|MEDIUM|LOW
   - Estimated time: X hours

2. **[Action 2]**: [Second action]
   - Priority: HIGH|MEDIUM|LOW
   - Estimated time: X hours

### Recommended Sequence

1. [Step 1]
2. [Step 2]
3. [Step 3]

---

## Questions & Clarifications

**Open Questions for Phase Y**:
1. [Question 1 that Phase Y team should consider]
2. [Question 2]

**Clarifications Needed**:
- [Any ambiguities that should be resolved before starting Phase Y]

---

## Communication Log

**Phase X Lead Agent**: [agent-name]
**Phase Y Assignee**: [agent-name]

**Handoff Discussion Notes**:
- [Date] - [Summary of discussion point]
- [Date] - [Additional notes]

---

## Sign-Off

**Phase X Completion Verified By**: [agent-name]
**Date**: YYYY-MM-DD

**Phase Y Readiness Confirmed By**: [agent-name]
**Date**: YYYY-MM-DD

**Status**: READY TO PROCEED|BLOCKED|NEEDS REVIEW

---

## Appendix

### References

- **PRD Phase X**: `prd/0X_*_PRD.md`
- **PRD Phase Y**: `prd/0Y_*_PRD.md`
- **Agent Reports**: `agent_communication/reports/[agent-name]/`

### Related Handoffs

- Previous: [phase(X-1)_to_phaseX.md]
- Current: This document
- Next: [phaseY_to_phase(Y+1).md]

---

**Document Version**: 1.0
**Last Updated**: YYYY-MM-DD HH:MM:SS
