# Per-Frequency Performance Metrics

Detailed breakdown of MSE for each frequency component.

| Frequency | Training MSE | Test MSE | Absolute Diff | Relative Diff | Generalization |
|-----------|--------------|----------|---------------|---------------|----------------|
| f1 = 1Hz | 0.015071 | 0.015477 | 0.000406 | 2.69% | ✓ Good |
| f2 = 3Hz | 0.041395 | 0.042874 | 0.001479 | 3.57% | ✓ Good |
| f3 = 5Hz | 0.043923 | 0.042711 | 0.001212 | 2.76% | ✓ Good |
| f4 = 7Hz | 0.197415 | 0.196350 | 0.001065 | 0.54% | ✓ Good |

## Summary Statistics

| Statistic | Training MSE | Test MSE |
|-----------|--------------|----------|
| Mean | 0.074451 | 0.074353 |
| Std Dev | 0.071887 | 0.071312 |
| Min | 0.015071 | 0.015477 |
| Max | 0.197415 | 0.196350 |

## Performance Rankings

### Best to Worst (by Test MSE)

| Rank | Frequency | Test MSE |
|------|-----------|----------|
| 1 | f1 = 1Hz | 0.015477 |
| 2 | f3 = 5Hz | 0.042711 |
| 3 | f2 = 3Hz | 0.042874 |
| 4 | f4 = 7Hz | 0.196350 |

## Analysis

**All frequencies show good generalization** (relative difference < 10%)

**Frequencies with High Test MSE** (> 0.01):

- **1Hz**: MSE = 0.015477
- **3Hz**: MSE = 0.042874
- **5Hz**: MSE = 0.042711
- **7Hz**: MSE = 0.196350
