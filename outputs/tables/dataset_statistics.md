# Dataset Statistics

| Property | Training Set | Test Set |
|----------|--------------|----------|
| Total Samples | 40,000 | 40,000 |
| Samples per Frequency | 10,000 | 10,000 |
| Number of Frequencies | 4 | 4 |
| Frequencies | 1Hz, 3Hz, 5Hz, 7Hz | 1Hz, 3Hz, 5Hz, 7Hz |
| Input Dimension | 5 | 5 |
| Output Dimension | 1 (scalar) | 1 (scalar) |
| Random Seed | 42 | 99 |
| Data Format | `.npy` (NumPy) | `.npy` (NumPy) |

## Dataset Structure

Each row contains:
- **S(t)**: Noisy mixed signal (1 value)
- **C**: One-hot frequency selector (4 values)
- **Target**: Clean target sinusoid (1 value)

**Total row format**: `[S(t), C1, C2, C3, C4, Target]` (6 values)

## Signal Statistics

| Statistic | Training Set | Test Set |
|-----------|--------------|----------|
| Input Mean | -0.000464 | 0.000167 |
| Input Std Dev | 0.355573 | 0.355541 |
| Input Min | -0.846181 | -0.850187 |
| Input Max | 0.876009 | 0.852705 |
| Target Mean | 0.000000 | 0.000000 |
| Target Std Dev | 0.707071 | 0.707071 |
| Target Min | -1.000000 | -1.000000 |
| Target Max | 1.000000 | 1.000000 |
