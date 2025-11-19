# Experiment Results

Generated: 2025-11-19 21:42:12


## Summary Table

| ID | Name | Hidden | Layers | LR | Epochs | Train MSE | Test MSE | Gen. Diff | 7Hz MSE | Time (s) |
|----|----|--------|--------|------|--------|-----------|----------|-----------|---------|----------|
| 1 | Baseline | 128 | 1 | 0.0001 | 20 | 0.2296 | 0.2293 | 0.11% | 0.0297 | 24.3 |
| 2 | Small Hidden | 64 | 1 | 0.0001 | 20 | 0.1601 | 0.1593 | 0.56% | 0.1960 | 12.1 |
| 3 | Large Hidden | 256 | 1 | 0.0001 | 20 | 0.1849 | 0.1853 | 0.23% | 0.1291 | 32.2 |
| 4 | 2-Layer LSTM | 128 | 2 | 0.0001 | 20 | 0.1493 | 0.1491 | 0.16% | 0.0376 | 41.7 |
| 5 | 3-Layer LSTM | 128 | 3 | 0.0001 | 20 | 0.2459 | 0.2457 | 0.07% | 0.2914 | 65.3 |
| 6 | Higher LR | 128 | 1 | 0.001 | 20 | 0.0261 | 0.0254 | 2.59% | 0.0086 | 21.4 |
| 7 | Lower LR | 128 | 1 | 1e-05 | 20 | 0.4126 | 0.4126 | 0.01% | 0.4718 | 20.4 |
| 8 | Large + 2-Layer | 256 | 2 | 0.0001 | 20 | 0.2596 | 0.2602 | 0.22% | 0.0660 | 67.9 |
| 9 | Optimal Candidate | 256 | 2 | 0.001 | 20 | 0.6878 | 0.6877 | 0.02% | 0.0856 | 67.7 |
| 10 | Large Batch | 128 | 1 | 0.0001 | 20 | 0.1909 | 0.1909 | 0.01% | 0.2515 | 12.1 |

## Best Results

**Best Test MSE**: Experiment 6 (Higher LR) - MSE = 0.025408

**Best 7Hz MSE**: Experiment 6 (Higher LR) - MSE = 0.008589

**Best Generalization**: Experiment 10 (Large Batch) - Diff = 0.0052%

## Per-Frequency Test MSE

| ID | Name | 1Hz | 3Hz | 5Hz | 7Hz |
|----|------|-----|-----|-----|-----|
| 1 | Baseline | 0.1783 | 0.4974 | 0.2120 | 0.0297 |
| 2 | Small Hidden | 0.0346 | 0.2225 | 0.1839 | 0.1960 |
| 3 | Large Hidden | 0.0397 | 0.1610 | 0.4116 | 0.1291 |
| 4 | 2-Layer LSTM | 0.0606 | 0.2232 | 0.2749 | 0.0376 |
| 5 | 3-Layer LSTM | 0.0750 | 0.1455 | 0.4710 | 0.2914 |
| 6 | Higher LR | 0.0301 | 0.0385 | 0.0245 | 0.0086 |
| 7 | Lower LR | 0.3070 | 0.4156 | 0.4560 | 0.4718 |
| 8 | Large + 2-Layer | 0.1966 | 0.3570 | 0.4209 | 0.0660 |
| 9 | Optimal Candidate | 0.8682 | 0.8861 | 0.9109 | 0.0856 |
| 10 | Large Batch | 0.0947 | 0.1157 | 0.3019 | 0.2515 |

## Analysis

### Hidden Size Impact

- **Hidden=64**: Avg Test MSE = 0.1593
- **Hidden=128**: Avg Test MSE = 0.2088
- **Hidden=256**: Avg Test MSE = 0.3777

### Number of Layers Impact

- **Layers=1**: Avg Test MSE = 0.2005
- **Layers=2**: Avg Test MSE = 0.3656
- **Layers=3**: Avg Test MSE = 0.2457
