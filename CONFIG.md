# Configuration Guide

## Configuration Files

- **config.yaml**: Primary configuration file with all hyperparameters
- **.env**: Environment-specific overrides (copy from `.env.example`)

## Parameter Reference

### Data Generation

| Parameter | Default | Description |
|-----------|---------|-------------|
| frequencies | [1, 3, 5, 7] | Target frequencies in Hz |
| sampling_rate | 1000 | Samples per second |
| duration | 10.0 | Signal duration in seconds |
| train_seed | 1 | Random seed for training set |
| test_seed | 2 | Random seed for test set |

### Model Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| input_size | 5 | Input features (signal + one-hot) |
| hidden_size | 128 | LSTM hidden dimension |
| num_layers | 1 | Number of LSTM layers |
| dropout | 0.0 | Dropout between layers |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| learning_rate | 0.0001 | Adam learning rate |
| num_epochs | 100 | Training epochs |
| batch_size | 32 | Samples per batch |
| clip_grad_norm | 1.0 | Gradient clipping threshold |
| device | auto | Compute device |

### Evaluation

| Parameter | Default | Description |
|-----------|---------|-------------|
| generalization_threshold | 0.1 | Max train/test MSE difference (10%) |

### Visualization

| Parameter | Default | Description |
|-----------|---------|-------------|
| comparison_freq_idx | 1 | Frequency index for comparison graph |
| time_window | 1000 | Samples to display |
| dpi | 300 | Graph resolution |

## Environment Variables

Environment variables override config.yaml values. See `.env.example` for available options.

## Configuration Precedence

1. config.yaml values
2. Environment variables (override config.yaml)
3. Command-line arguments
