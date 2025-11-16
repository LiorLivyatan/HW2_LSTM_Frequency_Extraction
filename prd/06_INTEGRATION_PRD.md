# Phase 6: Integration & Orchestration PRD

**Phase**: 6 of 6 (Final Phase)
**Priority**: Medium
**Estimated Effort**: 1-2 hours
**Dependencies**: All previous phases (1-5)
**Enables**: Complete end-to-end pipeline

---

## Table of Contents
1. [Objective](#objective)
2. [Requirements](#requirements)
3. [Architecture](#architecture)
4. [Implementation Details](#implementation-details)
5. [Testing Strategy](#testing-strategy)
6. [Deliverables](#deliverables)
7. [Success Criteria](#success-criteria)
8. [Risks and Mitigation](#risks-and-mitigation)

---

## Objective

Create an end-to-end orchestration system that ties all phases together into a single, reproducible pipeline.

### What This Phase Delivers
- **`main.py`**: Single entry point to run entire pipeline
- **Configuration management**: YAML or argparse for hyperparameters
- **CLI interface**: Easy-to-use command-line interface
- **Complete documentation**: README with usage instructions
- **Reproducibility**: Deterministic results with seed control
- **Requirements file**: All dependencies listed

### Why This Phase is Critical
- Makes the project easy to run and reproduce
- Demonstrates professional software engineering
- Enables quick experimentation with different configurations
- Simplifies grading for the instructor

---

## Requirements

### Functional Requirements

#### FR1: End-to-End Pipeline
- [ ] Single command to run entire pipeline:
  ```bash
  python main.py --mode all
  ```
- [ ] Individual phase execution:
  ```bash
  python main.py --mode data  # Run only data generation
  python main.py --mode train # Run only training
  python main.py --mode eval  # Run only evaluation
  python main.py --mode viz   # Run only visualization
  ```

#### FR2: Configuration Management
- [ ] Centralized configuration (YAML or Python dict)
- [ ] Configurable parameters:
  - Data generation seeds
  - Model architecture (hidden_size, num_layers)
  - Training hyperparameters (lr, epochs, batch_size)
  - Device selection (CPU/GPU)
  - Output directories

#### FR3: Command-Line Interface
- [ ] Argument parser (argparse)
- [ ] Help documentation (`--help`)
- [ ] Mode selection
- [ ] Config file override
- [ ] Verbose/quiet modes

#### FR4: Logging
- [ ] Structured logging to console and file
- [ ] Different log levels (INFO, DEBUG, WARNING, ERROR)
- [ ] Timestamp on log entries
- [ ] Progress tracking for each phase

#### FR5: Reproducibility
- [ ] Seed control for all random operations
- [ ] Deterministic PyTorch/NumPy behavior
- [ ] Configuration saved with outputs

### Non-Functional Requirements

#### NFR1: Usability
- Clear error messages
- Progress indicators
- Estimated time remaining
- User-friendly CLI

#### NFR2: Robustness
- Graceful error handling
- Keyboard interrupt handling
- Resume capability (if interrupted)
- Validation of inputs

#### NFR3: Documentation
- Comprehensive README.md
- Installation instructions
- Usage examples
- Troubleshooting guide

---

## Architecture

### Project Structure (Final)

```
HW2/
├── prd/                              # Product Requirements Documents
│   ├── 00_MASTER_PRD.md
│   ├── 01_DATA_GENERATION_PRD.md
│   ├── 02_MODEL_ARCHITECTURE_PRD.md
│   ├── 03_TRAINING_PIPELINE_PRD.md
│   ├── 04_EVALUATION_PRD.md
│   ├── 05_VISUALIZATION_PRD.md
│   └── 06_INTEGRATION_PRD.md
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── data_generation.py            # SignalGenerator
│   ├── dataset.py                    # FrequencyDataset
│   ├── model.py                      # FrequencyLSTM
│   ├── training.py                   # StatefulTrainer
│   ├── evaluation.py                 # Evaluator
│   └── visualization.py              # Visualizer
│
├── tests/                            # Unit tests
│   ├── __init__.py
│   ├── test_data_generation.py
│   ├── test_model.py
│   ├── test_training.py
│   ├── test_evaluation.py
│   └── test_visualization.py
│
├── data/                             # Generated datasets
│   ├── train_data.npy
│   └── test_data.npy
│
├── models/                           # Saved models
│   └── best_model.pth
│
├── outputs/                          # Results
│   ├── graphs/
│   │   ├── frequency_comparison.png
│   │   └── all_frequencies.png
│   ├── metrics.json
│   ├── predictions.npz
│   ├── training_history.json
│   └── run_config.yaml
│
├── logs/                             # Log files
│   └── lstm_extraction.log
│
├── main.py                           # Main orchestration script
├── config.yaml                       # Configuration file
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
├── ASSIGNMENT_REQUIREMENTS.md        # Assignment specs
└── .gitignore                        # Git ignore file
```

### Pipeline Flow

```
START
  │
  ├─→ Parse CLI Arguments
  │   └─→ Load Configuration
  │
  ├─→ [Mode: data or all]
  │   └─→ Phase 1: Generate Data
  │       ├─→ Create train_data.npy (Seed #1)
  │       └─→ Create test_data.npy (Seed #2)
  │
  ├─→ [Mode: train or all]
  │   └─→ Phase 2 & 3: Train Model
  │       ├─→ Initialize FrequencyLSTM
  │       ├─→ Create StatefulTrainer
  │       ├─→ Train with state preservation
  │       └─→ Save best_model.pth
  │
  ├─→ [Mode: eval or all]
  │   └─→ Phase 4: Evaluate
  │       ├─→ Load trained model
  │       ├─→ Run on train set (MSE_train)
  │       ├─→ Run on test set (MSE_test)
  │       ├─→ Check generalization
  │       └─→ Save metrics.json, predictions.npz
  │
  ├─→ [Mode: viz or all]
  │   └─→ Phase 5: Visualize
  │       ├─→ Create Graph 1 (frequency comparison)
  │       ├─→ Create Graph 2 (all frequencies)
  │       └─→ Save PNG files
  │
  └─→ COMPLETE
      └─→ Print summary report
```

---

## Implementation Details

### Configuration File (config.yaml)

```yaml
# LSTM Frequency Extraction Configuration

# Data Generation
data:
  frequencies: [1, 3, 5, 7]  # Hz
  sampling_rate: 1000         # Hz
  duration: 10.0              # seconds
  train_seed: 1
  test_seed: 2
  data_dir: "data"

# Model Architecture
model:
  input_size: 5
  hidden_size: 64
  num_layers: 1
  dropout: 0.0

# Training
training:
  learning_rate: 0.001
  num_epochs: 50
  batch_size: 1               # CRITICAL: Must be 1 for L=1
  clip_grad_norm: 1.0
  device: "auto"              # auto, cpu, or cuda
  save_dir: "models"

# Evaluation
evaluation:
  generalization_threshold: 0.1  # 10%

# Visualization
visualization:
  comparison_freq_idx: 1      # f₂ = 3Hz
  time_window: 1000           # First 1 second
  dpi: 300
  output_dir: "outputs/graphs"

# Paths
paths:
  train_data: "data/train_data.npy"
  test_data: "data/test_data.npy"
  model_checkpoint: "models/best_model.pth"
  metrics: "outputs/metrics.json"
  predictions: "outputs/predictions.npz"
  training_history: "outputs/training_history.json"
  log_file: "logs/lstm_extraction.log"
```

### Main Script (main.py)

```python
import argparse
import logging
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Import all components
from src.data_generation import SignalGenerator
from src.dataset import FrequencyDataset
from src.model import FrequencyLSTM
from src.training import StatefulTrainer
from src.evaluation import Evaluator
from src.visualization import Visualizer

# Set up logging
def setup_logging(log_file: str = 'logs/lstm_extraction.log', verbose: bool = False):
    """Configure logging."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: dict, output_path: str):
    """Save configuration used for this run."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def phase_data_generation(config: dict):
    """Phase 1: Generate datasets."""
    logging.info("=" * 60)
    logging.info("PHASE 1: Data Generation")
    logging.info("=" * 60)

    data_config = config['data']

    # Training set
    logging.info(f"Generating training set (Seed #{data_config['train_seed']})...")
    train_gen = SignalGenerator(
        frequencies=data_config['frequencies'],
        fs=data_config['sampling_rate'],
        duration=data_config['duration'],
        seed=data_config['train_seed']
    )
    train_gen.save_dataset(config['paths']['train_data'])

    # Test set
    logging.info(f"Generating test set (Seed #{data_config['test_seed']})...")
    test_gen = SignalGenerator(
        frequencies=data_config['frequencies'],
        fs=data_config['sampling_rate'],
        duration=data_config['duration'],
        seed=data_config['test_seed']
    )
    test_gen.save_dataset(config['paths']['test_data'])

    logging.info("✓ Data generation complete")

def phase_training(config: dict):
    """Phase 2 & 3: Model creation and training."""
    logging.info("=" * 60)
    logging.info("PHASE 2 & 3: Model Training")
    logging.info("=" * 60)

    # Device selection
    device_config = config['training']['device']
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)

    logging.info(f"Using device: {device}")

    # Create model
    model = FrequencyLSTM(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    )

    logging.info(model.get_model_summary())

    # Load data
    train_dataset = FrequencyDataset(config['paths']['train_data'])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False  # CRITICAL for L=1
    )

    # Loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )

    # Create trainer
    trainer = StatefulTrainer(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        clip_grad_norm=config['training']['clip_grad_norm']
    )

    # Train
    history = trainer.train(
        num_epochs=config['training']['num_epochs'],
        save_dir=config['training']['save_dir'],
        save_best=True
    )

    # Save training history
    trainer.save_history(config['paths']['training_history'])

    logging.info("✓ Training complete")

def phase_evaluation(config: dict):
    """Phase 4: Evaluation."""
    logging.info("=" * 60)
    logging.info("PHASE 4: Evaluation")
    logging.info("=" * 60)

    # Device selection
    device_config = config['training']['device']
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)

    # Load model
    model = FrequencyLSTM(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers']
    )

    checkpoint = torch.load(config['paths']['model_checkpoint'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    logging.info(f"Loaded model from epoch {checkpoint['epoch']}")

    # Load datasets
    train_dataset = FrequencyDataset(config['paths']['train_data'])
    test_dataset = FrequencyDataset(config['paths']['test_data'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Create evaluator
    evaluator = Evaluator(model, device=device)

    # Evaluate
    results = evaluator.evaluate_all(train_loader, test_loader)

    # Save results
    evaluator.save_metrics(results, config['paths']['metrics'])
    evaluator.save_predictions(results, config['paths']['predictions'])

    logging.info("✓ Evaluation complete")

def phase_visualization(config: dict):
    """Phase 5: Visualization."""
    logging.info("=" * 60)
    logging.info("PHASE 5: Visualization")
    logging.info("=" * 60)

    # Create visualizer
    visualizer = Visualizer(
        predictions_path=config['paths']['predictions'],
        data_path=config['paths']['test_data']
    )

    # Create graphs
    visualizer.create_all_visualizations(
        output_dir=config['visualization']['output_dir']
    )

    logging.info("✓ Visualization complete")

def main():
    """Main orchestration function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='LSTM Frequency Extraction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run entire pipeline
  python main.py --mode all

  # Run only data generation
  python main.py --mode data

  # Run training with custom config
  python main.py --mode train --config my_config.yaml

  # Run in verbose mode
  python main.py --mode all --verbose
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['all', 'data', 'train', 'eval', 'viz'],
        default='all',
        help='Which phase(s) to run'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    setup_logging(config['paths']['log_file'], args.verbose)

    # Print header
    logging.info("=" * 60)
    logging.info("LSTM Frequency Extraction System")
    logging.info("=" * 60)
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Configuration: {args.config}")
    logging.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 60)

    # Save configuration for this run
    save_config(config, 'outputs/run_config.yaml')

    # Run phases based on mode
    try:
        if args.mode in ['all', 'data']:
            phase_data_generation(config)

        if args.mode in ['all', 'train']:
            set_seeds(config['data']['train_seed'])  # Reproducibility
            phase_training(config)

        if args.mode in ['all', 'eval']:
            phase_evaluation(config)

        if args.mode in ['all', 'viz']:
            phase_visualization(config)

        # Print completion
        logging.info("=" * 60)
        logging.info("PIPELINE COMPLETE")
        logging.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("=" * 60)

    except KeyboardInterrupt:
        logging.warning("\n\nInterrupted by user")
        return 1

    except Exception as e:
        logging.error(f"\n\nError: {e}", exc_info=True)
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
```

### Requirements File (requirements.txt)

```txt
# Core Dependencies
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0

# Data and Configuration
pyyaml>=6.0

# Progress and Utilities
tqdm>=4.65.0

# Testing
pytest>=7.3.0

# Optional: For better performance
scipy>=1.10.0  # For FFT validation in tests

# Development
black>=23.0.0  # Code formatting
flake8>=6.0.0  # Linting
mypy>=1.3.0    # Type checking
```

---

## Testing Strategy

### Integration Tests

#### Test 1: End-to-End Pipeline
```python
def test_end_to_end_pipeline(tmp_path):
    """Test entire pipeline runs successfully."""
    # Create temporary config with small epochs for testing
    test_config = {
        'data': {
            'frequencies': [1, 3, 5, 7],
            'sampling_rate': 1000,
            'duration': 1.0,  # 1 second only for testing
            'train_seed': 1,
            'test_seed': 2,
            'data_dir': str(tmp_path / 'data')
        },
        'training': {
            'num_epochs': 2,  # Just 2 epochs for testing
            # ... other params
        },
        # ... rest of config
    }

    # Run each phase
    phase_data_generation(test_config)
    phase_training(test_config)
    phase_evaluation(test_config)
    phase_visualization(test_config)

    # Verify outputs exist
    assert (tmp_path / 'data' / 'train_data.npy').exists()
    assert (tmp_path / 'models' / 'best_model.pth').exists()
    assert (tmp_path / 'outputs' / 'metrics.json').exists()
    assert (tmp_path / 'outputs' / 'graphs' / 'frequency_comparison.png').exists()
```

---

## Deliverables

### Code Files
- [ ] `main.py` - Main orchestration script
- [ ] `config.yaml` - Configuration file
- [ ] `requirements.txt` - Dependencies list
- [ ] `.gitignore` - Git ignore rules

### Documentation
- [ ] `README.md` - Project documentation
- [ ] Usage examples
- [ ] Installation guide
- [ ] Troubleshooting guide

### Meta Files
- [ ] `outputs/run_config.yaml` - Configuration used for run
- [ ] `logs/lstm_extraction.log` - Execution log

---

## Success Criteria

### Functionality
- [ ] `python main.py --mode all` runs successfully
- [ ] All phases execute without errors
- [ ] Outputs are created in correct locations
- [ ] Results are reproducible with same seeds

### Usability
- [ ] CLI is intuitive
- [ ] Help documentation is clear
- [ ] Error messages are helpful
- [ ] Progress is visible

### Documentation
- [ ] README explains how to run
- [ ] Installation is straightforward
- [ ] Examples are provided

---

## Risks and Mitigation

### Risk 1: Dependency Conflicts
**Risk**: Different library versions cause errors

**Mitigation**:
- Pin versions in requirements.txt
- Test on clean virtual environment
- Provide Docker alternative (optional)

### Risk 2: Path Issues
**Risk**: Hardcoded paths break on different systems

**Mitigation**:
- Use pathlib for cross-platform paths
- Make all paths configurable
- Test on Windows/Mac/Linux

---

## Dependencies

### Depends On
- All previous phases (1-5)

---

## Estimated Effort

| Activity | Time Estimate |
|----------|---------------|
| Implement main.py | 0.5-1 hour |
| Create config.yaml | 0.25 hour |
| Write README.md | 0.5 hour |
| Testing | 0.25 hour |
| **Total** | **1.5-2 hours** |

---

## Next Steps

After completing Phase 6:
1. Run full pipeline end-to-end
2. Verify all outputs
3. Review README for clarity
4. **Assignment ready for submission!**

---

**Status**: Ready for Implementation
**Last Updated**: 2025-11-16
