"""
LSTM Frequency Extraction - Main Orchestration Script

This script provides a unified entry point for running the entire pipeline
or individual phases. It handles configuration, logging, and orchestrates
all components.

Usage:
    # Run entire pipeline
    python main.py --mode all

    # Run individual phases
    python main.py --mode data   # Data generation only
    python main.py --mode train  # Training only
    python main.py --mode eval   # Evaluation only
    python main.py --mode viz    # Visualization only

    # Custom configuration
    python main.py --mode all --config my_config.yaml

    # Verbose logging
    python main.py --mode all --verbose

Reference: prd/06_INTEGRATION_PRD.md
"""

import argparse
import logging
import yaml
import torch
import numpy as np
import os
from pathlib import Path
from datetime import datetime

# Optional: load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use system environment variables only

# Import all pipeline components
from src.data_generation import SignalGenerator
from src.dataset import FrequencyDataset
from src.model import FrequencyLSTM
from src.training import StatefulTrainer
from src.evaluation import Evaluator
from src.visualization import Visualizer


def setup_logging(log_file: str = 'logs/lstm_extraction.log', verbose: bool = False):
    """
    Configure logging for the pipeline.

    Args:
        log_file: Path to log file
        verbose: Enable DEBUG level logging
    """
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
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def apply_environment_overrides(config: dict) -> dict:
    """
    Apply environment variable overrides to configuration.

    Environment variables take precedence over config.yaml values.
    See .env.example for available variables.

    Args:
        config: Configuration dictionary from YAML file

    Returns:
        dict: Configuration with environment overrides applied
    """
    # Device configuration
    if os.getenv('LSTM_DEVICE'):
        config['training']['device'] = os.getenv('LSTM_DEVICE')

    # Directory paths
    if os.getenv('LSTM_DATA_DIR'):
        config['data']['data_dir'] = os.getenv('LSTM_DATA_DIR')
        # Update related paths
        data_dir = os.getenv('LSTM_DATA_DIR')
        config['paths']['train_data'] = f"{data_dir}/train_data.npy"
        config['paths']['test_data'] = f"{data_dir}/test_data.npy"

    if os.getenv('LSTM_MODEL_DIR'):
        config['training']['save_dir'] = os.getenv('LSTM_MODEL_DIR')
        model_dir = os.getenv('LSTM_MODEL_DIR')
        config['paths']['model_checkpoint'] = f"{model_dir}/best_model.pth"
        config['paths']['training_history'] = f"{model_dir}/training_history.json"

    if os.getenv('LSTM_OUTPUT_DIR'):
        output_dir = os.getenv('LSTM_OUTPUT_DIR')
        config['paths']['metrics'] = f"{output_dir}/metrics.json"
        config['paths']['predictions'] = f"{output_dir}/predictions.npz"
        config['visualization']['output_dir'] = f"{output_dir}/graphs"

    if os.getenv('LSTM_LOG_DIR'):
        log_dir = os.getenv('LSTM_LOG_DIR')
        config['paths']['log_file'] = f"{log_dir}/lstm_extraction.log"

    # Training configuration overrides
    if os.getenv('LSTM_NUM_EPOCHS'):
        config['training']['num_epochs'] = int(os.getenv('LSTM_NUM_EPOCHS'))

    if os.getenv('LSTM_BATCH_SIZE'):
        config['training']['batch_size'] = int(os.getenv('LSTM_BATCH_SIZE'))

    if os.getenv('LSTM_LEARNING_RATE'):
        config['training']['learning_rate'] = float(os.getenv('LSTM_LEARNING_RATE'))

    # Model configuration overrides
    if os.getenv('LSTM_HIDDEN_SIZE'):
        config['model']['hidden_size'] = int(os.getenv('LSTM_HIDDEN_SIZE'))

    if os.getenv('LSTM_NUM_LAYERS'):
        config['model']['num_layers'] = int(os.getenv('LSTM_NUM_LAYERS'))

    # Data generation overrides
    if os.getenv('LSTM_TRAIN_SEED'):
        config['data']['train_seed'] = int(os.getenv('LSTM_TRAIN_SEED'))

    if os.getenv('LSTM_TEST_SEED'):
        config['data']['test_seed'] = int(os.getenv('LSTM_TEST_SEED'))

    return config


def save_config(config: dict, output_path: str):
    """
    Save configuration used for this run.

    Args:
        config: Configuration dictionary
        output_path: Where to save the config
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def set_seeds(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def phase_data_generation(config: dict):
    """
    Phase 1: Generate datasets.

    Creates training and test datasets with different random seeds.

    Args:
        config: Configuration dictionary
    """
    logging.info("=" * 70)
    logging.info("PHASE 1: Data Generation")
    logging.info("=" * 70)

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

    logging.info("✓ Data generation complete\n")


def phase_training(config: dict):
    """
    Phase 2 & 3: Model creation and training.

    Builds the FrequencyLSTM model and trains it using StatefulTrainer
    with the CRITICAL L=1 state preservation pattern.

    Args:
        config: Configuration dictionary
    """
    logging.info("=" * 70)
    logging.info("PHASE 2 & 3: Model Training")
    logging.info("=" * 70)

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
        shuffle=False,  # CRITICAL for L=1 state preservation
        num_workers=0
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

    logging.info("✓ Training complete\n")


def phase_evaluation(config: dict):
    """
    Phase 4: Evaluation.

    Evaluates the trained model on both training and test sets,
    calculates MSE metrics, and checks generalization.

    Args:
        config: Configuration dictionary
    """
    logging.info("=" * 70)
    logging.info("PHASE 4: Evaluation")
    logging.info("=" * 70)

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

    # Use same batch size as training for consistency
    batch_size = config['training']['batch_size']

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Create evaluator
    evaluator = Evaluator(model, device=device)

    # Evaluate
    results = evaluator.evaluate_all(train_loader, test_loader)

    # Save results
    evaluator.save_metrics(results, config['paths']['metrics'])
    evaluator.save_predictions(results, config['paths']['predictions'])

    logging.info("✓ Evaluation complete\n")


def phase_visualization(config: dict):
    """
    Phase 5: Visualization.

    Creates publication-quality graphs demonstrating the LSTM's
    frequency extraction capability.

    Args:
        config: Configuration dictionary
    """
    logging.info("=" * 70)
    logging.info("PHASE 5: Visualization")
    logging.info("=" * 70)

    viz_config = config['visualization']

    # Create visualizer
    visualizer = Visualizer(
        predictions_path=config['paths']['predictions'],
        data_path=config['paths']['test_data'],
        training_history_path=config['paths']['training_history'],
        metrics_path=config['paths']['metrics'],
        train_data_path=config['paths']['train_data']
    )

    # Create graphs
    visualizer.create_all_visualizations(
        output_dir=viz_config['output_dir'],
        freq_idx=viz_config.get('comparison_freq_idx', 1),
        time_window=viz_config.get('time_window', 1000),
        dpi=viz_config.get('dpi', 300)
    )

    logging.info("✓ Visualization complete\n")


def main():
    """
    Main orchestration function.

    Parses CLI arguments, loads configuration, and executes the selected
    pipeline phase(s).
    """
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

    # Apply environment variable overrides
    config = apply_environment_overrides(config)

    # Check for verbose override from environment
    if os.getenv('LSTM_VERBOSE', '').lower() in ('true', '1', 'yes'):
        args.verbose = True

    # Setup logging with optional environment override
    log_level = os.getenv('LSTM_LOG_LEVEL', 'INFO')
    setup_logging(config['paths']['log_file'], args.verbose)

    # Print header
    logging.info("=" * 70)
    logging.info("LSTM Frequency Extraction System")
    logging.info("=" * 70)
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Configuration: {args.config}")
    logging.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 70)
    logging.info("")

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
        logging.info("=" * 70)
        logging.info("PIPELINE COMPLETE")
        logging.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("=" * 70)

    except KeyboardInterrupt:
        logging.warning("\n\nInterrupted by user")
        return 1

    except Exception as e:
        logging.error(f"\n\nError: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
