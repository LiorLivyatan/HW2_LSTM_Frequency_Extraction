"""
LSTM Frequency Extraction - Main Orchestration Script

This script provides a unified entry point for running the entire pipeline
or individual phases. It handles configuration, logging, and orchestrates
all components with robust error handling.

Usage:
    # Run entire pipeline
    python main.py --mode all

    # Run individual phases
    python main.py --mode data   # Data generation only
    python main.py --mode train  # Training only
    python main.py --mode eval   # Evaluation only
    python main.py --mode viz    # Visualization only
    python main.py --mode ui     # Launch Streamlit UI

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
import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Callable, Dict, Any

# Import all pipeline components
from src.data_generation import SignalGenerator
from src.dataset import FrequencyDataset
from src.model import FrequencyLSTM
from src.training import StatefulTrainer
from src.evaluation import Evaluator
from src.visualization import Visualizer


def setup_logging(log_file: str = 'logs/lstm_extraction.log', verbose: bool = False) -> None:
    """
    Configure logging for the pipeline.

    Sets up both file and stream handlers. Creates the log directory if it
    doesn't exist.

    Args:
        log_file: Path to the log file.
        verbose: If True, set logging level to DEBUG, else INFO.
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


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        dict: The configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        raise


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save the current configuration to a file.

    Args:
        config: The configuration dictionary to save.
        output_path: The path where the config should be saved.
    """
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    except Exception as e:
        logging.warning(f"Failed to save run configuration: {e}")


def set_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility across NumPy and PyTorch.

    Args:
        seed: The integer seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.debug(f"Random seeds set to {seed}")


def run_phase(phase_func: Callable[[Dict[str, Any]], None], config: Dict[str, Any], phase_name: str) -> bool:
    """
    Execute a pipeline phase with standardized logging and error handling.

    Args:
        phase_func: The function implementing the phase. Must accept config dict.
        config: The configuration dictionary.
        phase_name: Human-readable name of the phase for logging.

    Returns:
        bool: True if the phase completed successfully, False otherwise.
    """
    logging.info("=" * 70)
    logging.info(f"PHASE: {phase_name}")
    logging.info("=" * 70)

    try:
        phase_func(config)
        logging.info(f"✓ {phase_name} complete\n")
        return True
    except Exception as e:
        logging.error(f"❌ Error in {phase_name}: {e}", exc_info=True)
        return False


def phase_data_generation(config: Dict[str, Any]) -> None:
    """
    Phase 1: Generate training and test datasets.

    Uses SignalGenerator to create datasets with different random seeds
    to ensure noise independence.

    Args:
        config: Configuration dictionary containing 'data' and 'paths' sections.
    """
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


def phase_training(config: Dict[str, Any]) -> None:
    """
    Phase 2 & 3: Model creation and training.

    Initializes the FrequencyLSTM model and trains it using StatefulTrainer.
    Ensures reproducibility by setting seeds before training.

    Args:
        config: Configuration dictionary.
    """
    # Set seeds for training reproducibility
    set_seeds(config['data']['train_seed'])

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
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        save_dir=config['training']['save_dir'],
        save_best=True
    )


def phase_evaluation(config: Dict[str, Any]) -> None:
    """
    Phase 4: Evaluation.

    Loads the best trained model and evaluates it on both training and test sets.
    Computes MSE metrics and checks for generalization.

    Args:
        config: Configuration dictionary.
    """
    # Device selection
    device_config = config['training']['device']
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)

    # Load model architecture
    model = FrequencyLSTM(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers']
    )

    # Load weights
    checkpoint_path = config['paths']['model_checkpoint']
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}. Run training first.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
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


def phase_visualization(config: Dict[str, Any]) -> None:
    """
    Phase 5: Visualization.

    Generates static plots for model performance and signal comparison.

    Args:
        config: Configuration dictionary.
    """
    # Create visualizer
    visualizer = Visualizer(
        predictions_path=config['paths']['predictions'],
        data_path=config['paths']['test_data']
    )

    # Create graphs
    visualizer.create_all_visualizations(
        output_dir=config['visualization']['output_dir']
    )


def phase_ui(config: Dict[str, Any]) -> None:
    """
    Launch the Streamlit UI Dashboard.

    Checks if Streamlit is installed and then launches the dashboard script
    using a subprocess.

    Args:
        config: Configuration dictionary (unused but kept for signature consistency).
    """
    # Check if streamlit is installed
    if shutil.which("streamlit") is None:
        raise RuntimeError("Streamlit is not installed. Please run 'pip install streamlit plotly pandas'.")

    dashboard_path = Path("src/ui/dashboard.py")
    if not dashboard_path.exists():
        raise FileNotFoundError(f"Dashboard script not found at {dashboard_path}")

    logging.info("Launching UI Dashboard...")
    logging.info("Press Ctrl+C to stop the server.")
    
    # Run streamlit
    try:
        subprocess.run(["streamlit", "run", str(dashboard_path)], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Streamlit exited with error code {e.returncode}")
        raise


def main() -> int:
    """
    Main orchestration function.

    Parses CLI arguments, loads configuration, and executes the selected
    pipeline phase(s) using the robust `run_phase` wrapper.

    Returns:
        int: Exit code (0 for success, 1 for failure).
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

  # Launch UI
  python main.py --mode ui
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['all', 'data', 'train', 'eval', 'viz', 'ui'],
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

    # Setup logging first to capture any config loading errors
    setup_logging('logs/lstm_extraction.log', args.verbose)

    # Print header
    logging.info("=" * 70)
    logging.info("LSTM Frequency Extraction System")
    logging.info("=" * 70)
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Configuration: {args.config}")
    logging.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 70)
    logging.info("")

    try:
        # Load configuration
        config = load_config(args.config)
        
        # Save configuration for this run
        save_config(config, 'outputs/run_config.yaml')

        success = True

        # Execute phases based on mode
        if args.mode in ['all', 'data']:
            if not run_phase(phase_data_generation, config, "Data Generation"):
                success = False

        if success and args.mode in ['all', 'train']:
            if not run_phase(phase_training, config, "Model Training"):
                success = False

        if success and args.mode in ['all', 'eval']:
            if not run_phase(phase_evaluation, config, "Evaluation"):
                success = False

        if success and args.mode in ['all', 'viz']:
            if not run_phase(phase_visualization, config, "Visualization"):
                success = False
        
        if success and args.mode == 'ui':
            if not run_phase(phase_ui, config, "UI Dashboard"):
                success = False

        # Print completion status
        logging.info("=" * 70)
        if success:
            logging.info("PIPELINE COMPLETE SUCCESS")
        else:
            logging.error("PIPELINE FAILED")
        logging.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("=" * 70)

        return 0 if success else 1

    except KeyboardInterrupt:
        logging.warning("\n\nInterrupted by user")
        return 1
    except Exception as e:
        logging.critical(f"\n\nCritical Error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
