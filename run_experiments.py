"""
Experiment Runner for LSTM Frequency Extraction

This script runs systematic experiments with different hyperparameter configurations
to perform sensitivity analysis and find optimal model settings.

Usage:
    # Run all experiments
    python run_experiments.py

    # Run quick experiments (fewer epochs for testing)
    python run_experiments.py --quick

    # Run specific experiment by ID
    python run_experiments.py --experiment 3

Output:
    - outputs/experiments/experiment_results.json - All experiment results
    - outputs/experiments/experiment_table.md - Markdown table of results
    - outputs/experiments/sensitivity_*.png - Sensitivity analysis plots
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml

from src.data_generation import SignalGenerator
from src.dataset import FrequencyDataset
from src.model import FrequencyLSTM
from src.training import StatefulTrainer
from src.evaluation import Evaluator


# Define experiment configurations
EXPERIMENTS = [
    # Baseline (your current configuration)
    {
        'id': 1,
        'name': 'Baseline',
        'hidden_size': 128,
        'num_layers': 1,
        'learning_rate': 0.0001,
        'num_epochs': 100,
        'batch_size': 32,
        'description': 'Current baseline configuration'
    },

    # Hidden size experiments
    {
        'id': 2,
        'name': 'Small Hidden',
        'hidden_size': 64,
        'num_layers': 1,
        'learning_rate': 0.0001,
        'num_epochs': 100,
        'batch_size': 32,
        'description': 'Reduced hidden size for comparison'
    },
    {
        'id': 3,
        'name': 'Large Hidden',
        'hidden_size': 256,
        'num_layers': 1,
        'learning_rate': 0.0001,
        'num_epochs': 100,
        'batch_size': 32,
        'description': 'Increased hidden size for more capacity'
    },

    # Layer experiments
    {
        'id': 4,
        'name': '2-Layer LSTM',
        'hidden_size': 128,
        'num_layers': 2,
        'learning_rate': 0.0001,
        'num_epochs': 100,
        'batch_size': 32,
        'description': 'Two stacked LSTM layers'
    },
    {
        'id': 5,
        'name': '3-Layer LSTM',
        'hidden_size': 128,
        'num_layers': 3,
        'learning_rate': 0.0001,
        'num_epochs': 100,
        'batch_size': 32,
        'description': 'Three stacked LSTM layers'
    },

    # Learning rate experiments
    {
        'id': 6,
        'name': 'Higher LR',
        'hidden_size': 128,
        'num_layers': 1,
        'learning_rate': 0.001,
        'num_epochs': 100,
        'batch_size': 32,
        'description': 'Higher learning rate (10x baseline)'
    },
    {
        'id': 7,
        'name': 'Lower LR',
        'hidden_size': 128,
        'num_layers': 1,
        'learning_rate': 0.00001,
        'num_epochs': 100,
        'batch_size': 32,
        'description': 'Lower learning rate (0.1x baseline)'
    },

    # Combined improvements
    {
        'id': 8,
        'name': 'Large + 2-Layer',
        'hidden_size': 256,
        'num_layers': 2,
        'learning_rate': 0.0001,
        'num_epochs': 100,
        'batch_size': 32,
        'description': 'Large hidden size with 2 layers'
    },
    {
        'id': 9,
        'name': 'Optimal Candidate',
        'hidden_size': 256,
        'num_layers': 2,
        'learning_rate': 0.001,
        'num_epochs': 150,
        'batch_size': 32,
        'description': 'Best combination candidate'
    },

    # Batch size experiment
    {
        'id': 10,
        'name': 'Large Batch',
        'hidden_size': 128,
        'num_layers': 1,
        'learning_rate': 0.0001,
        'num_epochs': 100,
        'batch_size': 64,
        'description': 'Larger batch size'
    },
]


def setup_logging(verbose: bool = False):
    """Configure logging for experiments."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('logs/experiments.log'),
            logging.StreamHandler()
        ]
    )


def load_base_config(config_path: str = 'config.yaml') -> dict:
    """Load base configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_single_experiment(exp_config: dict, base_config: dict, output_dir: str) -> dict:
    """
    Run a single experiment with the given configuration.

    Args:
        exp_config: Experiment-specific configuration
        base_config: Base configuration from config.yaml
        output_dir: Directory to save experiment outputs

    Returns:
        dict: Experiment results including metrics
    """
    exp_id = exp_config['id']
    exp_name = exp_config['name']

    logging.info(f"\n{'='*70}")
    logging.info(f"EXPERIMENT {exp_id}: {exp_name}")
    logging.info(f"{'='*70}")
    logging.info(f"Config: hidden={exp_config['hidden_size']}, layers={exp_config['num_layers']}, "
                 f"lr={exp_config['learning_rate']}, epochs={exp_config['num_epochs']}")

    # Create experiment-specific directories
    exp_dir = Path(output_dir) / f"exp_{exp_id:02d}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set seeds for reproducibility
    np.random.seed(base_config['data']['train_seed'])
    torch.manual_seed(base_config['data']['train_seed'])

    start_time = time.time()

    # Create model with experiment configuration
    model = FrequencyLSTM(
        input_size=base_config['model']['input_size'],
        hidden_size=exp_config['hidden_size'],
        num_layers=exp_config['num_layers'],
        dropout=0.0
    )

    # Load training data
    train_dataset = FrequencyDataset(base_config['paths']['train_data'])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=exp_config['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # Setup training
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=exp_config['learning_rate']
    )

    # Create trainer
    trainer = StatefulTrainer(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        clip_grad_norm=base_config['training']['clip_grad_norm']
    )

    # Train
    history = trainer.train(
        num_epochs=exp_config['num_epochs'],
        save_dir=str(exp_dir),
        save_best=True
    )

    training_time = time.time() - start_time

    # Evaluate
    logging.info("Evaluating model...")

    # Load test data
    test_dataset = FrequencyDataset(base_config['paths']['test_data'])
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=exp_config['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # Reload best model
    checkpoint = torch.load(exp_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create evaluator
    evaluator = Evaluator(model, device=device)

    # Get metrics
    results = evaluator.evaluate_all(train_loader, test_loader)

    # Extract per-frequency test MSE
    per_freq_test = {}
    frequencies = base_config['data']['frequencies']
    for i, freq in enumerate(frequencies):
        per_freq_test[f'{freq}Hz'] = results['per_frequency']['test'][i]

    # Compile results
    experiment_result = {
        'id': exp_id,
        'name': exp_name,
        'description': exp_config['description'],
        'config': {
            'hidden_size': exp_config['hidden_size'],
            'num_layers': exp_config['num_layers'],
            'learning_rate': exp_config['learning_rate'],
            'num_epochs': exp_config['num_epochs'],
            'batch_size': exp_config['batch_size']
        },
        'metrics': {
            'train_mse': results['overall']['mse_train'],
            'test_mse': results['overall']['mse_test'],
            'generalization_diff': results['generalization']['relative_difference'],
            'best_epoch': history['best_epoch'],
            'final_loss': history['train_loss'][-1],
            'per_frequency_test': per_freq_test
        },
        'training_time': training_time,
        'total_parameters': sum(p.numel() for p in model.parameters())
    }

    logging.info(f"Results: Train MSE={results['overall']['mse_train']:.6f}, "
                 f"Test MSE={results['overall']['mse_test']:.6f}, "
                 f"Time={training_time:.1f}s")

    return experiment_result


def generate_results_table(results: list, output_path: str):
    """
    Generate a markdown table of experiment results.

    Args:
        results: List of experiment results
        output_path: Path to save markdown table
    """
    lines = []
    lines.append("# Experiment Results\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("")

    # Summary table
    lines.append("## Summary Table\n")
    lines.append("| ID | Name | Hidden | Layers | LR | Epochs | Train MSE | Test MSE | Gen. Diff | 7Hz MSE | Time (s) |")
    lines.append("|----|----|--------|--------|------|--------|-----------|----------|-----------|---------|----------|")

    for r in sorted(results, key=lambda x: x['id']):
        hz7_mse = r['metrics']['per_frequency_test'].get('7Hz', 'N/A')
        if isinstance(hz7_mse, float):
            hz7_mse = f"{hz7_mse:.4f}"

        lines.append(
            f"| {r['id']} | {r['name']} | {r['config']['hidden_size']} | "
            f"{r['config']['num_layers']} | {r['config']['learning_rate']} | "
            f"{r['config']['num_epochs']} | {r['metrics']['train_mse']:.4f} | "
            f"{r['metrics']['test_mse']:.4f} | {r['metrics']['generalization_diff']:.2%} | "
            f"{hz7_mse} | {r['training_time']:.1f} |"
        )

    lines.append("")

    # Best results
    lines.append("## Best Results\n")

    # Best overall MSE
    best_test = min(results, key=lambda x: x['metrics']['test_mse'])
    lines.append(f"**Best Test MSE**: Experiment {best_test['id']} ({best_test['name']}) - "
                 f"MSE = {best_test['metrics']['test_mse']:.6f}")
    lines.append("")

    # Best 7Hz performance
    best_7hz = min(results, key=lambda x: x['metrics']['per_frequency_test'].get('7Hz', float('inf')))
    lines.append(f"**Best 7Hz MSE**: Experiment {best_7hz['id']} ({best_7hz['name']}) - "
                 f"MSE = {best_7hz['metrics']['per_frequency_test']['7Hz']:.6f}")
    lines.append("")

    # Best generalization
    best_gen = min(results, key=lambda x: x['metrics']['generalization_diff'])
    lines.append(f"**Best Generalization**: Experiment {best_gen['id']} ({best_gen['name']}) - "
                 f"Diff = {best_gen['metrics']['generalization_diff']:.4%}")
    lines.append("")

    # Per-frequency breakdown
    lines.append("## Per-Frequency Test MSE\n")
    lines.append("| ID | Name | 1Hz | 3Hz | 5Hz | 7Hz |")
    lines.append("|----|------|-----|-----|-----|-----|")

    for r in sorted(results, key=lambda x: x['id']):
        pf = r['metrics']['per_frequency_test']
        lines.append(
            f"| {r['id']} | {r['name']} | {pf.get('1Hz', 0):.4f} | "
            f"{pf.get('3Hz', 0):.4f} | {pf.get('5Hz', 0):.4f} | "
            f"{pf.get('7Hz', 0):.4f} |"
        )

    lines.append("")

    # Analysis
    lines.append("## Analysis\n")

    # Hidden size analysis
    hidden_results = {}
    for r in results:
        h = r['config']['hidden_size']
        if h not in hidden_results:
            hidden_results[h] = []
        hidden_results[h].append(r['metrics']['test_mse'])

    lines.append("### Hidden Size Impact\n")
    for h in sorted(hidden_results.keys()):
        avg_mse = np.mean(hidden_results[h])
        lines.append(f"- **Hidden={h}**: Avg Test MSE = {avg_mse:.4f}")
    lines.append("")

    # Layer analysis
    layer_results = {}
    for r in results:
        l = r['config']['num_layers']
        if l not in layer_results:
            layer_results[l] = []
        layer_results[l].append(r['metrics']['test_mse'])

    lines.append("### Number of Layers Impact\n")
    for l in sorted(layer_results.keys()):
        avg_mse = np.mean(layer_results[l])
        lines.append(f"- **Layers={l}**: Avg Test MSE = {avg_mse:.4f}")
    lines.append("")

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    logging.info(f"Results table saved to {output_path}")


def generate_sensitivity_plots(results: list, output_dir: str):
    """
    Generate sensitivity analysis plots.

    Args:
        results: List of experiment results
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Plot 1: MSE vs Hidden Size
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by hidden size (single layer only for fair comparison)
    single_layer = [r for r in results if r['config']['num_layers'] == 1]
    hidden_sizes = sorted(set(r['config']['hidden_size'] for r in single_layer))

    train_mses = []
    test_mses = []

    for h in hidden_sizes:
        matching = [r for r in single_layer if r['config']['hidden_size'] == h]
        if matching:
            # Use the one with standard learning rate if multiple
            r = matching[0]
            train_mses.append(r['metrics']['train_mse'])
            test_mses.append(r['metrics']['test_mse'])

    x = np.arange(len(hidden_sizes))
    width = 0.35

    ax.bar(x - width/2, train_mses, width, label='Train MSE', color='steelblue')
    ax.bar(x + width/2, test_mses, width, label='Test MSE', color='coral')

    ax.set_xlabel('Hidden Size', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Sensitivity Analysis: Hidden Size vs MSE', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(hidden_sizes)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'sensitivity_hidden_size.png', dpi=300)
    plt.close()

    # Plot 2: MSE vs Number of Layers
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by num_layers (hidden=128, lr=0.0001 for fair comparison)
    standard_config = [r for r in results
                      if r['config']['hidden_size'] == 128
                      and r['config']['learning_rate'] == 0.0001]

    num_layers = sorted(set(r['config']['num_layers'] for r in standard_config))

    train_mses = []
    test_mses = []

    for n in num_layers:
        matching = [r for r in standard_config if r['config']['num_layers'] == n]
        if matching:
            r = matching[0]
            train_mses.append(r['metrics']['train_mse'])
            test_mses.append(r['metrics']['test_mse'])

    x = np.arange(len(num_layers))

    ax.bar(x - width/2, train_mses, width, label='Train MSE', color='steelblue')
    ax.bar(x + width/2, test_mses, width, label='Test MSE', color='coral')

    ax.set_xlabel('Number of LSTM Layers', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Sensitivity Analysis: Number of Layers vs MSE', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(num_layers)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'sensitivity_num_layers.png', dpi=300)
    plt.close()

    # Plot 3: MSE vs Learning Rate
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by learning rate (hidden=128, layers=1 for fair comparison)
    standard_config = [r for r in results
                      if r['config']['hidden_size'] == 128
                      and r['config']['num_layers'] == 1]

    learning_rates = sorted(set(r['config']['learning_rate'] for r in standard_config))

    train_mses = []
    test_mses = []

    for lr in learning_rates:
        matching = [r for r in standard_config if r['config']['learning_rate'] == lr]
        if matching:
            r = matching[0]
            train_mses.append(r['metrics']['train_mse'])
            test_mses.append(r['metrics']['test_mse'])

    x = np.arange(len(learning_rates))

    ax.bar(x - width/2, train_mses, width, label='Train MSE', color='steelblue')
    ax.bar(x + width/2, test_mses, width, label='Test MSE', color='coral')

    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Sensitivity Analysis: Learning Rate vs MSE', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{lr:.0e}' for lr in learning_rates])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'sensitivity_learning_rate.png', dpi=300)
    plt.close()

    # Plot 4: Per-Frequency MSE comparison (best configs)
    fig, ax = plt.subplots(figsize=(12, 6))

    frequencies = ['1Hz', '3Hz', '5Hz', '7Hz']

    # Select top 5 experiments by test MSE
    top_results = sorted(results, key=lambda x: x['metrics']['test_mse'])[:5]

    x = np.arange(len(frequencies))
    width = 0.15

    for i, r in enumerate(top_results):
        mses = [r['metrics']['per_frequency_test'][f] for f in frequencies]
        offset = (i - 2) * width
        ax.bar(x + offset, mses, width, label=f"Exp {r['id']}: {r['name']}")

    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('Test MSE', fontsize=12)
    ax.set_title('Per-Frequency MSE: Top 5 Configurations', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(frequencies)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'sensitivity_per_frequency.png', dpi=300)
    plt.close()

    # Plot 5: Overall comparison heatmap-style
    fig, ax = plt.subplots(figsize=(14, 8))

    # Sort by test MSE
    sorted_results = sorted(results, key=lambda x: x['metrics']['test_mse'])

    exp_names = [f"Exp {r['id']}" for r in sorted_results]
    test_mses = [r['metrics']['test_mse'] for r in sorted_results]

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(test_mses)))

    bars = ax.barh(exp_names, test_mses, color=colors)

    # Add value labels
    for bar, mse in zip(bars, test_mses):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{mse:.4f}', va='center', fontsize=9)

    ax.set_xlabel('Test MSE', fontsize=12)
    ax.set_title('All Experiments Ranked by Test MSE', fontsize=14)
    ax.axvline(x=0.01, color='green', linestyle='--', linewidth=2, label='Target MSE (0.01)')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'experiment_ranking.png', dpi=300)
    plt.close()

    logging.info(f"Sensitivity plots saved to {output_dir}")


def main():
    """Main function to run experiments."""
    parser = argparse.ArgumentParser(description='Run LSTM Frequency Extraction Experiments')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick experiments with fewer epochs')
    parser.add_argument('--experiment', type=int,
                       help='Run specific experiment by ID')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Setup
    Path('logs').mkdir(exist_ok=True)
    setup_logging(args.verbose)

    logging.info("=" * 70)
    logging.info("LSTM Frequency Extraction - Experiment Runner")
    logging.info("=" * 70)
    logging.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Total experiments defined: {len(EXPERIMENTS)}")

    # Load base config
    base_config = load_base_config()

    # Create output directory
    output_dir = 'outputs/experiments'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Determine which experiments to run
    if args.experiment:
        experiments = [e for e in EXPERIMENTS if e['id'] == args.experiment]
        if not experiments:
            logging.error(f"Experiment {args.experiment} not found!")
            return 1
    else:
        experiments = EXPERIMENTS

    # Modify for quick mode
    if args.quick:
        logging.info("Running in QUICK mode (20 epochs)")
        for exp in experiments:
            exp['num_epochs'] = 20

    # Run experiments
    results = []
    total_start = time.time()

    for i, exp_config in enumerate(experiments):
        logging.info(f"\nProgress: {i+1}/{len(experiments)}")

        try:
            result = run_single_experiment(exp_config, base_config, output_dir)
            results.append(result)

            # Save intermediate results
            with open(f'{output_dir}/experiment_results.json', 'w') as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            logging.error(f"Experiment {exp_config['id']} failed: {e}")
            continue

    total_time = time.time() - total_start

    # Generate outputs
    logging.info("\n" + "=" * 70)
    logging.info("Generating Analysis Outputs")
    logging.info("=" * 70)

    # Generate results table
    generate_results_table(results, f'{output_dir}/experiment_table.md')

    # Generate sensitivity plots
    generate_sensitivity_plots(results, output_dir)

    # Summary
    logging.info("\n" + "=" * 70)
    logging.info("EXPERIMENT RUN COMPLETE")
    logging.info("=" * 70)
    logging.info(f"Total experiments: {len(results)}")
    logging.info(f"Total time: {total_time/60:.1f} minutes")

    if results:
        best = min(results, key=lambda x: x['metrics']['test_mse'])
        logging.info(f"Best result: Experiment {best['id']} ({best['name']})")
        logging.info(f"  Test MSE: {best['metrics']['test_mse']:.6f}")
        logging.info(f"  7Hz MSE: {best['metrics']['per_frequency_test']['7Hz']:.6f}")

    logging.info(f"\nOutputs saved to: {output_dir}/")
    logging.info("  - experiment_results.json")
    logging.info("  - experiment_table.md")
    logging.info("  - sensitivity_*.png")

    return 0


if __name__ == '__main__':
    exit(main())
