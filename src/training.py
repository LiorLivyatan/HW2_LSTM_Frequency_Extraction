"""
Training Module for LSTM Frequency Extraction

This module implements the StatefulTrainer class which handles the CRITICAL
L=1 training pattern with proper LSTM state management.

CRITICAL CONCEPT - State Preservation Pattern:
    The trainer implements the key pedagogical concept of this assignment:
    processing samples individually (L=1) while preserving LSTM state across
    the sequence. This requires careful state detachment to prevent memory
    explosion while maintaining temporal learning capability.

Reference: prd/03_TRAINING_PIPELINE_PRD.md
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import json
import time
from typing import Dict, List, Optional


class StatefulTrainer:
    """
    Trainer for FrequencyLSTM with proper state management for L=1.

    This trainer implements the CRITICAL state preservation pattern:
    1. State initialized once per epoch (or None for auto-init)
    2. State preserved across ALL 40,000 samples within an epoch
    3. State detached after EACH backward pass to prevent memory explosion
    4. State reset at the start of each new epoch

    Without correct state management, the model CANNOT learn temporal patterns
    in the L=1 setting, and memory will explode after ~1000 samples.

    Args:
        model: FrequencyLSTM model instance
        train_loader: DataLoader with batch_size=1, shuffle=False
        criterion: Loss function (typically nn.MSELoss())
        optimizer: Optimizer (typically Adam)
        device: Device to train on ('cpu' or 'cuda')
        clip_grad_norm: Maximum gradient norm for clipping (None = no clipping)

    Example:
        >>> model = FrequencyLSTM()
        >>> loader = DataLoader(dataset, batch_size=1, shuffle=False)
        >>> trainer = StatefulTrainer(
        ...     model=model,
        ...     train_loader=loader,
        ...     criterion=nn.MSELoss(),
        ...     optimizer=optim.Adam(model.parameters(), lr=0.001)
        ... )
        >>> history = trainer.train(num_epochs=10)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device = None,
        clip_grad_norm: Optional[float] = 1.0
    ):
        """
        Initialize the StatefulTrainer.

        Args:
            model: The LSTM model to train
            train_loader: DataLoader for training data
            criterion: Loss function
            optimizer: Optimization algorithm
            device: Computation device (auto-detected if None)
            clip_grad_norm: Max gradient norm (helps with stability)
        """
        # Auto-detect device if not specified
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(device)
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.clip_grad_norm = clip_grad_norm

        # Training history
        self.history = {
            'train_loss': [],
            'epoch_times': [],
            'best_loss': float('inf'),
            'best_epoch': 0
        }

        print(f"StatefulTrainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Training samples: {len(train_loader):,}")
        print(f"  Gradient clipping: {clip_grad_norm}")

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch with proper state preservation.

        THIS IS THE CRITICAL FUNCTION implementing the L=1 state management pattern.

        State Management Flow:
            1. Initialize hidden_state = None at epoch start
            2. For each of 40,000 samples:
                a. Forward pass with previous state
                b. Compute loss and backward pass
                c. Update weights
                d. **CRITICAL**: Detach state from computation graph
            3. State is discarded at epoch end (will reinitialize next epoch)

        Args:
            epoch: Current epoch number (for logging)

        Returns:
            float: Average loss for the epoch

        Memory Management:
            The state detachment (step 2d) is CRITICAL. Without it:
            - Computation graph spans all 40,000 samples
            - Memory grows linearly: O(n) where n = 40,000
            - Training crashes with OOM after ~1000-5000 samples

            With detachment:
            - Computation graph only spans current sample
            - Memory is constant: O(1)
            - State VALUES preserved, gradient CONNECTIONS severed
        """
        self.model.train()

        # ================================================================
        # CRITICAL: Initialize state ONCE at epoch start
        # ================================================================
        # Setting to None lets PyTorch LSTM initialize to zeros automatically
        # Alternative: hidden_state = self.model.init_hidden(batch_size=1, device=self.device)
        hidden_state = None

        total_loss = 0.0
        num_samples = 0

        # Progress bar for monitoring
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}",
            leave=True,
            ncols=100
        )

        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move data to device
            inputs = inputs.to(self.device)    # Shape: (batch=1, features=5)
            targets = targets.to(self.device)  # Shape: (batch=1, 1)

            # Reshape for LSTM: (batch, seq_len, features)
            # For L=1: (1, 1, 5)
            inputs = inputs.unsqueeze(1)  # Add sequence dimension

            # ============================================================
            # CRITICAL SECTION: State-preserving forward pass
            # ============================================================

            # Forward pass with previous state
            # - First iteration: hidden_state is None, LSTM initializes to zeros
            # - Subsequent iterations: hidden_state contains previous (h, c)
            output, hidden_state = self.model(inputs, hidden_state)

            # Compute loss
            loss = self.criterion(output, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Optional gradient clipping (helps prevent explosion)
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.clip_grad_norm
                )

            # Update weights
            self.optimizer.step()

            # ============================================================
            # CRITICAL: Detach state from computation graph
            # ============================================================
            # This is THE KEY line that makes L=1 work without memory explosion!
            #
            # What this does:
            # - Creates new tensors with same VALUES as hidden_state
            # - But DETACHED from the computation graph
            # - State values flow forward (temporal learning preserved)
            # - Gradient connections severed (prevents backprop through history)
            #
            # Without this line:
            # - Each new sample adds to computation graph
            # - After N samples, graph has N nodes
            # - Backward pass tries to backprop through all N samples
            # - Memory: O(N), Time: O(N²)
            # - CRASH after ~1000-5000 samples
            #
            # With this line:
            # - Computation graph only contains current sample
            # - Memory: O(1), Time: O(1)
            # - Can train on millions of samples
            hidden_state = tuple(h.detach() for h in hidden_state)

            # ============================================================
            # END CRITICAL SECTION
            # ============================================================

            # Track metrics
            total_loss += loss.item()
            num_samples += 1

            # Update progress bar every 1000 samples
            if (batch_idx + 1) % 1000 == 0:
                avg_loss = total_loss / num_samples
                pbar.set_postfix({'loss': f'{avg_loss:.6f}'})

        # Calculate epoch average loss
        epoch_loss = total_loss / num_samples

        return epoch_loss

    def train(
        self,
        num_epochs: int,
        save_dir: str = 'models',
        save_best: bool = True,
        save_every: Optional[int] = None
    ) -> Dict:
        """
        Train for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save model checkpoints
            save_best: Whether to save the best model
            save_every: Save checkpoint every N epochs (None = only best)

        Returns:
            dict: Training history with losses and times
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*70}\n")

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            # Train one epoch with state preservation
            epoch_loss = self.train_epoch(epoch)

            epoch_time = time.time() - epoch_start_time

            # Update history
            self.history['train_loss'].append(epoch_loss)
            self.history['epoch_times'].append(epoch_time)

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Loss: {epoch_loss:.6f}")
            print(f"  Time: {epoch_time:.1f}s")

            # Save best model
            if save_best and epoch_loss < self.history['best_loss']:
                self.history['best_loss'] = epoch_loss
                self.history['best_epoch'] = epoch + 1

                checkpoint_path = Path(save_dir) / 'best_model.pth'
                self.save_checkpoint(checkpoint_path, epoch, epoch_loss)
                print(f"  ✓ New best model saved (loss: {epoch_loss:.6f})")

            # Save periodic checkpoints
            if save_every is not None and (epoch + 1) % save_every == 0:
                checkpoint_path = Path(save_dir) / f'checkpoint_epoch_{epoch+1}.pth'
                self.save_checkpoint(checkpoint_path, epoch, epoch_loss)
                print(f"  ✓ Checkpoint saved at epoch {epoch+1}")

            print()

        print(f"{'='*70}")
        print(f"Training complete!")
        print(f"  Best loss: {self.history['best_loss']:.6f} (epoch {self.history['best_epoch']})")
        print(f"  Total time: {sum(self.history['epoch_times']):.1f}s")
        print(f"{'='*70}\n")

        # Save training history
        history_path = Path(save_dir) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")

        return self.history

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        loss: float
    ) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            loss: Current loss value
        """
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'history': self.history
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path) -> Dict:
        """
        Load model checkpoint for resuming training.

        Args:
            path: Path to checkpoint file

        Returns:
            dict: Checkpoint data
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)

        print(f"Checkpoint loaded from {path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Loss: {checkpoint['loss']:.6f}")

        return checkpoint


def main():
    """
    Test the StatefulTrainer setup (without full training).

    This demonstrates the trainer initialization and verifies the
    state management pattern is ready for validation by lstm-state-debugger.
    """
    print("=" * 70)
    print("StatefulTrainer - Setup Test")
    print("=" * 70)
    print()

    # Import dependencies
    from src.model import FrequencyLSTM
    from src.dataset import FrequencyDataset

    # Create dataset and loader
    print("Creating dataset and loader...")
    dataset = FrequencyDataset('data/train_data.npy')

    # CRITICAL: L=1 configuration
    loader = DataLoader(
        dataset,
        batch_size=1,      # CRITICAL: L=1 constraint
        shuffle=False,     # CRITICAL: preserve temporal order
        num_workers=0      # Avoid multiprocessing issues
    )
    print()

    # Create model
    print("Creating model...")
    model = FrequencyLSTM(
        input_size=5,
        hidden_size=64,
        num_layers=1
    )
    print(model.get_model_summary())
    print()

    # Create trainer
    print("Creating trainer...")
    trainer = StatefulTrainer(
        model=model,
        train_loader=loader,
        criterion=nn.MSELoss(),
        optimizer=optim.Adam(model.parameters(), lr=0.001),
        clip_grad_norm=1.0
    )
    print()

    print("=" * 70)
    print("Setup complete!")
    print("=" * 70)
    print()
    print("CRITICAL NEXT STEP:")
    print("  ⚠️  BEFORE running any training, invoke lstm-state-debugger agent")
    print("  ⚠️  to validate the state management implementation")
    print()
    print("The lstm-state-debugger will verify:")
    print("  1. State detachment occurs after backward() - line 178")
    print("  2. State is preserved between samples - line 162")
    print("  3. No memory leaks from computation graph accumulation")
    print("  4. Gradient flow is correct")
    print()
    print("After validation passes, training can begin safely.")


if __name__ == "__main__":
    main()
