"""
Model Architecture Module for LSTM Frequency Extraction

This module implements the FrequencyLSTM class - a PyTorch LSTM model that performs
conditional regression to extract individual frequency components from mixed noisy signals.

IMPORTANT: L=1 refers to sequence_length=1 (processing one time point per forward pass),
NOT num_layers=1. The num_layers parameter (number of stacked LSTM layers) is fully
experimentally tunable.

Critical Design Choices:
    - PyTorch LSTM: Enables explicit state management required for L=1 training
    - batch_first=True: Convenient (batch, seq, features) input format
    - Linear output: No activation (regression task, not classification)
    - State return: Always returns (h_n, c_n) for state preservation

Reference: prd/02_MODEL_ARCHITECTURE_PRD.md
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class FrequencyLSTM(nn.Module):
    """
    LSTM model for extracting individual frequency components from
    a mixed noisy signal via conditional regression.

    Architecture:
        Input (5) → LSTM (hidden_size) → FC (1) → Output

    The model accepts:
        - S(t): Noisy mixed signal (1 value)
        - C: One-hot frequency selector (4 values)
    And predicts:
        - Target_i(t): Clean sinusoid for selected frequency

    This architecture supports the L=1 constraint by accepting and returning
    LSTM states explicitly, allowing state preservation across individual samples.

    Args:
        input_size (int): Size of input features.
            [S(t), C1, C2, C3, C4] - typically 5 for 4 frequencies
        hidden_size (int): Number of LSTM hidden units.
            Tunable options: [32, 64, 128, 256]
        num_layers (int): Number of LSTM layers.
            Tunable options: [1, 2, 3, 4, ...]
        dropout (float): Dropout probability between LSTM layers.
            Only applied if num_layers > 1

    Example:
        >>> model = FrequencyLSTM(input_size=5, hidden_size=128, num_layers=1)
        >>> x = torch.randn(1, 1, 5)  # (batch=1, seq=1, features=5)
        >>> output, hidden = model(x)
        >>> print(output.shape)  # torch.Size([1, 1])
        >>> print(hidden[0].shape)  # torch.Size([1, 1, 128])
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        """
        Initialize the FrequencyLSTM model.

        Args:
            input_size: Number of input features (5 for this task)
            hidden_size: LSTM hidden dimension size
            num_layers: Number of stacked LSTM layers
            dropout: Dropout rate (only used if num_layers > 1)
        """
        super(FrequencyLSTM, self).__init__()

        # Store hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # LSTM layer with batch_first for convenient (batch, seq, features) input
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input: (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0.0  # Dropout only for multi-layer
        )

        # Fully connected output layer: hidden_size → 1 (scalar regression)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
                For L=1 training: (1, 1, 5)
                - x[:, :, 0]: S(t) - noisy mixed signal
                - x[:, :, 1:5]: One-hot frequency selector [C1, C2, C3, C4]

            hidden: Optional tuple of (h_0, c_0) LSTM states
                - h_0: Hidden state of shape (num_layers, batch_size, hidden_size)
                - c_0: Cell state of shape (num_layers, batch_size, hidden_size)
                If None, initialized to zeros automatically by PyTorch

        Returns:
            tuple containing:
                - output: Predicted target of shape (batch_size, 1)
                    Clean sinusoid value for the selected frequency
                - (h_n, c_n): Updated LSTM states as tuple
                    - h_n: New hidden state (num_layers, batch_size, hidden_size)
                    - c_n: New cell state (num_layers, batch_size, hidden_size)

        Shape Flow:
            Input x:       (batch_size, seq_len, input_size)
            LSTM output:   (batch_size, seq_len, hidden_size)
            Last timestep: (batch_size, hidden_size)
            FC output:     (batch_size, 1)
        """
        # LSTM forward pass
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        # h_n shape: (num_layers, batch_size, hidden_size)
        # c_n shape: (num_layers, batch_size, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)

        # Extract the last timestep's output
        # For L=1 (seq_len=1), this is just lstm_out[:, 0, :]
        # For general case, use lstm_out[:, -1, :]
        last_output = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)

        # Pass through fully connected layer (no activation - linear regression)
        output = self.fc(last_output)  # Shape: (batch_size, 1)

        # Return prediction and updated states
        return output, (h_n, c_n)

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden and cell states to zeros.

        This is used at the start of each epoch to reset the LSTM state.
        For L=1 training, the state is preserved between samples WITHIN an epoch,
        but reset at epoch boundaries.

        Args:
            batch_size: Batch size (e.g., 1 for sequential, 32 for parallel batches)
            device: Device to create tensors on. If None, uses CPU

        Returns:
            tuple: (h_0, c_0) initialized to zeros
                - h_0: Hidden state of shape (num_layers, batch_size, hidden_size)
                - c_0: Cell state of shape (num_layers, batch_size, hidden_size)

        Example:
            >>> model = FrequencyLSTM()
            >>> h_0, c_0 = model.init_hidden(batch_size=32)
            >>> print(h_0.shape)  # torch.Size([1, 32, 128])
        """
        if device is None:
            device = torch.device('cpu')

        h_0 = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=device
        )
        c_0 = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=device
        )
        return (h_0, c_0)

    def get_or_reset_hidden(
        self,
        current_batch_size: int,
        expected_batch_size: int,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get existing hidden state or reset if batch size doesn't match.

        This handles variable batch sizes (e.g., when the last batch is smaller).
        If the current batch size doesn't match the expected size, we reinitialize
        the hidden state with the correct dimensions.

        Args:
            current_batch_size: Size of the current batch
            expected_batch_size: Expected batch size
            hidden: Existing hidden state tuple (h, c), or None
            device: Device to create tensors on

        Returns:
            tuple: (h, c) with correct batch dimensions

        Example:
            >>> model = FrequencyLSTM()
            >>> # Normal batch (32 samples)
            >>> h = model.get_or_reset_hidden(32, 32, existing_hidden, device)
            >>> # Last batch (16 samples) - will reinitialize
            >>> h = model.get_or_reset_hidden(16, 32, existing_hidden, device)
        """
        # If no hidden state exists, initialize new one
        if hidden is None:
            return self.init_hidden(current_batch_size, device)

        # If batch size matches, return existing state
        if current_batch_size == expected_batch_size:
            return hidden

        # Batch size changed (e.g., last batch is smaller), reinitialize
        return self.init_hidden(current_batch_size, device)

    def count_parameters(self) -> int:
        """
        Count total number of trainable parameters in the model.

        Returns:
            int: Total number of trainable parameters

        Example:
            >>> model = FrequencyLSTM(hidden_size=128, num_layers=1)
            >>> print(f"Parameters: {model.count_parameters():,}")
            Parameters: 67,713
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_summary(self) -> str:
        """
        Get a formatted string summary of the model architecture.

        Returns:
            str: Multi-line string with model configuration and parameter count

        Example:
            >>> model = FrequencyLSTM()
            >>> print(model.get_model_summary())
        """
        summary = []
        summary.append("=" * 60)
        summary.append("FrequencyLSTM Model Summary")
        summary.append("=" * 60)
        summary.append(f"Input size:       {self.input_size}")
        summary.append(f"Hidden size:      {self.hidden_size}")
        summary.append(f"Num layers:       {self.num_layers}")
        summary.append(f"Dropout:          {self.dropout}")
        summary.append(f"Output size:      1")
        summary.append("-" * 60)
        summary.append(f"LSTM parameters:  {sum(p.numel() for p in self.lstm.parameters()):,}")
        summary.append(f"FC parameters:    {sum(p.numel() for p in self.fc.parameters()):,}")
        summary.append(f"Total parameters: {self.count_parameters():,}")
        summary.append("=" * 60)

        # Add architecture details
        summary.append("\nArchitecture:")
        summary.append(f"  Input ({self.input_size}) → LSTM ({self.hidden_size}) → FC (1) → Output")
        summary.append("\nState shapes (for batch_size=1):")
        summary.append(f"  Hidden state: ({self.num_layers}, 1, {self.hidden_size})")
        summary.append(f"  Cell state:   ({self.num_layers}, 1, {self.hidden_size})")
        summary.append("=" * 60)

        return "\n".join(summary)


def main():
    """
    Test the FrequencyLSTM model with dummy data.

    This demonstrates:
        1. Model initialization
        2. Forward pass with no initial state
        3. Forward pass with previous state
        4. State propagation across multiple steps
    """
    print("="* 70)
    print("FrequencyLSTM Model - Test Run")
    print("=" * 70)
    print()

    print("Creating model...")
    model = FrequencyLSTM(
        input_size=5,
        hidden_size=128,
        num_layers=1,
        dropout=0.0
    )

    print(model.get_model_summary())
    print()

    # Test 1: Forward pass without state
    print("Test 1: Forward pass without initial state")
    print("-" * 70)
    x = torch.randn(1, 1, 5)  # (batch=1, seq=1, features=5)
    print(f"Input shape: {x.shape}")

    output, (h_n, c_n) = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Output value: {output.item():.4f}")
    print(f"Hidden state shape: {h_n.shape}")
    print(f"Cell state shape: {c_n.shape}")
    print()

    # Test 2: Forward pass with previous state
    print("Test 2: Forward pass with previous state")
    print("-" * 70)
    x_next = torch.randn(1, 1, 5)
    print(f"Input shape: {x_next.shape}")

    output_next, (h_n_next, c_n_next) = model(x_next, (h_n, c_n))
    print(f"Output shape: {output_next.shape}")
    print(f"Output value: {output_next.item():.4f}")
    print(f"Hidden state changed: {not torch.equal(h_n, h_n_next)}")
    print(f"Cell state changed: {not torch.equal(c_n, c_n_next)}")
    print()

    # Test 3: Initialize states manually
    print("Test 3: Manual state initialization")
    print("-" * 70)
    h_0, c_0 = model.init_hidden(batch_size=1)
    print(f"Initialized hidden state shape: {h_0.shape}")
    print(f"Initialized cell state shape: {c_0.shape}")
    print(f"Hidden state all zeros: {torch.all(h_0 == 0).item()}")
    print(f"Cell state all zeros: {torch.all(c_0 == 0).item()}")
    print()

    # Test 4: Gradient flow
    print("Test 4: Gradient flow test")
    print("-" * 70)
    model.train()
    x = torch.randn(1, 1, 5, requires_grad=True)
    target = torch.randn(1, 1)

    output, _ = model(x)
    loss = nn.MSELoss()(output, target)
    loss.backward()

    print(f"Loss: {loss.item():.6f}")

    has_grads = all(
        param.grad is not None and not torch.all(param.grad == 0)
        for param in model.parameters()
    )
    print(f"All parameters have gradients: {has_grads}")
    print()

    # Test 5: State preservation simulation (L=1 pattern)
    print("Test 5: Simulating L=1 state preservation")
    print("-" * 70)
    model.eval()

    hidden_state = model.init_hidden(batch_size=1)

    print("Processing 5 sequential samples...")
    for i in range(5):
        x_sample = torch.randn(1, 1, 5)
        output, hidden_state = model(x_sample, hidden_state)
        print(f"  Sample {i+1}: output = {output.item():.4f}, "
              f"h_norm = {torch.norm(hidden_state[0]).item():.4f}")

        # CRITICAL: In actual training, state would be detached here
        # hidden_state = tuple(h.detach() for h in hidden_state)

    print()

    print("=" * 70)
    print("Model tests complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Review model architecture and parameter count")
    print("  2. Proceed to Phase 3: Training Pipeline")
    print("  3. Remember: State must be detached after backward() in training!")


if __name__ == "__main__":
    main()
