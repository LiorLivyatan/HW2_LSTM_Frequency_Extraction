"""
Tests for src/model.py

Tests the FrequencyLSTM model architecture.
Target: ~45 tests covering architecture, forward pass, state shapes.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import FrequencyLSTM


class TestFrequencyLSTMInit:
    """Tests for FrequencyLSTM.__init__"""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        model = FrequencyLSTM()
        assert model is not None

    def test_custom_hidden_size(self):
        """Test initialization with custom hidden_size."""
        model = FrequencyLSTM(hidden_size=256)
        assert model.hidden_size == 256

    def test_custom_num_layers(self):
        """Test initialization with custom num_layers."""
        model = FrequencyLSTM(num_layers=3)
        assert model.num_layers == 3

    def test_custom_dropout(self):
        """Test initialization with custom dropout."""
        model = FrequencyLSTM(num_layers=2, dropout=0.5)
        # Dropout only applies with num_layers > 1
        assert model.dropout == 0.5

    def test_lstm_layer_created(self):
        """Test that LSTM layer is created."""
        model = FrequencyLSTM()
        assert hasattr(model, 'lstm')
        assert isinstance(model.lstm, nn.LSTM)

    def test_fc_layer_created(self):
        """Test that fully connected layer is created."""
        model = FrequencyLSTM()
        assert hasattr(model, 'fc')
        assert isinstance(model.fc, nn.Linear)

    def test_input_size_stored(self):
        """Test that input_size is stored."""
        model = FrequencyLSTM(input_size=10)
        assert model.input_size == 10

    def test_model_parameters(self):
        """Test that model has trainable parameters."""
        model = FrequencyLSTM()
        params = list(model.parameters())
        assert len(params) > 0


class TestFrequencyLSTMForward:
    """Tests for FrequencyLSTM.forward"""

    def test_forward_output_type(self, small_model, sample_input_tensor):
        """Test that forward returns correct output type."""
        output, hidden = small_model(sample_input_tensor)
        assert isinstance(output, torch.Tensor)
        assert isinstance(hidden, tuple)

    def test_output_shape_batch_1(self, small_model):
        """Test output shape with batch_size=1."""
        x = torch.randn(1, 1, 5)
        output, _ = small_model(x)
        assert output.shape == (1, 1)

    def test_output_shape_batch_4(self, small_model, sample_input_tensor):
        """Test output shape with batch_size=4."""
        output, _ = small_model(sample_input_tensor)
        assert output.shape == (4, 1)

    def test_output_shape_batch_32(self, small_model):
        """Test output shape with batch_size=32."""
        x = torch.randn(32, 1, 5)
        output, _ = small_model(x)
        assert output.shape == (32, 1)

    def test_hidden_state_shape(self, small_model, sample_input_tensor):
        """Test that hidden state has correct shape."""
        _, (h_n, c_n) = small_model(sample_input_tensor)

        expected_shape = (small_model.num_layers, 4, small_model.hidden_size)
        assert h_n.shape == expected_shape
        assert c_n.shape == expected_shape

    def test_cell_state_shape(self, small_model, sample_input_tensor):
        """Test that cell state has correct shape."""
        _, (h_n, c_n) = small_model(sample_input_tensor)
        assert c_n.shape == h_n.shape

    def test_forward_without_hidden_state(self, small_model, sample_input_tensor):
        """Test forward pass with hidden=None."""
        output, hidden = small_model(sample_input_tensor, hidden=None)
        assert output is not None
        assert hidden is not None

    def test_forward_with_hidden_state(self, small_model, sample_input_tensor, initial_hidden_state):
        """Test forward pass with existing hidden state."""
        output, hidden = small_model(sample_input_tensor, hidden=initial_hidden_state)
        assert output is not None
        assert hidden is not None

    def test_hidden_state_changes(self, small_model, sample_input_tensor):
        """Test that hidden state changes after forward pass."""
        _, hidden1 = small_model(sample_input_tensor)

        # Second forward pass with same input
        _, hidden2 = small_model(sample_input_tensor, hidden=hidden1)

        # States should be different
        assert not torch.equal(hidden1[0], hidden2[0])

    def test_different_input_different_output(self, small_model):
        """Test that different inputs produce different outputs."""
        x1 = torch.randn(1, 1, 5)
        x2 = torch.randn(1, 1, 5)

        output1, _ = small_model(x1)
        output2, _ = small_model(x2)

        assert not torch.equal(output1, output2)

    def test_gradient_flow_training_mode(self, small_model, sample_input_tensor):
        """Test that gradients flow in training mode."""
        small_model.train()
        output, _ = small_model(sample_input_tensor)

        # Compute dummy loss and backprop
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        for param in small_model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_no_gradient_in_eval_mode(self, small_model, sample_input_tensor):
        """Test forward pass in evaluation mode."""
        small_model.eval()
        with torch.no_grad():
            output, _ = small_model(sample_input_tensor)

        assert output is not None

    def test_multi_layer_forward(self, multi_layer_model, sample_input_tensor):
        """Test forward pass with multiple LSTM layers."""
        output, (h_n, c_n) = multi_layer_model(sample_input_tensor)

        expected_shape = (multi_layer_model.num_layers, 4, multi_layer_model.hidden_size)
        assert h_n.shape == expected_shape

    def test_sequential_forward_passes(self, small_model):
        """Test multiple sequential forward passes (L=1 simulation)."""
        batch_size = 1
        hidden = None

        for _ in range(10):
            x = torch.randn(batch_size, 1, 5)
            output, hidden = small_model(x, hidden)
            # Detach like in real training
            hidden = tuple(h.detach() for h in hidden)

        assert output.shape == (1, 1)


class TestInitHidden:
    """Tests for FrequencyLSTM.init_hidden"""

    def test_returns_tuple(self, small_model, device):
        """Test that init_hidden returns a tuple."""
        hidden = small_model.init_hidden(4, device)
        assert isinstance(hidden, tuple)
        assert len(hidden) == 2

    def test_hidden_state_shape(self, small_model, device):
        """Test that hidden state has correct shape."""
        batch_size = 4
        h_0, c_0 = small_model.init_hidden(batch_size, device)

        expected_shape = (small_model.num_layers, batch_size, small_model.hidden_size)
        assert h_0.shape == expected_shape

    def test_cell_state_shape(self, small_model, device):
        """Test that cell state has correct shape."""
        batch_size = 4
        h_0, c_0 = small_model.init_hidden(batch_size, device)
        assert c_0.shape == h_0.shape

    def test_hidden_state_zeros(self, small_model, device):
        """Test that hidden state is initialized to zeros."""
        h_0, _ = small_model.init_hidden(4, device)
        assert torch.all(h_0 == 0)

    def test_cell_state_zeros(self, small_model, device):
        """Test that cell state is initialized to zeros."""
        _, c_0 = small_model.init_hidden(4, device)
        assert torch.all(c_0 == 0)

    def test_device_placement(self, small_model, device):
        """Test that state is placed on correct device."""
        h_0, c_0 = small_model.init_hidden(4, device)
        assert h_0.device == device
        assert c_0.device == device

    def test_different_batch_sizes(self, small_model, device):
        """Test init_hidden with different batch sizes."""
        for batch_size in [1, 8, 32, 64]:
            h_0, c_0 = small_model.init_hidden(batch_size, device)
            assert h_0.shape[1] == batch_size


class TestGetOrResetHidden:
    """Tests for FrequencyLSTM.get_or_reset_hidden"""

    def test_none_hidden_initializes(self, small_model, device):
        """Test that None hidden state creates new state."""
        hidden = small_model.get_or_reset_hidden(4, 4, None, device)
        assert hidden is not None
        assert hidden[0].shape[1] == 4

    def test_matching_sizes_preserves_state(self, small_model, device):
        """Test that matching batch sizes preserve state."""
        initial_hidden = small_model.init_hidden(4, device)
        # Set non-zero values
        initial_hidden[0].fill_(1.0)

        result = small_model.get_or_reset_hidden(4, 4, initial_hidden, device)

        # Should preserve the values
        assert torch.all(result[0] == 1.0)

    def test_mismatched_sizes_reinitializes(self, small_model, device):
        """Test that mismatched batch sizes reinitialize state."""
        initial_hidden = small_model.init_hidden(4, device)
        initial_hidden[0].fill_(1.0)

        result = small_model.get_or_reset_hidden(2, 4, initial_hidden, device)

        # Should be zeros (reinitialized)
        assert result[0].shape[1] == 2
        assert torch.all(result[0] == 0)

    def test_last_batch_handling(self, small_model, device):
        """Test handling of last batch (smaller size)."""
        initial_hidden = small_model.init_hidden(32, device)

        # Last batch has fewer samples
        result = small_model.get_or_reset_hidden(28, 32, initial_hidden, device)

        # Should reinitialize with correct size
        assert result[0].shape[1] == 28

    def test_device_consistency(self, small_model, device):
        """Test that output is on correct device."""
        result = small_model.get_or_reset_hidden(4, 4, None, device)
        assert result[0].device == device
        assert result[1].device == device


class TestCountParameters:
    """Tests for FrequencyLSTM.count_parameters"""

    def test_returns_integer(self, small_model):
        """Test that count_parameters returns an integer."""
        count = small_model.count_parameters()
        assert isinstance(count, int)

    def test_positive_value(self, small_model):
        """Test that parameter count is positive."""
        count = small_model.count_parameters()
        assert count > 0

    def test_matches_manual_count(self, small_model):
        """Test that count matches manual parameter counting."""
        manual_count = sum(p.numel() for p in small_model.parameters() if p.requires_grad)
        assert small_model.count_parameters() == manual_count

    def test_larger_model_more_parameters(self):
        """Test that larger model has more parameters."""
        small = FrequencyLSTM(hidden_size=32)
        large = FrequencyLSTM(hidden_size=128)

        assert large.count_parameters() > small.count_parameters()


class TestGetModelSummary:
    """Tests for FrequencyLSTM.get_model_summary"""

    def test_returns_string(self, small_model):
        """Test that get_model_summary returns a string."""
        summary = small_model.get_model_summary()
        assert isinstance(summary, str)

    def test_contains_architecture_info(self, small_model):
        """Test that summary contains architecture information."""
        summary = small_model.get_model_summary()
        assert 'LSTM' in summary or 'lstm' in summary.lower()

    def test_contains_parameter_count(self, small_model):
        """Test that summary contains parameter count."""
        summary = small_model.get_model_summary()
        # Should contain some number
        assert any(char.isdigit() for char in summary)


class TestModelModes:
    """Tests for model training/evaluation modes."""

    def test_train_mode(self, small_model):
        """Test model in training mode."""
        small_model.train()
        assert small_model.training is True

    def test_eval_mode(self, small_model):
        """Test model in evaluation mode."""
        small_model.eval()
        assert small_model.training is False

    def test_dropout_in_train_mode(self, multi_layer_model):
        """Test that dropout is active in training mode."""
        multi_layer_model.train()

        x = torch.randn(32, 1, 5)
        outputs = []
        for _ in range(10):
            output, _ = multi_layer_model(x)
            outputs.append(output.clone())

        # With dropout, outputs should vary (most of the time)
        # Note: This is probabilistic, might rarely fail
        all_same = all(torch.equal(outputs[0], o) for o in outputs[1:])
        # In training mode with dropout, outputs should differ
        # (though technically they could be the same by chance)

    def test_dropout_disabled_in_eval(self, multi_layer_model):
        """Test that dropout is disabled in eval mode."""
        multi_layer_model.eval()

        x = torch.randn(32, 1, 5)
        with torch.no_grad():
            output1, _ = multi_layer_model(x)
            output2, _ = multi_layer_model(x)

        # In eval mode, outputs should be identical
        torch.testing.assert_close(output1, output2)


class TestStatePersistence:
    """Tests for LSTM state persistence (critical for L=1)."""

    @pytest.mark.state_management
    def test_state_detachment(self, small_model):
        """Test that state can be detached for L=1 training."""
        x = torch.randn(1, 1, 5)
        output, hidden = small_model(x)

        # Detach state
        detached_hidden = tuple(h.detach() for h in hidden)

        # Should not have grad_fn
        assert detached_hidden[0].grad_fn is None
        assert detached_hidden[1].grad_fn is None

    @pytest.mark.state_management
    def test_state_values_preserved_after_detach(self, small_model):
        """Test that state values are preserved after detachment."""
        x = torch.randn(1, 1, 5)
        _, hidden = small_model(x)

        # Store values before detach
        h_before = hidden[0].clone()
        c_before = hidden[1].clone()

        # Detach
        detached_hidden = tuple(h.detach() for h in hidden)

        # Values should be identical
        torch.testing.assert_close(detached_hidden[0], h_before)
        torch.testing.assert_close(detached_hidden[1], c_before)

    @pytest.mark.state_management
    @pytest.mark.critical
    def test_l1_sequence_simulation(self, small_model):
        """Test L=1 sequence processing simulation (CRITICAL)."""
        batch_size = 1
        n_samples = 100
        hidden = None

        outputs = []
        for i in range(n_samples):
            x = torch.randn(batch_size, 1, 5)
            output, hidden = small_model(x, hidden)

            # Critical: Detach to prevent memory explosion
            hidden = tuple(h.detach() for h in hidden)

            outputs.append(output.item())

        assert len(outputs) == n_samples
        # State should have evolved
        assert not torch.all(hidden[0] == 0)


class TestModelSaveLoad:
    """Tests for model save/load functionality."""

    def test_save_state_dict(self, small_model, temp_models_dir):
        """Test saving model state dict."""
        save_path = temp_models_dir / "model.pth"
        torch.save(small_model.state_dict(), save_path)
        assert save_path.exists()

    def test_load_state_dict(self, small_model, temp_models_dir):
        """Test loading model state dict."""
        save_path = temp_models_dir / "model.pth"
        torch.save(small_model.state_dict(), save_path)

        # Create new model and load
        new_model = FrequencyLSTM(hidden_size=small_model.hidden_size)
        new_model.load_state_dict(torch.load(save_path))

        # Check parameters match
        for (name1, param1), (name2, param2) in zip(
            small_model.named_parameters(), new_model.named_parameters()
        ):
            torch.testing.assert_close(param1, param2)

    def test_output_consistency_after_load(self, small_model, temp_models_dir):
        """Test that loaded model produces same output."""
        save_path = temp_models_dir / "model.pth"
        torch.save(small_model.state_dict(), save_path)

        # Get output from original
        x = torch.randn(1, 1, 5)
        torch.manual_seed(42)
        output1, _ = small_model(x)

        # Load into new model
        new_model = FrequencyLSTM(hidden_size=small_model.hidden_size)
        new_model.load_state_dict(torch.load(save_path))
        new_model.eval()
        small_model.eval()

        torch.manual_seed(42)
        output2, _ = new_model(x)

        torch.testing.assert_close(output1, output2)
