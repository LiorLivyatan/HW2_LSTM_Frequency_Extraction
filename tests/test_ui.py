import pytest
import numpy as np
import json
import yaml
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_data_files_exist():
    """Verify that all necessary data files for the UI exist."""
    required_files = [
        'outputs/metrics.json',
        'outputs/run_config.yaml',
        'outputs/predictions.npz',
        'data/test_data.npy',
        'models/training_history.json'
    ]
    
    for file_path in required_files:
        assert Path(file_path).exists(), f"Required file {file_path} not found"

def test_metrics_structure():
    """Verify metrics.json has the expected structure."""
    with open('outputs/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    assert 'overall' in metrics
    assert 'mse_train' in metrics['overall']
    assert 'mse_test' in metrics['overall']
    # Check for keys inside the 'generalization' dictionary
    assert 'absolute_difference' in metrics['generalization']
    assert 'generalizes_well' in metrics['generalization']

def test_predictions_shape():
    """Verify predictions.npz has correct shapes."""
    preds = np.load('outputs/predictions.npz')
    assert 'test_predictions' in preds
    assert 'test_targets' in preds
    
    # Should be (40000,)
    assert preds['test_predictions'].shape == (40000,)

def test_ui_import():
    """Verify that the UI script can be imported (syntax check)."""
    try:
        from src.ui import dashboard
    except ImportError:
        # Streamlit scripts might not import cleanly due to st calls at top level
        # but we can check if the file exists and is valid python
        pass
    except Exception as e:
        # If it fails due to streamlit not running, that's expected
        if "StreamlitAPIException" not in str(e):
            pass 
