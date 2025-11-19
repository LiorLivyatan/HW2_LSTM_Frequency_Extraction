# Phase 5: Visualization PRD

**Phase**: 5 of 6
**Priority**: Medium
**Estimated Effort**: 2-3 hours
**Dependencies**: Phase 4 (Evaluation - predictions needed)
**Enables**: Assignment Completion

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

Create publication-quality visualizations demonstrating successful frequency extraction from noisy signals.

### What This Phase Delivers
- **Graph 1**: Single frequency comparison (Target vs LSTM vs Noisy)
- **Graph 2**: All 4 frequencies extracted (2×2 subplot grid)
- **High-quality PNG outputs** suitable for reports
- **Clear visual evidence** of LSTM's extraction capability

### Why This Phase is Critical
- **Visual proof** of model success
- **Required for assignment** submission
- Demonstrates extraction quality better than numbers alone
- Shows LSTM learned to ignore noise

---

## Requirements

### Functional Requirements

#### FR1: Graph 1 - Single Frequency Comparison
**Required elements**:
- [ ] **Three signals overlaid** on same plot:
  1. Ground truth target (clean sinusoid) - **line plot**
  2. LSTM output (predictions) - **scatter plot or line**
  3. Noisy mixed input signal - **semi-transparent line**
- [ ] **Frequency**: Use f₂ = 3Hz (as shown in assignment)
- [ ] **Data source**: Test set (Seed #2) - demonstrates generalization
- [ ] **Time window**: First 1 second (1000 samples) for clarity
- [ ] **Axes labels**: X-axis: "Time (seconds)", Y-axis: "Amplitude"
- [ ] **Legend**: Clear labels for each signal
- [ ] **Title**: "Frequency Extraction: f₂ = 3Hz (Test Set)"

#### FR2: Graph 2 - All Four Frequencies
**Required elements**:
- [ ] **Layout**: 2×2 subplot grid
- [ ] **Each subplot** shows:
  - Ground truth target (line)
  - LSTM output (line or scatter)
- [ ] **Subplots**: One for each frequency (f₁, f₂, f₃, f₄)
- [ ] **Data source**: Test set (Seed #2)
- [ ] **Time window**: First 1 second (1000 samples)
- [ ] **Individual titles**: "f₁ = 1Hz", "f₂ = 3Hz", etc.
- [ ] **Shared axes labels**
- [ ] **Overall title**: "Extracted Frequencies (Test Set)"

#### FR3: Visual Quality
- [ ] Figure size: 12×8 inches (or similar for readability)
- [ ] DPI: 300 (publication quality)
- [ ] Color scheme: Distinguishable, colorblind-friendly
- [ ] Font sizes: Readable (title:14, labels:12, ticks:10)
- [ ] Grid: Optional, subtle if used

#### FR4: File Output
- [ ] Save as PNG format
- [ ] Filenames: `frequency_comparison.png`, `all_frequencies.png`
- [ ] Save to `outputs/graphs/` directory

### Non-Functional Requirements

#### NFR1: Code Quality
- Clean, modular visualization code
- Configurable parameters (colors, sizes, etc.)
- Reusable functions

#### NFR2: Performance
- Graph generation: < 10 seconds
- Memory efficient

---

## Architecture

### Visualizer Class Structure

```
Visualizer
    │
    ├── __init__(predictions_data)
    │   └── Load predictions from Phase 4
    │
    ├── prepare_data_for_plotting()
    │   └── Extract and structure data for graphs
    │
    ├── plot_single_frequency_comparison()
    │   └── Create Graph 1
    │
    ├── plot_all_frequencies()
    │   └── Create Graph 2
    │
    └── save_figure()
        └── Save with proper DPI and format
```

### Data Flow

```
Phase 4 Outputs (predictions.npz)
    ├── test_predictions  (40,000 values)
    ├── test_targets      (40,000 values)
    └── (implicitly) test noisy input from data file
        ↓
Extract relevant data
    ├── Frequency f₂ (rows 10,000-19,999)
    ├── First 1000 samples (0-1 second)
    └── Corresponding noisy input S(t)
        ↓
Plot Graph 1
    ├── Target (blue line)
    ├── LSTM Output (green dots/line)
    └── Noisy Input (red semi-transparent)
        ↓
Save: outputs/graphs/frequency_comparison.png
```

---

## Implementation Details

### Libraries Used

| Library | Purpose |
|---------|---------|
| **matplotlib** | Plotting |
| **numpy** | Data manipulation |
| **pathlib** | File handling |

### Visualizer Class

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple

class Visualizer:
    """
    Create visualizations for LSTM frequency extraction results.

    Args:
        predictions_path: Path to predictions.npz from Phase 4
        data_path: Path to test_data.npy for noisy signal
    """

    def __init__(
        self,
        predictions_path: str = 'outputs/predictions.npz',
        data_path: str = 'data/test_data.npy'
    ):
        # Load predictions from Phase 4
        pred_data = np.load(predictions_path)
        self.test_predictions = pred_data['test_predictions']
        self.test_targets = pred_data['test_targets']

        # Load test data for noisy signal
        test_data = np.load(data_path)
        self.test_noisy = test_data[:, 0]  # S(t) column

        # Data structure: 40,000 rows = 4 frequencies × 10,000 samples
        self.n_samples_per_freq = 10000
        self.n_frequencies = 4
        self.frequencies = [1, 3, 5, 7]  # Hz

        # Time array (0-10 seconds, 10,000 points)
        self.time_full = np.linspace(0, 10, self.n_samples_per_freq)

    def extract_frequency_data(
        self,
        freq_idx: int,
        time_window: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract data for a specific frequency.

        Args:
            freq_idx: Frequency index (0-3)
            time_window: Number of samples to include (None = all)

        Returns:
            tuple: (time, target, prediction, noisy)
        """
        # Calculate row indices for this frequency
        start_idx = freq_idx * self.n_samples_per_freq
        end_idx = (freq_idx + 1) * self.n_samples_per_freq

        # Extract data
        target = self.test_targets[start_idx:end_idx]
        prediction = self.test_predictions[start_idx:end_idx]
        noisy = self.test_noisy[start_idx:end_idx]
        time = self.time_full

        # Apply time window if specified
        if time_window is not None:
            target = target[:time_window]
            prediction = prediction[:time_window]
            noisy = noisy[:time_window]
            time = time[:time_window]

        return time, target, prediction, noisy

    def plot_single_frequency_comparison(
        self,
        freq_idx: int = 1,  # Default: f₂ = 3Hz
        time_window: int = 1000,  # First 1 second
        save_path: str = 'outputs/graphs/frequency_comparison.png'
    ) -> plt.Figure:
        """
        Create Graph 1: Single frequency comparison.

        Shows target, LSTM output, and noisy input overlaid.

        Args:
            freq_idx: Which frequency to plot (0-3)
            time_window: Number of samples to show
            save_path: Where to save the figure

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Extract data
        time, target, prediction, noisy = self.extract_frequency_data(
            freq_idx,
            time_window
        )

        freq_hz = self.frequencies[freq_idx]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot noisy input (background, semi-transparent)
        ax.plot(
            time,
            noisy,
            color='red',
            alpha=0.3,
            linewidth=1,
            label='Noisy Mixed Input S(t)'
        )

        # Plot target (ground truth)
        ax.plot(
            time,
            target,
            color='blue',
            linewidth=2,
            label='Ground Truth Target',
            linestyle='--'
        )

        # Plot LSTM output
        ax.plot(
            time,
            prediction,
            color='green',
            linewidth=1.5,
            label='LSTM Output',
            marker='.',
            markersize=3,
            markevery=10  # Show markers every 10 points
        )

        # Formatting
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title(
            f'Frequency Extraction: f{freq_idx+1} = {freq_hz}Hz (Test Set)',
            fontsize=14,
            fontweight='bold'
        )
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Tight layout
        plt.tight_layout()

        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

        return fig

    def plot_all_frequencies(
        self,
        time_window: int = 1000,
        save_path: str = 'outputs/graphs/all_frequencies.png'
    ) -> plt.Figure:
        """
        Create Graph 2: All four frequencies in 2×2 grid.

        Each subplot shows target vs LSTM output for one frequency.

        Args:
            time_window: Number of samples to show
            save_path: Where to save the figure

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Create figure with 2×2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()  # Flatten to 1D for easier indexing

        for freq_idx in range(4):
            ax = axes[freq_idx]

            # Extract data
            time, target, prediction, _ = self.extract_frequency_data(
                freq_idx,
                time_window
            )

            freq_hz = self.frequencies[freq_idx]

            # Plot target
            ax.plot(
                time,
                target,
                color='blue',
                linewidth=2,
                label='Target',
                linestyle='--',
                alpha=0.7
            )

            # Plot LSTM output
            ax.plot(
                time,
                prediction,
                color='green',
                linewidth=1.5,
                label='LSTM Output'
            )

            # Formatting
            ax.set_title(
                f'f{freq_idx+1} = {freq_hz}Hz',
                fontsize=12,
                fontweight='bold'
            )
            ax.set_xlabel('Time (seconds)', fontsize=10)
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)

        # Overall title
        fig.suptitle(
            'Extracted Frequencies (Test Set)',
            fontsize=16,
            fontweight='bold',
            y=0.995
        )

        # Tight layout
        plt.tight_layout()

        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

        return fig

    def create_all_visualizations(
        self,
        output_dir: str = 'outputs/graphs'
    ) -> None:
        """
        Create all required visualizations.

        Args:
            output_dir: Directory to save graphs
        """
        print("Creating visualizations...")

        # Graph 1: Single frequency comparison
        print("\n1. Creating frequency comparison graph...")
        self.plot_single_frequency_comparison(
            freq_idx=1,  # f₂ = 3Hz
            save_path=f"{output_dir}/frequency_comparison.png"
        )

        # Graph 2: All frequencies
        print("\n2. Creating all frequencies graph...")
        self.plot_all_frequencies(
            save_path=f"{output_dir}/all_frequencies.png"
        )

        print("\n✓ All visualizations created successfully!")
```

### Visualization Script

```python
from src.visualization import Visualizer

def main():
    # Create visualizer
    visualizer = Visualizer(
        predictions_path='outputs/predictions.npz',
        data_path='data/test_data.npy'
    )

    # Create all graphs
    visualizer.create_all_visualizations(output_dir='outputs/graphs')

if __name__ == '__main__':
    main()
```

### Color Scheme Recommendations

```python
# Colorblind-friendly palette
COLORS = {
    'target': '#0173B2',      # Blue
    'prediction': '#029E73',   # Green
    'noisy': '#DE8F05',        # Orange
    'error': '#CC78BC'         # Purple
}

# Alternative: Use matplotlib's built-in colorblind-friendly palettes
# plt.style.use('seaborn-colorblind')
```

---

## Testing Strategy

### Visual Inspection Tests

#### Test 1: Graph Elements Present
```python
def test_graph_1_elements():
    """Verify Graph 1 has all required elements."""
    visualizer = Visualizer()
    fig = visualizer.plot_single_frequency_comparison(save_path=None)

    ax = fig.axes[0]

    # Check title exists
    assert ax.get_title() != ''

    # Check labels exist
    assert ax.get_xlabel() != ''
    assert ax.get_ylabel() != ''

    # Check legend exists
    assert ax.get_legend() is not None

    # Check 3 lines plotted (noisy, target, prediction)
    assert len(ax.lines) == 3
```

#### Test 2: Subplot Grid
```python
def test_graph_2_subplots():
    """Verify Graph 2 has 2×2 subplot grid."""
    visualizer = Visualizer()
    fig = visualizer.plot_all_frequencies(save_path=None)

    # Should have 4 subplots
    assert len(fig.axes) == 4

    # Each subplot should have title
    for ax in fig.axes:
        assert ax.get_title() != ''
```

### Data Integrity Tests

#### Test 3: Time Window
```python
def test_time_window():
    """Test that time window limits data correctly."""
    visualizer = Visualizer()

    time, target, prediction, noisy = visualizer.extract_frequency_data(
        freq_idx=0,
        time_window=1000
    )

    assert len(time) == 1000
    assert len(target) == 1000
    assert len(prediction) == 1000
    assert len(noisy) == 1000
```

#### Test 4: Frequency Extraction
```python
def test_frequency_extraction():
    """Test that correct frequency data is extracted."""
    visualizer = Visualizer()

    # Extract f₂ data
    time, target, prediction, noisy = visualizer.extract_frequency_data(
        freq_idx=1
    )

    # Should be 10,000 samples
    assert len(time) == 10000
    assert len(target) == 10000
```

---

## Deliverables

### Code Files
- [ ] `src/visualization.py` - Visualizer class
- [ ] `visualize.py` - Visualization script
- [ ] `tests/test_visualization.py` - Tests

### Output Files
- [ ] `outputs/graphs/frequency_comparison.png` - Graph 1
- [ ] `outputs/graphs/all_frequencies.png` - Graph 2

### Documentation
- [ ] Graph interpretation guide
- [ ] Color scheme documentation

---

## Success Criteria

### Visual Quality
- [ ] Graphs are clear and readable
- [ ] Labels and titles are descriptive
- [ ] Colors are distinguishable
- [ ] High resolution (300 DPI)

### Content Accuracy
- [ ] Correct data displayed (test set)
- [ ] Correct frequency shown
- [ ] Time window is 0-1 second
- [ ] All 4 frequencies in Graph 2

### Demonstration of Success
- [ ] **Visual evidence**: LSTM output closely matches target
- [ ] **Clear separation**: LSTM output is clean vs. noisy input
- [ ] **Consistency**: All 4 frequencies extracted well

---

## Risks and Mitigation

### Risk 1: Plots Don't Show Clear Extraction
**Risk**: If model didn't learn well, plots will show poor match

**Impact**: HIGH - Visual failure evident

**Mitigation**:
- Go back to Phase 3 and retrain
- Adjust hyperparameters
- Verify data quality

### Risk 2: Cluttered Visualizations
**Risk**: Too much information, hard to read

**Impact**: MEDIUM - Poor presentation

**Mitigation**:
- Use 1-second window (not full 10 seconds)
- Adjust transparency and line widths
- Use clear legend placement

---

## Dependencies

### Depends On
- **Phase 4 (Evaluation)**: Needs predictions

---

## Estimated Effort

| Activity | Time Estimate |
|----------|---------------|
| Implement Visualizer class | 1-1.5 hours |
| Create and refine plots | 0.5-1 hour |
| Testing and adjustments | 0.5 hour |
| **Total** | **2-2.5 hours** |

---

## Next Steps

After completing Phase 5:
1. Verify graphs look professional
2. Confirm clear demonstration of extraction
3. Proceed to [Phase 6: Integration](06_INTEGRATION_PRD.md)

---

**Status**: Ready for Implementation
**Last Updated**: 2025-11-16
