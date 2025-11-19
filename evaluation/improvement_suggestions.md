# Improvement Suggestions

## Testing & QA
- Add `tests/` with PyTest covering: `SignalGenerator` (shape, per-sample randomization), `StatefulTrainer` (state detach, OOM prevention), and `Evaluator` (per-frequency slicing logic). Target `pytest --cov=src --maxfail=1` with >70% coverage.
- Provide CPU-only smoke tests to keep CI fast; mark GPU tests with `@pytest.mark.gpu` and skip when `torch.cuda.is_available()` is false.

## Reproducibility & Config
- Validate `config.yaml` via a lightweight schema (e.g., `cerberus` or custom checks) to prevent silent misconfigurations. Log resolved config alongside a run UUID in `outputs/run_config.yaml` (already present—enforce parity).
- Add CLI overrides for key params (hidden size, batch size, epochs) while persisting the final, resolved config for every run.

## Model & Training
- Introduce early stopping and a cosine LR scheduler; log LR per epoch. Persist `checkpoint_epoch_<N>.pth` at a low cadence (e.g., every 10 epochs).
- Explore modest regularization (weight decay ~1e-4) and a 2-layer LSTM variant; compare via the existing tables.

## Metrics & Analysis
- Add MAE and correlation per frequency to complement MSE. Emit a compact `outputs/summary.md` with overall and per-frequency metrics for quick review.
- Add confidence bands to visualization (shaded error regions) and a small panel for generalization deltas.

## Packaging & DX
- Add `pre-commit` with `black`, `flake8`, and `mypy` to enforce consistency. Include `Makefile` shortcuts: `make setup`, `make all`, `make test`.
- Publish a `tests/README.md` explaining dataset fixtures (subset `.npy` with 400 samples) to keep tests fast and deterministic.

## Documentation
- Add an “Experiment Playbook” section to `README.md` that lists recommended hyperparameter grids and expected outcome ranges.
- Include a short “Limitations” note: absolute error remains above 0.01 MSE target; paths to reduce error (capacity, epochs, scheduler).

## Optional Extensions
- Provide a Jupyter notebook demo for interactive inspection and FFT verification.
- Emit ONNX export for inference and confirm parity on a 1k-sample slice.

