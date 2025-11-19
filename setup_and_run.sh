#!/bin/bash

# LSTM Frequency Extraction - Setup and Run Script
# This script handles environment setup (venv), dependency installation,
# and executes the full pipeline with UI launch.

set -e  # Exit immediately if a command exits with a non-zero status.

VENV_DIR=".venv"
REQUIREMENTS_FILE="requirements.txt"

# 1. Check/Create Virtual Environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment found in $VENV_DIR."
fi

# 2. Activate Virtual Environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# 3. Install Dependencies
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing dependencies from $REQUIREMENTS_FILE..."
    pip install --upgrade pip
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "WARNING: $REQUIREMENTS_FILE not found. Skipping dependency installation."
fi

# 4. Run Pipeline with UI
echo "Starting LSTM Frequency Extraction Pipeline..."
echo "Running: python main.py --mode all --launch-ui"
python main.py --mode all --launch-ui
