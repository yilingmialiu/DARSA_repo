#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Clear any existing PYTHONPATH
export PYTHONPATH=""

# Add project root to PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:$PYTHONPATH"

# Change to project directory
cd "${SCRIPT_DIR}"

# Run the training script
python experiments/visda_2017/DARSA_train_source.py "$@" 