#!/bin/bash
# ==============================================================================
# AWS Pipeline Runner with Live Logging
# Runs inside tmux or background so SSH disconnect does not stop execution
# ==============================================================================
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

LOG_FILE="pipeline_aws_$(date +%Y%m%d_%H%M%S).log"

echo "=============================================================================="
echo " Starting Full Entity Resolution Pipeline on AWS"
echo " Logging to: $LOG_FILE"
echo "=============================================================================="

# Enable unbuffered python output for real-time logging
export PYTHONUNBUFFERED=1

# Run the full pipeline with live tee to terminal and log file
python3 -m src.pipeline --phase full 2>&1 | tee "$LOG_FILE"

echo "=============================================================================="
echo " Pipeline execution finished!"
echo " Output files located in: output/"
echo "=============================================================================="
