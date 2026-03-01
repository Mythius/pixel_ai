#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUDA_VISIBLE_DEVICES="" "$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/trainmodel/trainmodel.py" generate "$1" --num "$2" --model "$SCRIPT_DIR/model1"
