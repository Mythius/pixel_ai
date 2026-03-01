#!/bin/bash
# generateAndSave.sh
# Generates pixel art, runs it through pixelreduce, saves to labeler/generated/
#
# Usage:
#   ./generateAndSave.sh "a small red dragon"
#   ./generateAndSave.sh "a small red dragon" 4

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Find Python: prefer the project venv, fall back to python3/python in PATH
if [ -x "$SCRIPT_DIR/venv/bin/python" ]; then
  PYTHON="$SCRIPT_DIR/venv/bin/python"
elif command -v python3 &>/dev/null; then
  PYTHON="$(command -v python3)"
elif command -v python &>/dev/null; then
  PYTHON="$(command -v python)"
else
  echo "ERROR: No Python interpreter found. Create a venv or install Python 3."
  exit 1
fi
echo "[$(date +%H:%M:%S)] Using Python: $PYTHON"

TRAINMODEL="$SCRIPT_DIR/trainmodel/trainmodel.py"
PIXELREDUCE="$SCRIPT_DIR/pixel_reducer/pixelreduce.py"
OUTPUT_DIR="$SCRIPT_DIR/trainmodel/output"
GENERATED_DIR="$SCRIPT_DIR/labeler/generated"
META_FILE="$GENERATED_DIR/metadata.json"

USER_PROMPT="$1"
NUM="${2:-1}"
FULL_PROMPT="pixel image: (32x32) $USER_PROMPT"
TS=$(date +"%Y%m%d_%H%M%S")

# ── Validate ──────────────────────────────────────────────────────────────────
if [ -z "$USER_PROMPT" ]; then
  echo "Usage: $0 \"your prompt\" [num_images]"
  echo "  Example: $0 \"a small red dragon\" 2"
  exit 1
fi

mkdir -p "$GENERATED_DIR"

# ── Step 1: Generate ──────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Generating: \"$FULL_PROMPT\" ×$NUM"
CUDA_VISIBLE_DEVICES="" "$PYTHON" "$TRAINMODEL" generate "$FULL_PROMPT" \
  --num "$NUM" \
  --model "$SCRIPT_DIR/model1"

if [ $? -ne 0 ]; then
  echo "[$(date +%H:%M:%S)] ERROR: Generation failed"
  exit 1
fi

# ── Step 2: Collect output files ──────────────────────────────────────────────
declare -a CANDIDATES  # each entry: "filepath:index"

if [ "$NUM" -eq 1 ] && [ -f "$OUTPUT_DIR/generated.png" ]; then
  CANDIDATES+=("$OUTPUT_DIR/generated.png:1")
fi
for i in $(seq 1 "$NUM"); do
  F="$OUTPUT_DIR/generated_${i}.png"
  [ -f "$F" ] && CANDIDATES+=("$F:$i")
done

if [ ${#CANDIDATES[@]} -eq 0 ]; then
  echo "[$(date +%H:%M:%S)] ERROR: No output files found in $OUTPUT_DIR"
  exit 1
fi

# ── Step 3: Pixel-reduce + move ───────────────────────────────────────────────
declare -a SAVED_NAMES

for ENTRY in "${CANDIDATES[@]}"; do
  SRC="${ENTRY%:*}"
  IDX="${ENTRY##*:}"
  DEST_NAME="gen_${TS}_${IDX}.png"
  DEST="$GENERATED_DIR/$DEST_NAME"

  echo "[$(date +%H:%M:%S)] Reducing: $(basename "$SRC") → $DEST_NAME"
  "$PYTHON" "$PIXELREDUCE" "$SRC" "$SRC" --width 32 --colors 12

  if [ $? -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] WARN: pixel reduce failed, keeping original"
  fi

  mv "$SRC" "$DEST"
  SAVED_NAMES+=("$DEST_NAME")
  echo "[$(date +%H:%M:%S)] Saved: $DEST"
done

# ── Step 4: Update metadata.json ─────────────────────────────────────────────
# Pass data via env vars to avoid quote/escape issues with arbitrary prompt text
FILENAMES=$(IFS=","; echo "${SAVED_NAMES[*]}")

FULL_PROMPT="$FULL_PROMPT" \
USER_PROMPT="$USER_PROMPT" \
TS="$TS" \
META_FILE="$META_FILE" \
FILENAMES="$FILENAMES" \
"$PYTHON" - <<'PYEOF'
import json, os

meta_file = os.environ['META_FILE']
meta = {}
if os.path.exists(meta_file):
    with open(meta_file) as f:
        meta = json.load(f)

for fname in os.environ['FILENAMES'].split(','):
    if fname:
        meta[fname] = {
            'prompt':      os.environ['FULL_PROMPT'],
            'user_prompt': os.environ['USER_PROMPT'],
            'ts':          os.environ['TS'],
        }

with open(meta_file, 'w') as f:
    json.dump(meta, f, indent=2)

count = len([f for f in os.environ['FILENAMES'].split(',') if f])
print(f"[metadata] {count} record(s) written to {meta_file}")
PYEOF

echo "[$(date +%H:%M:%S)] Done. ${#SAVED_NAMES[@]} image(s) in $GENERATED_DIR"
