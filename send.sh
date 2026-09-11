#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-cvail-emergency}"
REMOTE_DIR="${REMOTE_DIR:-~/miTransientNERF}"
DESTINATION="${REMOTE}:${REMOTE_DIR%/}/"

if [ "$#" -lt 1 ]; then
  echo "Uso: $0 <archivo-o-carpeta> [archivo-o-carpeta ...]" >&2
  exit 1
fi

for item in "$@"; do
  if [ ! -e "$item" ]; then
    echo "Error: '$item' no existe." >&2
    exit 1
  fi
done

# --relative preserves paths such as loaders/loader_synthetic.py instead of
# flattening every explicitly listed file into the remote project root.
rsync -avh --progress --relative \
  --partial \
  --partial-dir='.rsync-partial' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  "$@" "$DESTINATION"
