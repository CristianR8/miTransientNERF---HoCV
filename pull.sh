#!/usr/bin/env bash
set -euo pipefail

REMOTE_BASE="cvail:~/miTransientNERF/results"
LOCAL_DEST="."

if [ "$#" -lt 1 ]; then
  echo "Uso: $0 <archivo-o-carpeta-remoto> [archivo-o-carpeta-remoto ...]" >&2
  exit 1
fi

for item in "$@"; do
  rsync -avh --progress "${REMOTE_BASE}/${item}" "$LOCAL_DEST"
done
