#!/usr/bin/env bash
# Sync the phare_load package source into docs/phare_load/ so that the
# stlite web app served from GitHub Pages stays in sync with src/.
#
# Run from the repository root:   bash docs/sync_source.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/src/phare_load"
DST="$ROOT/docs/phare_load"

mkdir -p "$DST"
cp "$SRC"/__init__.py    "$DST"/__init__.py
cp "$SRC"/config.py      "$DST"/config.py
cp "$SRC"/geometry.py    "$DST"/geometry.py
cp "$SRC"/load.py        "$DST"/load.py
cp "$SRC"/models.py      "$DST"/models.py
cp "$SRC"/plotting.py    "$DST"/plotting.py
cp "$SRC"/plotting3d.py  "$DST"/plotting3d.py

echo "Synced $(ls "$DST" | wc -l | tr -d ' ') files into docs/phare_load/"

# Rebuild docs/index.html with the Python sources inlined as base64.
python3 "$ROOT/docs/build_index.py"
