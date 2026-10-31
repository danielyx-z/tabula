#!/bin/bash
# Sync the Tabula source tree from this workstation to the WATcloud workspace.
#
# The cluster copy is a plain rsync checkout (not a git clone), because the
# GitHub repo is private and the login nodes have no deploy key. This script is
# the update path: it pushes code only and never touches cluster-local state.
#
#   ./scripts/sync_to_watcloud.sh              # sync + verify import
#   ./scripts/sync_to_watcloud.sh --dry-run    # show what would change
#   REMOTE=watcloud DEST='~/tabula_gs' ./scripts/sync_to_watcloud.sh
#
# Never synced (cluster-local, and would be catastrophic to overwrite):
#   data/     processed shards + .chunk_cache
#   runs/     checkpoints
#   .venv/    5 GiB virtualenv
#   logs/     Slurm job output
#
# Do not sync while a training job is running if you are changing src/: the job
# has already imported its modules, so it will not pick up the change, but the
# workspace will no longer match what the job is running.

set -euo pipefail

REMOTE="${REMOTE:-watcloud}"
DEST="${DEST:-~/tabula_gs}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Everything that is code or documentation. Explicit list on purpose: an
# `rsync -a .` would ship data/, runs/ and .venv/.
PATHS=(
    src tests scripts docs
    README.md AGENTS.md PLAN.md
    pyproject.toml requirements.txt .gitignore
)

cd "$REPO_DIR"

echo "==> $REPO_DIR -> $REMOTE:$DEST"

if [[ "${1:-}" == "--dry-run" ]]; then
    rsync -azn --itemize-changes \
          --exclude='__pycache__' --exclude='*.pyc' \
          "${PATHS[@]}" "$REMOTE:$DEST/"
    echo "==> dry run: nothing written"
    exit 0
fi

rsync -az --itemize-changes \
      --exclude='__pycache__' --exclude='*.pyc' \
      "${PATHS[@]}" "$REMOTE:$DEST/"

# Fail loudly if the remote tree cannot even import the training entrypoint.
# This catches the classic mistake of syncing a module that imports a file the
# sync did not include.
if ssh "$REMOTE" "test -x $DEST/.venv/bin/python"; then
    echo "==> verifying import on $REMOTE"
    ssh "$REMOTE" "cd $DEST && PYTHONPATH=src .venv/bin/python -c \
        'import model.train, corpus.chunk_store, corpus.dataset; print(\"import ok\")'"
else
    echo "==> no venv at $REMOTE:$DEST/.venv; skipping import check"
fi

echo "==> done. Run tests on the cluster with:"
echo "    ssh $REMOTE 'cd $DEST && for t in tests/test_*.py; do"
echo "        printf \"%-28s \" \$t; PYTHONPATH=src .venv/bin/python \$t >/dev/null 2>&1 && echo PASS || echo FAIL; done'"
