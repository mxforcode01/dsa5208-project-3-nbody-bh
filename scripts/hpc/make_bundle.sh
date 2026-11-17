#!/bin/bash
# Packs the current commit for copying to a cluster:  ./scripts/hpc/make_bundle.sh
# The bundle records the commit in SOURCE_VERSION, so results can be traced to it without a
# .git folder, and keeps Unix line endings and executable bits. Refuses uncommitted changes to tracked files (untracked files are never packed).
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "ERROR: commit your changes first:" >&2; git status --short --untracked-files=no >&2; exit 1; }
hash=$(git rev-parse --short HEAD)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git -c core.autocrlf=false archive --prefix=dsa5208-project-3-nbody/ HEAD | tar -x -C "$tmp"
printf '%s\n' "$hash" >"$tmp/dsa5208-project-3-nbody/SOURCE_VERSION"
out="dsa5208-project-3-nbody-$hash.tar.gz"
tar -czf "$out" -C "$tmp" dsa5208-project-3-nbody
echo "$PWD/$out"
