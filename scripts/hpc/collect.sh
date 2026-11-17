#!/bin/bash
# Bundle one run's results for sending back:  ./scripts/hpc/collect.sh results/hpc/<run>
# Prints a summary (jobs, CSV rows, failures) and writes results/hpc/<run>.tar.gz.
set -u
RUN_DIR=${1:?usage: collect.sh results/hpc/<run-directory>}
RUN_DIR=$(cd "$RUN_DIR" && pwd) || exit 1
NAME=$(basename "$RUN_DIR")

echo "== $NAME"
if command -v qstat >/dev/null 2>&1 && [ -f "$RUN_DIR/jobs.txt" ]; then
  still=0
  while read -r _ rest; do
    id=${rest##* }
    qstat "$id" >/dev/null 2>&1 && still=$(( still + 1 ))
  done <"$RUN_DIR/jobs.txt"
  [ "$still" -gt 0 ] && echo "WARNING: $still job(s) from this run are still queued or running"
fi
[ -f "$RUN_DIR/tests.log" ] && echo "tests: $(grep -E 'tests passed|tests failed' "$RUN_DIR/tests.log" | tail -n1)"
echo "CSV files (data rows):"
for f in "$RUN_DIR"/*.csv; do
  [ -f "$f" ] && printf '  %-28s %5d\n' "$(basename "$f")" "$(( $(wc -l <"$f") - 1 ))"
done
if [ -s "$RUN_DIR/failures.txt" ]; then
  echo "FAILED runs ($(wc -l <"$RUN_DIR/failures.txt")): see $RUN_DIR/failures.txt and logs/"
else
  echo "failed runs: none"
fi
tar --exclude "$NAME/tmp_*" -czf "$RUN_DIR.tar.gz" -C "$(dirname "$RUN_DIR")" "$NAME"
echo "bundle: $RUN_DIR.tar.gz ($(du -h "$RUN_DIR.tar.gz" | cut -f1))"
echo "copy it to your computer with, for example:"
echo "  scp <nusnet-id>@vanda.nus.edu.sg:$RUN_DIR.tar.gz ."
