#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_extremes.sh — clean-and-rerun helper for the extremes analysis
#
# Wipes stale figures/tables/NetCDFs from a previous run so that no obsolete
# output (e.g. removed indices such as R10mm / Rx1day / Rx5day / R95p) lingers,
# then re-runs script2_extremes.py.  Optionally chains script3_drivers.py.
#
# Usage:
#   ./run_extremes.sh            # clean + run script2 only
#   ./run_extremes.sh --drivers  # clean + run script2, then script3
#   ./run_extremes.sh --no-clean # run script2 without wiping previous output
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

CLEAN=1
RUN_DRIVERS=0
for arg in "$@"; do
    case "$arg" in
        --no-clean) CLEAN=0 ;;
        --drivers)  RUN_DRIVERS=1 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

cd "$(dirname "$0")"

if [[ "$CLEAN" -eq 1 ]]; then
    echo "Cleaning stale extremes output (figures / tables / netcdf) …"
    rm -rf output_extremes/figures output_extremes/tables output_extremes/netcdf
    mkdir -p output_extremes/figures output_extremes/tables output_extremes/netcdf
fi

echo "Running script2_extremes.py …"
python script2_extremes.py

if [[ "$RUN_DRIVERS" -eq 1 ]]; then
    echo "Running script3_drivers.py …"
    python script3_drivers.py
fi

echo "Done.  Output under output_extremes/  (and output_drivers/ if --drivers)."
