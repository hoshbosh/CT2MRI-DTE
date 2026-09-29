#!/bin/bash
# Alignment guards for the label pipeline. All run offline on synthetic data.
set -euo pipefail
cd "$(dirname "$0")/../.."
for t in brain_dataset_utils/tests/test_*.py; do
    echo "=== $t"
    python -u "$t" > /dev/null && echo "  PASS" || { echo "  FAIL"; exit 1; }
done
echo "all alignment guards pass"
