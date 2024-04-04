#!/bin/bash

# Stop on errors
set -Eeuo pipefail

for n in $(seq 20 20 80)
do
    python openforge/evaluate_lbp_efficiency.py --num_concepts=${n}
done