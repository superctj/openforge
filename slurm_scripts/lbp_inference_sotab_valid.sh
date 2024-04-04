#!/bin/bash

python openforge/mrf_inference_lbp.py \
    --prior_data=/home/congtj/openforge/exps/arts-context_top-40-nodes/sotab_v2_test_mrf_data_valid_with_ml_prior.csv \
    --ternary_alpha=0.8111335981502871 \
    --ternary_beta=0.8404549755313954 \
    --ternary_gamma=0.301188420089518 \
    --num_iters=860 \
    --damping=0.9937032396217518 \
    --temperature=0.4049583208001407 \
    --log_dir=/home/congtj/openforge/logs/sotab_v2_test/pgmax_lbp