#!/bin/bash

# Run baseline training (no reasoning vocabulary)
# Note: Hydra configs are in exp/conf/
# Override parameters with: key=value (e.g., training.learning_rate=1e-5)
uv run exp/grpo_train.py \
    model=monad \
    training=grpo_monad_4090
