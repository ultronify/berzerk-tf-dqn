#!/bin/bash

set -e
set -o pipefail

python cli.py \
    --max_eps 20 \
    --max_steps 200 \
    --batch_size 32 \
    --logging_level critical \
    --render \
    --epsilon 0.6 \
    --learning_rate 0.01 \
    --train_per_episode 64 \
    --rounds_per_episode 10 \
    --game_id Skiing-v0 \
    --update_freq 1 \
    --model_location=skiing \
    --checkpoint_location=skiing \
    --eval_freq=5