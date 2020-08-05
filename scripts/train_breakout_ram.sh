#!/bin/bash

set -e
set -o pipefail

python cli.py \
    --max_eps 512 \
    --max_steps 2048 \
    --batch_size 256 \
    --logging_level critical \
    --epsilon 0.8 \
    --learning_rate 0.002 \
    --train_per_episode 32 \
    --rounds_per_episode 10 \
    --game_id Breakout-ram-v0 \
    --update_freq 1 \
    --model_location=breakout-ram \
    --max_eval_eps 10 \
    --max_buffer_size 100000 \
    --checkpoint_location=breakout-ram \
    --eval_freq=1 --render --mode train