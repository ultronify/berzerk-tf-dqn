#!/bin/bash

set -e
set -o pipefail

python cli.py \
    --max_eps 2048 \
    --max_steps 2048 \
    --batch_size 512 \
    --logging_level critical \
    --epsilon 0.85 \
    --epsilon_decay 0.9 \
    --learning_rate 0.002 \
    --train_per_episode 1 \
    --rounds_per_episode 16 \
    --game_id Breakout-v0 \
    --update_freq 1 \
    --model_location=breakout \
    --max_eval_eps 10 \
    --max_buffer_size 10000 \
    --checkpoint_location=breakout \
    --eval_freq=1 --render --mode train