#!/bin/bash

set -e
set -o pipefail

python cli.py \
    --max_eps 200 \
    --max_steps 6000 \
    --batch_size 64 \
    --logging_level critical \
    --render \
    --epsilon 0.6 \
    --learning_rate 0.01 \
    --train_per_episode 16 \
    --rounds_per_episode 1 \
    --game_id Assault-v0 \
    --update_freq 1 \
    --model_location=assault \
    --checkpoint_location=assault \
    --eval_freq=20