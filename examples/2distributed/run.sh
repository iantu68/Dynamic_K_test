#!/bin/bash
echo 'Run training on MoE-BERT...'
python main.py \
    --model_name 'bert' \
    --train_batch_size 100\
    --eval_batch_size 1 \
    --num_epochs 1 \
    --cuda \
    --debug \
    --log_interval 10 \
    --work_dir 'logs/' \
    --moe \
    --moe-num-experts 8\
    --moe-top-k 2\
    # --use_wandb \
    ${@:2}