#!/bin/bash
echo 'Run training on MoE-BERT...'
python main.py \
    --model_name 'bert' \
    --train_batch_size 256 \
    --eval_batch_size 256 \
    --num_epochs 2\
    --cuda \
    --debug \
    --log_interval 1 \
    --work_dir 'logs/' \
    --moe \
    --moe-num-experts 8\
    --moe-top-k 2\
    # --use_wandb \
    ${@:2}
