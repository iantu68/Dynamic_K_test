#!/bin/bash
echo 'Run training on MoE-BERT...'
python main_temp.py \
    --model_name 'bert' \
    --train_batch_size 200\
    --eval_batch_size 2 \
    --num_epochs 2 \
    --cuda \
    --debug \
    --log_interval 10 \
    --work_dir 'logs/' \
    --moe \
    --moe-num-experts 8\
    --moe-top-k 2\
    # --use_wandb \
    ${@:2}