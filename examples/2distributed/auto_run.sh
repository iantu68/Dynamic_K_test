#!/bin/bash

total_epochs=10
dir="TinyBERT_batch64_60Epochs_Expert_dropout_0.5"

# 循环运行 run.sh 脚本
for ((i=0; i<=$total_epochs; i++))
do
    echo "======================================"
    echo "Running training for epoch $i..."
    echo "======================================"
    sh run.sh
    echo "Training for epoch $i completed."

    echo "======================================"
    echo "Plot Graph..."
    python plot_fmoe.py
    echo "Plot Graph successfully."

    echo "======================================"
    echo "Moving File..."
    mkdir -p $dir/$i
    mv *.npy *.txt *.png $dir/$i
    echo "Files moved successfully."
done