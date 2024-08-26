#!/bin/bash

total_epochs=1
dir="SST2_BERTTiny_100epoch_batch256"

# 循环运行 run.sh 脚本
for ((i=1; i<=$total_epochs; i++))
do
    echo "======================================"
    echo "Running training for epoch $i..."
    echo "======================================"
    python bert.py
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