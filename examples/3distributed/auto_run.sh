#!/bin/bash

total_epochs=5
dir="New_Medium_Base_10epoch_batch16"

# 循环运行 run.sh 脚本
for ((i=1; i<=$total_epochs; i++))
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