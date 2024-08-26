#!/bin/bash
total_epochs=5
dir="SQUAD_Medium_Dropout_100epoch_256"

# 循环运行 run.sh 脚本
for ((i=1; i<=$total_epochs; i++))
do
    echo "======================================"
    echo "Running training for epoch $i..."
    echo "======================================"
    
    # Print the current epoch for debugging
    echo "Debug: Starting epoch $i"
    
    # Ensure run.sh starts with the correct shebang
    if head -n 1 run.sh | grep -q '^#!/bin/bash'; then
        echo "run.sh has the correct shebang."
    else
        echo "Error: run.sh does not start with #!/bin/bash"
        exit 1
    fi

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

