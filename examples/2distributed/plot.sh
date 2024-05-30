#!/bin/bash
echo 'Plot Expert Mean Gradients Value...'
python plot_expert_grads.py
echo 'Done!'
echo '=============================='
echo 'Plot Gate Count ...'
python gate_count_plot.py
echo 'Done!'
echo '=============================='
echo 'Plot loss plot...'
python loss_plot.py
echo 'Done!'



# echo '=============================='
# echo 'Plot Gate Gradients Value...'
# python gate_grads_plot.py
# echo 'Done!'
# echo '=============================='
# echo 'Plot Gate Gradients Slpoe Value...'
# python expert_slope_plot.py
# echo 'Done!'
# echo '=============================='
# echo 'Plot Experts  Slpoe Value...'
# python expert_slope_plot.py
# echo 'Done!'

