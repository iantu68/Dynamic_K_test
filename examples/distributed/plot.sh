#!/bin/bash
echo 'Plot Expert Gradients Value...'
python expert_grads_plot.py
echo 'Done!'
echo '=============================='
echo 'Plot Gate Gradients Value...'
python gate_grads_plot.py
echo 'Done!'
echo '=============================='
echo 'Plot Gate Gradients Slpoe Value...'
python expert_slope_plot.py
echo 'Done!'
