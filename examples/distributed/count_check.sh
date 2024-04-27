#!/bin/bash

read -p "Inout Layer Number:" num

if ! [[  $num =~ ^[0-9]+$ ]]; then
	echo "Please Enter a Number"
	echo 1
fi

for ((i=0; i<$num; i++))
do
	echo "===================="
	echo "Layer_$i"
	echo "===================="
	cat gate_count_layer_$i.txt
	echo "======================================"
done

