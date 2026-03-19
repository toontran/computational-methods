#!/bin/bash

# Fixed experiment settings
method_name="isvd"

# List of k values
k_values=(
    1
    2
    4
    32
    128
)

# List of matrices
matrices=(
    "kernel_stocks_1000_0.7071"
    "kernel_stocks_1000_1.0"
    "kernel_stocks_1000_2.2361"
    # "bad_case1_1000"
    # "bad_case2_1000"
    # "bad_case3_1000"
    # "FIDAP/ex3"
    # "HB/plat1919"
    # "HB/1138_bus"
)

# Window sizes to sweep
win_sizes=(
    # 4
    16
    64
)

usage() {
    total_combinations=$((${#matrices[@]} * ${#k_values[@]} * ${#win_sizes[@]}))

    echo "Usage: $0 <experiment_number>"
    echo "Experiment number should be between 0 and $((total_combinations - 1))"
    echo ""
    echo "Fixed settings:"
    echo "  Method: $method_name"
    echo ""
    echo "Index mapping:"
    echo "  matrix index   = experiment_number / (${#k_values[@]} * ${#win_sizes[@]})"
    echo "  within-matrix  = experiment_number % (${#k_values[@]} * ${#win_sizes[@]})"
    echo "  k index        = within-matrix / ${#win_sizes[@]}"
    echo "  win_size index = within-matrix % ${#win_sizes[@]}"
    echo ""

    echo "Available matrices:"
    for i in "${!matrices[@]}"; do
        echo "  $i: ${matrices[i]}"
    done

    echo ""
    echo "Available k values:"
    for i in "${!k_values[@]}"; do
        echo "  $i: k=${k_values[i]}"
    done

    echo ""
    echo "Available win_size values:"
    for i in "${!win_sizes[@]}"; do
        echo "  $i: win_size=${win_sizes[i]}"
    done

    echo ""
    echo "mem_size = k + win_size"
}

# Check argument
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

experiment_number=$1
total_combinations=$((${#matrices[@]} * ${#k_values[@]} * ${#win_sizes[@]}))

# Validate experiment number
if ! [[ "$experiment_number" =~ ^[0-9]+$ ]] || [ "$experiment_number" -ge "$total_combinations" ]; then
    echo "Error: Invalid experiment number."
    echo "Experiment number should be between 0 and $((total_combinations - 1))"
    usage
    exit 1
fi

# Map experiment number to matrix, k, and win_size
per_matrix=$((${#k_values[@]} * ${#win_sizes[@]}))
matrix_index=$((experiment_number / per_matrix))
within_matrix=$((experiment_number % per_matrix))
k_index=$((within_matrix / ${#win_sizes[@]}))
win_index=$((within_matrix % ${#win_sizes[@]}))

matrix_name="${matrices[$matrix_index]}"
k="${k_values[$k_index]}"
win_size="${win_sizes[$win_index]}"
mem_size=$((k + win_size))

# Create logs directory
mkdir -p logs

# Safe log filename
safe_matrix_name="${matrix_name//\//_}"
log_filename="logs/${safe_matrix_name}_${method_name}_mem${mem_size}_win${win_size}_k${k}.txt"

# Display selected configuration
echo "Running experiment number: $experiment_number"
echo "Matrix ($matrix_index): $matrix_name"
echo "Method: $method_name"
echo "k ($k_index): $k"
echo "win_size ($win_index): $win_size"
echo "mem_size: $mem_size"

# Run the Python script
echo "Running main.py ..."
python3 -u main.py "$matrix_name" "$method_name" "$win_size" "$k" 2>&1 | tee -a "$log_filename"