#!/bin/bash

# Fixed experiment settings
method_name="isvd"
mem_size=110 #110, 129

# List of matrices
matrices=(
    # "kernel_stocks_1000_0.7071"
    # "kernel_stocks_1000_1.0"
    # "kernel_stocks_1000_2.2361"
    "bad_case1_1000"
    "bad_case2_1000"
    "bad_case3_1000"
    # "FIDAP/ex3"
    # "HB/plat1919"
    # "HB/1138_bus"
)

# Manually specified k values
# k_values=(
#     2
#     4
#     8
#     32
#     64
#     128
# )
k_values=(
    1
    4
    8
    32
    64
    102
    106
    # 108
    109
)

usage() {
    echo "Usage: $0 <experiment_number>"
    echo "Experiment number should be between 0 and $((${#matrices[@]} * ${#k_values[@]} - 1))"
    echo ""
    echo "Fixed settings:"
    echo "  Method:   $method_name"
    echo "  mem_size: $mem_size"
    echo "  win_size: mem_size - k"
    echo ""
    echo "Matrix index = experiment_number / ${#k_values[@]}"
    echo "k index      = experiment_number % ${#k_values[@]}"
    echo ""

    echo "Available matrices:"
    for i in "${!matrices[@]}"; do
        echo "  $i: ${matrices[i]}"
    done

    echo ""
    echo "Available k values:"
    for i in "${!k_values[@]}"; do
        echo "  $i: ${k_values[i]}    -> win_size=$((mem_size - ${k_values[i]}))"
    done

    echo ""
    echo "Example:"
    example_exp=15
    example_matrix_index=$((example_exp / ${#k_values[@]}))
    example_k_index=$((example_exp % ${#k_values[@]}))
    if [ "$example_exp" -lt "$((${#matrices[@]} * ${#k_values[@]}))" ]; then
        echo "  experiment_number=$example_exp"
        echo "  matrix_index=$example_matrix_index -> ${matrices[$example_matrix_index]}"
        echo "  k_index=$example_k_index -> k=${k_values[$example_k_index]}"
        echo "  win_size=$((mem_size - ${k_values[$example_k_index]}))"
    fi
}

# Check argument
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

experiment_number=$1
total_combinations=$((${#matrices[@]} * ${#k_values[@]}))

# Validate experiment number
if ! [[ "$experiment_number" =~ ^[0-9]+$ ]] || [ "$experiment_number" -ge "$total_combinations" ]; then
    echo "Error: Invalid experiment number."
    echo "Experiment number should be between 0 and $((total_combinations - 1))"
    usage
    exit 1
fi

# Map experiment number to matrix and k
matrix_index=$((experiment_number / ${#k_values[@]}))
k_index=$((experiment_number % ${#k_values[@]}))

matrix_name="${matrices[$matrix_index]}"
k="${k_values[$k_index]}"
win_size=$((mem_size - k))

# Safety check
if [ "$win_size" -le 0 ]; then
    echo "Error: win_size must be positive, but got win_size=$win_size from mem_size=$mem_size and k=$k"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Safe log filename
safe_matrix_name="${matrix_name//\//_}"
log_filename="logs/${safe_matrix_name}_${method_name}_mem${mem_size}_win${win_size}_k${k}.txt"

# Display selected configuration
echo "Running experiment number: $experiment_number"
echo "Matrix ($matrix_index): $matrix_name"
echo "Method: $method_name"
echo "mem_size: $mem_size"
echo "k ($k_index): $k"
echo "win_size: $win_size"

# Run the Python script
echo "Running main.py ..."
python3 -u main.py "$matrix_name" "$method_name" "$win_size" "$k" 2>&1 | tee -a "$log_filename"
