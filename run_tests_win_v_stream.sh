#!/bin/bash

# Fixed experiment settings
matrix_name="kernel_stocks_5000_0.7071"
method_name="isvd"
win_size=1000

# Manually specified k values
k_values=(1 2 4 50 100 200 500 999)

usage() {#!/bin/bash

# Fixed experiment settings
matrix_name="kernel_stocks_5000_0.7071"
method_name="isvd"
mem_size=1000

# Manually specified k values
k_values=(
    1 
    2
    4
    8
    50
    100
    200
    500
    999
)

usage() {
    echo "Usage: $0 <experiment_number>"
    echo "Experiment number should be between 0 and $((${#k_values[@]} - 1))"
    echo ""
    echo "Fixed settings:"
    echo "  Matrix:   $matrix_name"
    echo "  Method:   $method_name"
    echo "  mem_size: $mem_size"
    echo "  win_size: mem_size - k"
    echo ""
    echo "Available k values:"
    for i in "${!k_values[@]}"; do
        echo "  $i: ${k_values[i]}    -> win_size=$((mem_size - ${k_values[i]}))"
    done
}

# Check argument
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

experiment_number=$1
total_combinations=${#k_values[@]}

# Validate experiment number
if ! [[ "$experiment_number" =~ ^[0-9]+$ ]] || [ "$experiment_number" -ge "$total_combinations" ]; then
    echo "Error: Invalid experiment number."
    echo "Experiment number should be between 0 and $((total_combinations - 1))"
    usage
    exit 1
fi

# Select k and derive win_size
k="${k_values[$experiment_number]}"
win_size=$((mem_size - k))
# win_size=1

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
echo "Matrix: $matrix_name"
echo "Method: $method_name"
echo "mem_size: $mem_size"
echo "win_size: $win_size"
echo "k: $k"

# Run the Python script
echo "Running main.py ..."
python3 -u main.py "$matrix_name" "$method_name" "$win_size" "$k" 2>&1 | tee -a "$log_filename"
    echo "Usage: $0 <experiment_number>"
    echo "Experiment number should be between 0 and $((${#k_values[@]} - 1))"
    echo ""
    echo "Fixed settings:"
    echo "  Matrix:   $matrix_name"
    echo "  Method:   $method_name"
    echo "  mem_size: $mem_size"
    echo ""
    echo "Available k values:"
    for i in "${!k_values[@]}"; do
        echo "  $i: ${k_values[i]}"
    done
}

# Check argument
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

experiment_number=$1
total_combinations=${#k_values[@]}

# Validate experiment number
if ! [[ "$experiment_number" =~ ^[0-9]+$ ]] || [ "$experiment_number" -ge "$total_combinations" ]; then
    echo "Error: Invalid experiment number."
    echo "Experiment number should be between 0 and $((total_combinations - 1))"
    usage
    exit 1
fi

# Select k
k="${k_values[$experiment_number]}"

# Create logs directory
mkdir -p logs

# Safe log filename
safe_matrix_name="${matrix_name//\//_}"
log_filename="logs/${safe_matrix_name}_${method_name}_mem${mem_size}_k${k}.txt"

# Display selected configuration
echo "Running experiment number: $experiment_number"
echo "Matrix: $matrix_name"
echo "Method: $method_name"
echo "win_size: $win_size"
echo "k: $k"

# Run the Python script
echo "Running main.py ..."
python3 -u main.py "$matrix_name" "$method_name" "$win_size" "$k" 2>&1 | tee -a "$log_filename"