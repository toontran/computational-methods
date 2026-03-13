#!/bin/bash

# Fixed experiment settings
method_name="isvd"

# List of matrices
matrices=(
    # "kernel_stocks_1000_0.7071"
    # "kernel_stocks_1000_1.0"
    # "kernel_stocks_1000_2.2361"
    # "bad_case1_1000"
    # "bad_case2_1000"
    # "bad_case3_1000"
    "FIDAP/ex3"
    "HB/plat1919"
    "HB/1138_bus"
)

get_mem_size() {
    local matrix_name="$1"
    case "$matrix_name" in
        bad_case1_1000|bad_case2_1000|bad_case3_1000)
            echo 129
            ;;
        *)
            echo 110
            ;;
    esac
}

get_k_values() {
    local matrix_name="$1"
    case "$matrix_name" in
        bad_case1_1000|bad_case2_1000|bad_case3_1000)
            echo "2 4 8 32 64 128"
            ;;
        *)
            echo "1 4 8 32 64 102 106 109"
            ;;
    esac
}

count_total_combinations() {
    local total=0
    local matrix_name
    local k_values_local
    local k_array

    for matrix_name in "${matrices[@]}"; do
        k_values_local="$(get_k_values "$matrix_name")"
        read -r -a k_array <<< "$k_values_local"
        total=$((total + ${#k_array[@]}))
    done

    echo "$total"
}

usage() {
    local total_combinations
    local running_start
    local running_end
    local matrix_name
    local mem_size_local
    local k_values_local
    local k_array
    local i

    total_combinations=$(count_total_combinations)

    echo "Usage: $0 <experiment_number>"
    echo "Experiment number should be between 0 and $((total_combinations - 1))"
    echo ""
    echo "Fixed settings:"
    echo "  Method: $method_name"
    echo "  win_size: mem_size - k"
    echo ""
    echo "Experiment numbering is assigned sequentially across matrices."
    echo ""

    echo "Available matrices and their parameter sets:"
    running_start=0
    for matrix_name in "${matrices[@]}"; do
        mem_size_local=$(get_mem_size "$matrix_name")
        k_values_local="$(get_k_values "$matrix_name")"
        read -r -a k_array <<< "$k_values_local"
        running_end=$((running_start + ${#k_array[@]} - 1))

        echo "  $matrix_name"
        echo "    mem_size=$mem_size_local"
        echo "    experiment_numbers=$running_start..$running_end"
        echo "    k values / win_size:"
        for i in "${!k_array[@]}"; do
            echo "      $i: k=${k_array[i]} -> win_size=$((mem_size_local - k_array[i]))"
        done
        echo ""

        running_start=$((running_end + 1))
    done

    echo "Example:"
    if [ "$total_combinations" -gt 0 ]; then
        echo "  $0 0"
    fi
}

# Check argument
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

experiment_number=$1
total_combinations=$(count_total_combinations)

# Validate experiment number
if ! [[ "$experiment_number" =~ ^[0-9]+$ ]] || [ "$experiment_number" -ge "$total_combinations" ]; then
    echo "Error: Invalid experiment number."
    echo "Experiment number should be between 0 and $((total_combinations - 1))"
    usage
    exit 1
fi

# Map experiment number to matrix and k, allowing per-matrix k lists
remaining_index=$experiment_number
selected_matrix=""
selected_mem_size=""
selected_k=""

for matrix_name in "${matrices[@]}"; do
    mem_size_local=$(get_mem_size "$matrix_name")
    k_values_local="$(get_k_values "$matrix_name")"
    read -r -a k_array <<< "$k_values_local"

    if [ "$remaining_index" -lt "${#k_array[@]}" ]; then
        selected_matrix="$matrix_name"
        selected_mem_size="$mem_size_local"
        selected_k="${k_array[$remaining_index]}"
        break
    fi

    remaining_index=$((remaining_index - ${#k_array[@]}))
done

matrix_name="$selected_matrix"
mem_size="$selected_mem_size"
k="$selected_k"
win_size=$((mem_size - k))

# Safety check
if [ -z "$matrix_name" ] || [ -z "$mem_size" ] || [ -z "$k" ]; then
    echo "Error: Failed to map experiment number to a valid configuration."
    exit 1
fi

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
echo "k: $k"
echo "win_size: $win_size"

# Run the Python script
echo "Running main.py ..."
python3 -u main.py "$matrix_name" "$method_name" "$win_size" "$k" 2>&1 | tee -a "$log_filename"
