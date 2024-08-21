#!/bin/bash

# List of matrices
matrices=(
    "FIDAP/ex10"
    "Boeing/msc10848"
    "Boeing/bcsstk36"
    "HB/bcsstk17"
    "Boeing/crystm02"
    "Simon/olafu"
    "Pothen/bodyy4"
    "Boeing/crystm03" # new symmetric
    "Oberwolfach/t3dl_e"
    "Lourakis/bundle1"
    "JGD_Trefethen/Trefethen_20000b"
    "Oberwolfach/gyro"
    "Schulthess/N_reactome" # new arbitrary
    "JGD_Homology/mk12-b3"
    "Kemelmacher/Kemelmacher"
    "Toledo/deltaX"
    "Meszaros/ge"
    "HB/ash219" # test arbitrary
)

# Function to print usage
usage() {
    echo "Usage: $0 <matrix_index>"
    echo "Matrix index should be between 1 and ${#matrices[@]}"
    echo "Available matrices:"
    for i in "${!matrices[@]}"; do
        echo "$((i+1)): ${matrices[i]}"
    done
}

# Check if an argument is provided
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

# Get the matrix index from the argument
index=$1

# Check if the index is valid
if [ $index -lt 1 ] || [ $index -gt ${#matrices[@]} ]; then
    echo "Error: Invalid matrix index."
    usage
    exit 1
fi

# Get the full matrix name from the list
matrix_full_name="${matrices[$((index-1))]}"

# Extract the matrix name (everything after the last '/')
matrix_name=$(basename "$matrix_full_name")

# Create the logs directory if it doesn't exist
mkdir -p logs

# Run the Python script and log the output
echo "Running isvd_orders.py for matrix: $matrix_full_name"
python3 -u isvd_orders.py "$matrix_full_name" 2>&1 | tee -a "logs/${matrix_name}.txt"