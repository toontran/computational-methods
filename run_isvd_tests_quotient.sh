#!/bin/bash

# List of matrices
matrices=(
#     "bad_matrix_10"
#    "hyperboloid_10000_2.5"
#    "hyperboloid_10000_1.5"
#    "hyperboloid_10000_0.5"
#    "hyperboloid_10000_-0.5"
#    "hyperboloid_10000_-0.25"
#    "hyperboloid_10000_-0.75"
#    "hyperboloid_10000_-1.5"
#    "hyperboloid_10000_-2.5"
#     "hyperboloid_10000_-5.0"
#     "hyperboloid_10000_-4.0"
#     "hyperboloid_10000_-3.0"
#     "hyperboloid_10000_-2.0"
#     "hyperboloid_10000_-1.0" 
    "hyperboloid_10000_0.0" 
    "hyperboloid_10000_1.0"
    "hyperboloid_10000_2.0" 
#     "hyperboloid_10000_3.0"
#     "hyperboloid_10000_4.0"
#     "hyperboloid_10000_5.0"
#     "kronecker_graph_13_0.3" 
#     "kronecker_graph_13_0.6"
#     "kronecker_graph_13_0.9" 
#     "hyperboloid_100000_0.1" 
#     "kronecker_graph_16_0.3"
#     "Schmid/thermal2" # 1M2
#     "AMD/G3_circuit" #1M5
#     "Janna/Serena" #1M3
#     "Janna/StocF-1465" #1M4
#     "ND/nd6k" # new
#     "G2_circuit"
#     "UTEP/Dubcova1"
#     "JGD_Trefethen/Trefethen_20000"
#     "ACUSIM/Pres_Poisson"
#     "TKK/smt"
#     "FIDAP/ex10" # SPD
#     "Boeing/crystm03" # new symmetric
#     "Oberwolfach/t3dl_e"
#     "Lourakis/bundle1"
#     "JGD_Trefethen/Trefethen_20000b" # Too high condition number
#     "Oberwolfach/gyro"
    "Boeing/msc10848" # old
    "Boeing/bcsstk36"
    "HB/bcsstk17"
    "Boeing/crystm02"
    "Simon/olafu"
    "Pothen/bodyy4"
    "Mallya/lhr11" # Asymmetric
    "Grund/bayer10" #
    "FEMLAB/poisson3Da" #
    # "SNAP/cit-HepTh" # 10^66 condition number !!
    "Gaertner/big"
    "Hollinger/g7jac040" #
    # "Schulthess/N_reactome" # new arbitrary
    # "JGD_Homology/mk12-b3"
    # "Kemelmacher/Kemelmacher"
    # "Toledo/deltaX"
    # "Meszaros/ge"
    # "HB/ash219" # test arbitrary
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
python3 -u isvd_orders_quotient.py "$matrix_full_name" 2>&1 | tee -a "logs/${matrix_name}.txt"
