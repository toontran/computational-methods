#!/bin/bash

window_size="1" # 1 if FD v iSVD
k=8

# List of matrices
matrices=(

    # "hyperboloid_1000_1.0"
    # "hyperboloid_1000_5.0"
    # "hyperboloid_1000_10.0"
    # "hyperboloid_1000_20.0"

    # "test_coherence_100000_1.0_low_low"
    # "test_coherence_100000_1.0_med_low"

    # "test_cve_1000_1.0_high"
    # "test_cve_1000_1.0_low"

    # "test_coherence_1000_1.0_low_low"
    # "test_coherence_1000_1.0_med_low"
    # "test_coherence_1000_1.0_high_low"

    # "test_coherence_1000_10.0_low_low"
    # "test_coherence_1000_10.0_med_low"
    # "test_coherence_1000_10.0_high_low"
    # "test_coherence_1000_1.0_low_med"
    # "test_coherence_1000_1.0_med_med"
    # "test_coherence_1000_1.0_high_med"
    # "test_coherence_1000_1.0_low_high"
    # "test_coherence_1000_1.0_med_high"
    # "test_coherence_1000_1.0_high_high"

    # "test_coherence_1000_100.0_low_low"
    # "test_coherence_1000_100.0_med_low"


    # "kernel_random_1000_1.0"
    # "hyperboloid_1000_1.0"
    # "synthetic_1000_20"

    # "kernel_stocks_10000_1.0"
    # "kernel_stocks_10000_10.0"

    # "kernel_stocks_100000_1.0"
    # "kernel_stocks_100000_10.0"
    # "kernel_stocks_100000_2.2361"
    # "kernel_stocks_100000_0.7071"
    # "kernel_stocks_100000_0.2236"

    "kernel_stocks_1000_10.0"
    "kernel_stocks_1000_2.2361"
    "kernel_stocks_1000_0.7071"
    "kernel_stocks_1000_0.2236"

    # "bad_case1_1000"
    # "bad_case2_1000"
    # "bad_case3_1000"

    #"kernel_random_20000_1.0"
    # "kernel_random_20000_5.0"
    #"kernel_random_20000_10.0"
    # "kernel_random_20000_20.0"

    # "sparsify_hyperboloid_20000_1.0_0.9"
    # "sparsify_hyperboloid_20000_1.0_0.99"
    # "sparsify_hyperboloid_20000_10.0_0.9"
    # "sparsify_hyperboloid_20000_10.0_0.99"

    # "hyperboloid_20000_1.0"
    # "hyperboloid_20000_5.0"
    # "hyperboloid_20000_10.0"
    # "hyperboloid_20000_50.0"

    # "hyperboloid_20000_100.0"

    #"kernel_swissroll_20000_1.0"
    # "kernel_swissroll_20000_5.0"
    # "kernel_swissroll_20000_10.0"
    # "kernel_swissroll_20000_50.0"

    #"kernel_torus_20000_1.0"
    # "kernel_torus_20000_5.0"
    #"kernel_torus_20000_10.0"
    # "kernel_torus_20000_50.0"

    # "kernel_gaussianmixture_10000_1.0"
    # "kernel_gaussianmixture_10000_5.0"
    # "kernel_gaussianmixture_10000_10.0"
    # "kernel_gaussianmixture_10000_50.0"

    # "kernel_gaussianmixture_1000_1.0"

    # "Boeing/msc10848" # old
    # "Boeing/bcsstk36"
    # "HB/bcsstk17"
    # "Boeing/crystm02"
    # "Simon/olafu"
    # "Pothen/bodyy4"

    # "Williams/pdb1HYS" #33k

    # "Janna/Bump_2911"

    # "Rothberg/cfd1" #70k
    # "Andrews/Andrews" #60k

    # "Janna/Queen_4147"
    # "Janna/Flan_1565"

    # "Wissgott/parabolic_fem"
    # "Botonakis/thermomech_dM"
    # "AMD/G2_circuit"

)

# List of methods
methods=(
    # "nystrom_1_isvd_-1" 
    # "nystrom_1_isvd_-2" 
    # "nystrom_1_isvd_-3" 
    # "nystrom_1_isvd_-4" 
    # "nystrom_1_isvd_-5" 
    # "nystrom_1_isvd" 
    # "nystrom_3_isvd"
    # "nystrom_5_isvd"
    # "nystrom_1_isvd_100"
    # "nystrom_1_isvd_500"
    # "isvd_1_nystrom" 
    # "isvd_3_nystrom"
    # "isvd_5_nystrom"
    # "nystrom"
    "isvd"
    # "isvdls"
    # "isvdls2"
    # "nystrom_1_isvdnew"
    # "nystrom_3_isvdnew"
    # "isvd1by1"
    # "isvd1by1new"
    #"isvddemix"
    # "isvddemix2"
    # "isvddemix3"
    # "isvdst"
    "isvdstG" # <------ set here
    # "isvddemixst"
    # "isvddemixstG"
)

# Function to print usage
usage() {
    echo "Usage: $0 <experiment_number>"
    echo "Experiment number should be between 0 and $((${#matrices[@]} * ${#methods[@]} - 1))"
    echo ""
    echo "This will run the experiment with:"
    echo "  - Matrix: <experiment_number> // ${#methods[@]}"
    echo "  - Method: <experiment_number> % ${#methods[@]}"
    echo ""
    echo "Available matrices:"
    for i in "${!matrices[@]}"; do
        echo "$i: ${matrices[i]}"
    done
    echo ""
    echo "Available methods:"
    for i in "${!methods[@]}"; do
        echo "$i: ${methods[i]}"
    done
    echo ""
    echo "For example, experiment number 15 would use:"
    echo "  - Matrix: 15 // ${#methods[@]} = $(( 15 / ${#methods[@]} )): ${matrices[$(( 15 / ${#methods[@]} ))]}"
    echo "  - Method: 15 % ${#methods[@]} = $(( 15 % ${#methods[@]} )): ${methods[$(( 15 % ${#methods[@]} ))]}"
}

# Check if an argument is provided
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

# Get the experiment number from the argument
experiment_number=$1

# Calculate total number of combinations
total_combinations=$((${#matrices[@]} * ${#methods[@]}))

# Check if the experiment number is valid
if ! [[ "$experiment_number" =~ ^[0-9]+$ ]] || [ "$experiment_number" -ge "$total_combinations" ]; then
    echo "Error: Invalid experiment number."
    echo "Experiment number should be between 0 and $((total_combinations - 1))"
    usage
    exit 1
fi

# Calculate matrix index and method index
matrix_index=$((experiment_number / ${#methods[@]}))
method_index=$((experiment_number % ${#methods[@]}))

# Get the matrix and method names
matrix_name="${matrices[$matrix_index]}"
method_name="${methods[$method_index]}"

# Create the logs directory if it doesn't exist
mkdir -p logs

# Create a descriptive log filename
log_filename="logs/${matrix_name}_${method_name}.txt"

# Display the selected combination
echo "Running experiment number: $experiment_number"
echo "Matrix ($matrix_index): $matrix_name"
echo "Method ($method_index): $method_name"

# Run the Python script with both matrix and method parameters and log the output
echo "Running main.py for matrix: $matrix_name with method: $method_name"
python3 -u main.py "$matrix_name" "$method_name" "$window_size" "$k" 2>&1 | tee -a "$log_filename"
