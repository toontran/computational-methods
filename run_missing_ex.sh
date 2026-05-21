method_name="isvd"

# 5
matrix_name="HB/1138_bus"
win_size=8
k=16
safe_matrix_name="${matrix_name//\//_}"
log_filename="logs/${safe_matrix_name}_${method_name}_mem${mem_size}_win${win_size}_k${k}.txt"
echo python3 -u main.py "$matrix_name" "$method_name" "$win_size" "$k" 2>&1 \| tee -a "$log_filename"
python3 -u main.py "$matrix_name" "$method_name" "$win_size" "$k" 2>&1 | tee -a "$log_filename"
