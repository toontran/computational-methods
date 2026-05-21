#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <root_folder> <M> <zip_output_dir>" >&2
    echo "Example: $0 output 50 sampled_zips" >&2
    exit 1
fi

root_folder="$1"
M="$2"
zip_output_dir="$3"

if [ ! -d "$root_folder" ]; then
    echo "Error: root folder does not exist: $root_folder" >&2
    exit 1
fi

if ! [[ "$M" =~ ^[0-9]+$ ]] || [ "$M" -le 0 ]; then
    echo "Error: M must be a positive integer" >&2
    exit 1
fi

mkdir -p "$zip_output_dir"

process_one_folder() {
    local folder="$1"
    local M="$2"
    local zip_output_dir="$3"

    local folder_base
    folder_base=$(basename "$folder")
    local zipfile="$zip_output_dir/${folder_base}_sampled.zip"

    if [ -f "$zipfile" ]; then
        echo "[${folder_base}] skipped: zip already exists at ${zipfile}"
        return
    fi

    local tmpdir
    tmpdir=$(mktemp -d)

    local all_index_pairs="$tmpdir/all_index_pairs.txt"
    local families_file="$tmpdir/families.txt"
    local standalone_file="$tmpdir/standalone.txt"
    local common_indices_file="$tmpdir/common_indices.txt"
    local selected_files_file="$tmpdir/selected_files.txt"

    trap 'rm -rf "$tmpdir"' RETURN

    : > "$all_index_pairs"
    : > "$families_file"
    : > "$standalone_file"
    : > "$selected_files_file"

    find "$folder" -maxdepth 1 -type f -printf '%f\n' | while read -r fname; do
        if [[ "$fname" =~ ^(.+)_([0-9]+)\.([^.]+)$ ]]; then
            family="${BASH_REMATCH[1]}"
            idx="${BASH_REMATCH[2]}"
            ext="${BASH_REMATCH[3]}"
            printf '%s\t%s\t%s\t%s\n' "$family" "$idx" "$ext" "$fname"
        else
            printf '%s/%s\n' "$folder" "$fname" >> "$standalone_file"
        fi
    done > "$all_index_pairs"

    if [ ! -s "$all_index_pairs" ]; then
        if [ -s "$standalone_file" ]; then
            sort -u "$standalone_file" > "$selected_files_file"
            echo "[${folder_base}] no indexed families, zipping standalone files only"
            rm -f "$zipfile"
            zip -j "$zipfile" -@ < "$selected_files_file" >/dev/null
        else
            echo "[${folder_base}] skipped: no files found"
        fi
        return
    fi

    cut -f1 "$all_index_pairs" | sort -u > "$families_file"
    local num_families
    num_families=$(wc -l < "$families_file" | tr -d ' ')

    cut -f1,2 "$all_index_pairs" | sort -u \
    | awk -F'\t' -v nf="$num_families" '
        { count[$2]++ }
        END {
            for (idx in count) {
                if (count[idx] == nf) print idx
            }
        }
    ' | sort -n > "$common_indices_file"

    local N
    N=$(wc -l < "$common_indices_file" | tr -d ' ')

    if [ "$N" -eq 0 ]; then
        echo "[${folder_base}] skipped: no common indices across all families"
        return
    fi

    if [ "$M" -gt "$N" ]; then
        echo "[${folder_base}] warning: M=$M > N=$N, using M=$N"
        M="$N"
    fi

    mapfile -t common_indices < "$common_indices_file"

    if [ "$M" -eq 1 ]; then
        selected_indices=("${common_indices[0]}")
    else
        mapfile -t selected_indices < <(
            for ((j=0; j<M; j++)); do
                pos=$(( (j * (N - 1)) / (M - 1) ))
                echo "${common_indices[$pos]}"
            done | awk '!seen[$0]++'
        )
    fi

    awk -F'\t' '
        NR==FNR { wanted[$1]=1; next }
        ($2 in wanted) { print $4 }
    ' <(printf '%s\n' "${selected_indices[@]}") "$all_index_pairs" \
    | sort -u \
    | while read -r fname; do
        printf '%s/%s\n' "$folder" "$fname"
    done >> "$selected_files_file"

    if [ -s "$standalone_file" ]; then
        cat "$standalone_file" >> "$selected_files_file"
    fi

    sort -u "$selected_files_file" -o "$selected_files_file"

    local file_count
    file_count=$(wc -l < "$selected_files_file" | tr -d ' ')

    echo "[${folder_base}] zipping ${file_count} files -> ${zipfile}"
    rm -f "$zipfile"
    zip -j "$zipfile" -@ < "$selected_files_file" >/dev/null
}

export -f process_one_folder

find "$root_folder" -mindepth 1 -maxdepth 1 -type d | sort | while read -r subfolder; do
    process_one_folder "$subfolder" "$M" "$zip_output_dir"
done

echo "Done. Zip files are in: $zip_output_dir"
