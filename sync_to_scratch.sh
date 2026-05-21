#!/usr/bin/env bash

set -u

# Configuration
OUTPUT_NAME="${OUTPUT_NAME:-output22}"
TARGET_END_IDX="${TARGET_END_IDX:-1000}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
STABILITY_SECONDS="${STABILITY_SECONDS:-180}"
ZIP_AFTER_SYNC="${ZIP_AFTER_SYNC:-0}"
ZIP_SAMPLE_M="${ZIP_SAMPLE_M:-50}"
ZIP_OUTPUT_DIR="${ZIP_OUTPUT_DIR:-/scratch/ttran02/${OUTPUT_NAME}_sampled_zips}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${SCRIPT_DIR}/${OUTPUT_NAME}"
DEST_ROOT="${DEST_ROOT:-/scratch/ttran02/${OUTPUT_NAME}}"
STATE_FILE="${SCRIPT_DIR}/.sync_to_scratch_${OUTPUT_NAME}.state"
LOG_FILE="${SCRIPT_DIR}/sync_to_scratch_${OUTPUT_NAME}.log"

mkdir -p "$(dirname "$STATE_FILE")"
touch "$STATE_FILE"

declare -A PENDING_SINCE=()
declare -A PENDING_SIZE=()

log() {
    local message="$1"
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$message" | tee -a "$LOG_FILE"
}

is_already_synced() {
    local run_dir="$1"
    local run_name

    run_name="$(basename "$run_dir")"
    grep -Fqx "$run_dir" "$STATE_FILE" && is_run_complete "$DEST_ROOT/$run_name"
}

mark_synced() {
    local run_dir="$1"
    printf '%s\n' "$run_dir" >> "$STATE_FILE"
}

get_latest_window_info() {
    local run_dir="$1"

    find "$run_dir" -maxdepth 1 -type f \
        \( -name 'window_info.txt' -o -name 'window_info_*.txt' \) \
        -printf '%f\n' 2>/dev/null | sort -V | tail -n 1
}

get_end_idx() {
    local window_info_path="$1"

    sed -n 's/.*"end_idx":[[:space:]]*{"type":[[:space:]]*"scalar",[[:space:]]*"value":[[:space:]]*\([0-9][0-9]*\)}.*/\1/p' "$window_info_path"
}

get_dir_size_bytes() {
    local dir_path="$1"

    du -sb "$dir_path" 2>/dev/null | awk '{print $1}'
}

is_run_complete() {
    local run_dir="$1"

    [[ -f "$run_dir/other_info.txt" ]]
}

create_sampled_zip() {
    local folder="$1"
    local sample_m="$2"
    local zip_output_dir="$3"
    local folder_base
    local zipfile
    local tmpdir
    local all_index_pairs
    local families_file
    local standalone_file
    local common_indices_file
    local selected_files_file
    local cleanup
    local num_families
    local total_common
    local effective_m
    local file_count

    if ! [[ "$sample_m" =~ ^[0-9]+$ ]] || (( sample_m <= 0 )); then
        log "zip failed for $(basename "$folder"): ZIP_SAMPLE_M must be a positive integer"
        return 1
    fi

    folder_base="$(basename "$folder")"
    zipfile="${zip_output_dir}/${folder_base}_sampled.zip"

    mkdir -p "$zip_output_dir"
    if [[ -f "$zipfile" ]]; then
        log "zip skipped for $folder_base: already exists at $zipfile"
        return 0
    fi

    tmpdir="$(mktemp -d)"
    all_index_pairs="$tmpdir/all_index_pairs.txt"
    families_file="$tmpdir/families.txt"
    standalone_file="$tmpdir/standalone.txt"
    common_indices_file="$tmpdir/common_indices.txt"
    selected_files_file="$tmpdir/selected_files.txt"
    cleanup() {
        local dir="${tmpdir:-}"
        if [[ -n "$dir" ]]; then
            rm -rf -- "$dir"
        fi
    }
    trap cleanup RETURN

    : > "$all_index_pairs"
    : > "$families_file"
    : > "$standalone_file"
    : > "$selected_files_file"

    while IFS= read -r fname; do
        if [[ "$fname" =~ ^(.+)_([0-9]+)\.([^.]+)$ ]]; then
            family="${BASH_REMATCH[1]}"
            idx="${BASH_REMATCH[2]}"
            ext="${BASH_REMATCH[3]}"
            printf '%s\t%s\t%s\t%s\n' "$family" "$idx" "$ext" "$fname" >> "$all_index_pairs"
        else
            printf '%s/%s\n' "$folder" "$fname" >> "$standalone_file"
        fi
    done < <(find "$folder" -maxdepth 1 -type f -printf '%f\n')

    if [[ ! -s "$all_index_pairs" ]]; then
        if [[ -s "$standalone_file" ]]; then
            sort -u "$standalone_file" > "$selected_files_file"
            rm -f "$zipfile"
            zip -j "$zipfile" -@ < "$selected_files_file" >/dev/null
            log "zip created for $folder_base: standalone files only -> $zipfile"
            return 0
        fi

        log "zip skipped for $folder_base: no files found"
        return 0
    fi

    cut -f1 "$all_index_pairs" | sort -u > "$families_file"
    num_families="$(wc -l < "$families_file" | tr -d ' ')"

    cut -f1,2 "$all_index_pairs" | sort -u \
        | awk -F'\t' -v nf="$num_families" '
            { count[$2]++ }
            END {
                for (idx in count) {
                    if (count[idx] == nf) print idx
                }
            }
        ' | sort -n > "$common_indices_file"

    total_common="$(wc -l < "$common_indices_file" | tr -d ' ')"
    if (( total_common == 0 )); then
        log "zip skipped for $folder_base: no common indices across all families"
        return 0
    fi

    effective_m="$sample_m"
    if (( effective_m > total_common )); then
        effective_m="$total_common"
    fi

    mapfile -t common_indices < "$common_indices_file"
    if (( effective_m == 1 )); then
        selected_indices=("${common_indices[0]}")
    else
        mapfile -t selected_indices < <(
            for ((j = 0; j < effective_m; j++)); do
                pos=$(( (j * (total_common - 1)) / (effective_m - 1) ))
                echo "${common_indices[$pos]}"
            done | awk '!seen[$0]++'
        )
    fi

    awk -F'\t' '
        NR==FNR { wanted[$1]=1; next }
        ($2 in wanted) { print $4 }
    ' <(printf '%s\n' "${selected_indices[@]}") "$all_index_pairs" \
        | sort -u \
        | while IFS= read -r fname; do
            printf '%s/%s\n' "$folder" "$fname"
        done >> "$selected_files_file"

    if [[ -s "$standalone_file" ]]; then
        cat "$standalone_file" >> "$selected_files_file"
    fi

    sort -u "$selected_files_file" -o "$selected_files_file"
    file_count="$(wc -l < "$selected_files_file" | tr -d ' ')"

    rm -f "$zipfile"
    zip -j "$zipfile" -@ < "$selected_files_file" >/dev/null
    log "zip created for $folder_base: ${file_count} files -> $zipfile"
}

sync_run_dir() {
    local run_dir="$1"
    local run_name
    run_name="$(basename "$run_dir")"

    mkdir -p "$DEST_ROOT"
    if rsync -a "$run_dir/" "$DEST_ROOT/$run_name/"; then
        if [[ "$ZIP_AFTER_SYNC" == "1" ]]; then
            if ! create_sampled_zip "$DEST_ROOT/$run_name" "$ZIP_SAMPLE_M" "$ZIP_OUTPUT_DIR"; then
                log "zip failed for $run_name after sync; proceeding to remove source because scratch sync succeeded"
            fi
        fi

        if rm -rf "$run_dir"; then
            mark_synced "$run_dir"
            if [[ "$ZIP_AFTER_SYNC" == "1" ]]; then
                log "synced $run_name to $DEST_ROOT/$run_name, zipped to $ZIP_OUTPUT_DIR, and removed $run_dir"
            else
                log "synced $run_name to $DEST_ROOT/$run_name and removed $run_dir"
            fi
            return 0
        fi

        if [[ "$ZIP_AFTER_SYNC" == "1" ]]; then
            log "synced $run_name to $DEST_ROOT/$run_name and zipped it, but failed to remove $run_dir"
        else
            log "synced $run_name to $DEST_ROOT/$run_name but failed to remove $run_dir"
        fi
        return 1
    fi

    log "rsync failed for $run_name"
    return 1
}

check_run_dir() {
    local run_dir="$1"
    local run_name
    local latest_window_info_name
    local latest_window_info_path
    local end_idx
    local current_time
    local current_size
    local pending_since
    local pending_size
    local elapsed

    run_name="$(basename "$run_dir")"

    if is_already_synced "$run_dir"; then
        unset 'PENDING_SINCE[$run_dir]' 'PENDING_SIZE[$run_dir]'
        return 0
    fi

    latest_window_info_name="$(get_latest_window_info "$run_dir")"
    if [[ -z "$latest_window_info_name" ]]; then
        unset 'PENDING_SINCE[$run_dir]' 'PENDING_SIZE[$run_dir]'
        return 0
    fi

    latest_window_info_path="$run_dir/$latest_window_info_name"
    end_idx="$(get_end_idx "$latest_window_info_path")"
    if [[ -z "$end_idx" ]]; then
        log "skipping $run_name: could not parse end_idx from $latest_window_info_name"
        unset 'PENDING_SINCE[$run_dir]' 'PENDING_SIZE[$run_dir]'
        return 0
    fi

    if [[ "$end_idx" != "$TARGET_END_IDX" ]]; then
        unset 'PENDING_SINCE[$run_dir]' 'PENDING_SIZE[$run_dir]'
        return 0
    fi

    if ! is_run_complete "$run_dir"; then
        unset 'PENDING_SINCE[$run_dir]' 'PENDING_SIZE[$run_dir]'
        return 0
    fi

    current_size="$(get_dir_size_bytes "$run_dir")"
    if [[ -z "$current_size" ]]; then
        log "skipping $run_name: could not measure directory size"
        unset 'PENDING_SINCE[$run_dir]' 'PENDING_SIZE[$run_dir]'
        return 0
    fi

    current_time="$(date +%s)"
    pending_since="${PENDING_SINCE[$run_dir]-}"
    pending_size="${PENDING_SIZE[$run_dir]-}"

    if [[ -z "$pending_since" || -z "$pending_size" ]]; then
        PENDING_SINCE["$run_dir"]="$current_time"
        PENDING_SIZE["$run_dir"]="$current_size"
        log "candidate $run_name matched end_idx=$TARGET_END_IDX; monitoring for ${STABILITY_SECONDS}s of stability"
        return 0
    fi

    if [[ "$current_size" != "$pending_size" ]]; then
        PENDING_SINCE["$run_dir"]="$current_time"
        PENDING_SIZE["$run_dir"]="$current_size"
        log "candidate $run_name size changed from $pending_size to $current_size; resetting stability timer"
        return 0
    fi

    elapsed=$((current_time - pending_since))
    if (( elapsed < STABILITY_SECONDS )); then
        return 0
    fi

    if sync_run_dir "$run_dir"; then
        unset 'PENDING_SINCE[$run_dir]' 'PENDING_SIZE[$run_dir]'
    fi
}

main() {
    log "watching $SOURCE_ROOT and syncing completed runs to $DEST_ROOT when end_idx=$TARGET_END_IDX"

    while true; do
        if [[ ! -d "$SOURCE_ROOT" ]]; then
            log "source directory $SOURCE_ROOT does not exist; retrying in ${CHECK_INTERVAL_SECONDS}s"
            sleep "$CHECK_INTERVAL_SECONDS"
            continue
        fi

        while IFS= read -r run_dir; do
            check_run_dir "$run_dir"
        done < <(find "$SOURCE_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)

        sleep "$CHECK_INTERVAL_SECONDS"
    done
}

main "$@"
