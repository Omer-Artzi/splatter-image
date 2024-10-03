#!/bin/bash

# Function to delete matching files from both "pose" and "rgb" folders
delete_matching_files() {
    pose_folder="$1/pose"
    rgb_folder="$1/rgb"
    percent="$2"

    # Check if both pose and rgb folders exist
    if [[ ! -d "$pose_folder" || ! -d "$rgb_folder" ]]; then
        echo "Either 'pose' or 'rgb' folder is missing in $1. Skipping..."
        return
    fi

    # Find all matching file names in the pose folder (assuming pose and rgb have the same files)
    pose_files=($(find "$pose_folder" -maxdepth 1 -type f | xargs -n 1 basename))

    # Calculate the number of files to delete based on percentage
    total_files=${#pose_files[@]}
    num_to_delete=$((total_files * percent / 100))
    echo "Total files: $total_files, Files to delete: $num_to_delete"

    # Shuffle the file list and delete the first num_to_delete files from both folders
    if (( num_to_delete > 0 )); then
        files_to_delete=($(shuf -e "${pose_files[@]}" | head -n "$num_to_delete"))

        for pose_file in "${files_to_delete[@]}"; do
            rgb_file="${pose_file%.txt}.png"
            echo "Deleting $rgb_folder/$rgb_file and $pose_folder/$pose_file..."
            rm -f "$pose_folder/$pose_file"
            rm -f "$rgb_folder/$rgb_file"
        done
    fi
}

# Check if correct number of arguments are given
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 directory percentage"
    exit 1
fi

directory="$1"
percent="$2"

# Validate percentage input
if ! [[ "$percent" =~ ^[0-9]+$ ]] || (( percent < 0 || percent > 100 )); then
    echo "Percentage must be a number between 0 and 100"
    exit 1
fi

# Iterate over subdirectories
for subfolder in "$directory"/*/; do
    if [[ -d "$subfolder" ]]; then
        echo "Processing folder: $subfolder"
        delete_matching_files "$subfolder" "$percent"
    fi
done

echo "Deletion process complete."

