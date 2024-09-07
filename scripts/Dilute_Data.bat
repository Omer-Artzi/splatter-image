#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 directory_path percentage"
    exit 1
}

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    usage
fi

# Get the directory path and percentage from arguments
dirPath="$1"
percent="$2"

# Check if the directory exists
if [ ! -d "$dirPath" ]; then
    echo "Directory not found."
    exit 1
fi

# Validate the percentage
if [ "$percent" -lt 0 ] || [ "$percent" -gt 100 ]; then
    echo "Invalid percentage. Must be between 0 and 100."
    exit 1
fi

# Get a list of files in the directory (excluding directories)
files=($(find "$dirPath" -maxdepth 1 -type f))

# Calculate the number of files to delete
numFiles=${#files[@]}
numToDelete=$(echo "($percent / 100) * $numFiles" | bc | awk '{print int($1 + 0.5)}')

# Check if there are files to delete
if [ "$numFiles" -eq 0 ]; then
    echo "No files found in the directory."
    exit 1
fi

# Randomly select files to delete
filesToDelete=($(shuf -e "${files[@]}" -n "$numToDelete"))

# Delete the selected files
for file in "${filesToDelete[@]}"; do
    rm -rf "$file"
    echo "Deleted: $file"
done

echo "Random deletion of files complete."
