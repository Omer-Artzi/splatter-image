#!/bin/bash

# Check if two arguments (directory and percentage) are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <directory> <percentage>"
    exit 1
fi

# Assign variables for directory and percentage
DIR="$1"
PERCENTAGE="$2"

# Check if the provided percentage is a valid number between 1 and 100
if ! [[ "$PERCENTAGE" =~ ^[0-9]+$ ]] || [ "$PERCENTAGE" -lt 1 ] || [ "$PERCENTAGE" -gt 100 ]; then
    echo "Please provide a valid percentage between 1 and 100."
    exit 1
fi

# Get the list of subdirectories (only directories, not files)
# shellcheck disable=SC2207
SUBDIRS=( $(find "$DIR" -mindepth 1 -maxdepth 1 -type d) )

# Calculate how many subdirectories should be deleted
TOTAL_SUBDIRS=${#SUBDIRS[@]}
DELETE_COUNT=$(( TOTAL_SUBDIRS * PERCENTAGE / 100 ))

# Shuffle the list of subdirectories and pick the ones to delete
# shellcheck disable=SC2207
DELETE_SUBDIRS=( $(shuf -e "${SUBDIRS[@]}" -n "$DELETE_COUNT") )

# Delete the selected subdirectories
for subdir in "${DELETE_SUBDIRS[@]}"; do
    echo "Deleting: $subdir"
    rm -rf "$subdir"
done

echo "$DELETE_COUNT subdirectories deleted out of $TOTAL_SUBDIRS."