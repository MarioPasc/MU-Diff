#!/bin/bash

# Convenience script for volume prediction using experiment configurations
# Usage: ./predict_volume.sh experiment_name input_dir output_dir

set -e

if [ $# -ne 3 ]; then
    echo "Usage: $0 <experiment_name> <input_dir> <output_dir>"
    echo ""
    echo "Arguments:"
    echo "  experiment_name: Name of the experiment (e.g., synthesize_T1CE)"
    echo "  input_dir:      Directory containing input NIfTI files"
    echo "  output_dir:     Directory to save predicted volume"
    echo ""
    echo "Expected input files in input_dir:"
    echo "  - t1ce.nii.gz (or t1c.nii.gz)"
    echo "  - t1.nii.gz (or t1n.nii.gz)"
    echo "  - t2.nii.gz (or t2w.nii.gz)"
    echo "  - flair.nii.gz (or t2f.nii.gz)"
    exit 1
fi

EXPERIMENT_NAME=$1
INPUT_DIR=$2
OUTPUT_DIR=$3

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$(dirname "$SCRIPT_DIR")/engine"
EXPERIMENTS_DIR="$(dirname "$SCRIPT_DIR")/experiments"

# Function to find input file with multiple possible naming conventions
find_input_file() {
    local input_dir=$1
    local patterns=("${@:2}")
    
    for pattern in "${patterns[@]}"; do
        for ext in ".nii.gz" ".nii"; do
            file="${input_dir}/${pattern}${ext}"
            if [ -f "$file" ]; then
                echo "$file"
                return 0
            fi
        done
    done
    
    echo ""
    return 1
}

echo "MU-Diff Volume Prediction"
echo "========================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Find input files with various naming conventions
echo "Searching for input files..."

T1CE_FILE=$(find_input_file "$INPUT_DIR" "t1ce" "t1c" "T1CE" "T1C")
T1_FILE=$(find_input_file "$INPUT_DIR" "t1" "t1n" "T1" "T1N")
T2_FILE=$(find_input_file "$INPUT_DIR" "t2" "t2w" "T2" "T2W")
FLAIR_FILE=$(find_input_file "$INPUT_DIR" "flair" "t2f" "FLAIR" "T2F")

# Report found files
echo "Found input files:"
[ -n "$T1CE_FILE" ] && echo "  T1CE: $T1CE_FILE" || echo "  T1CE: NOT FOUND"
[ -n "$T1_FILE" ] && echo "  T1: $T1_FILE" || echo "  T1: NOT FOUND"
[ -n "$T2_FILE" ] && echo "  T2: $T2_FILE" || echo "  T2: NOT FOUND"
[ -n "$FLAIR_FILE" ] && echo "  FLAIR: $FLAIR_FILE" || echo "  FLAIR: NOT FOUND"
echo ""

# Extract target modality from experiment name
TARGET_MODALITY=""
if [[ "$EXPERIMENT_NAME" == *"T1CE"* ]]; then
    TARGET_MODALITY="T1CE"
elif [[ "$EXPERIMENT_NAME" == *"FLAIR"* ]]; then
    TARGET_MODALITY="FLAIR"
elif [[ "$EXPERIMENT_NAME" == *"T2"* ]]; then
    TARGET_MODALITY="T2"
elif [[ "$EXPERIMENT_NAME" == *"T1"* ]]; then
    TARGET_MODALITY="T1"
else
    echo "Error: Cannot determine target modality from experiment name: $EXPERIMENT_NAME"
    echo "Expected experiment names: synthesize_T1CE, synthesize_FLAIR, synthesize_T2, synthesize_T1"
    exit 1
fi

echo "Target modality: $TARGET_MODALITY"

# Check required input files based on target modality
REQUIRED_FILES=()
case "$TARGET_MODALITY" in
    "T1CE")
        REQUIRED_FILES=("$FLAIR_FILE" "$T2_FILE" "$T1_FILE")
        REQUIRED_NAMES=("FLAIR" "T2" "T1")
        ;;
    "FLAIR")
        REQUIRED_FILES=("$T1CE_FILE" "$T1_FILE" "$T2_FILE")
        REQUIRED_NAMES=("T1CE" "T1" "T2")
        ;;
    "T2")
        REQUIRED_FILES=("$T1CE_FILE" "$T1_FILE" "$FLAIR_FILE")
        REQUIRED_NAMES=("T1CE" "T1" "FLAIR")
        ;;
    "T1")
        REQUIRED_FILES=("$FLAIR_FILE" "$T1CE_FILE" "$T2_FILE")
        REQUIRED_NAMES=("FLAIR" "T1CE" "T2")
        ;;
esac

# Validate required files exist
MISSING_FILES=()
for i in "${!REQUIRED_FILES[@]}"; do
    if [ -z "${REQUIRED_FILES[$i]}" ]; then
        MISSING_FILES+=("${REQUIRED_NAMES[$i]}")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo "Error: Missing required input files for target $TARGET_MODALITY:"
    printf '  %s\n' "${MISSING_FILES[@]}"
    echo ""
    echo "Required input modalities for $TARGET_MODALITY: ${REQUIRED_NAMES[*]}"
    exit 1
fi

echo "All required input files found!"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command arguments
CMD_ARGS=(
    "--target_modality" "$TARGET_MODALITY"
    "--output_dir" "$OUTPUT_DIR"
    "--exp" "$EXPERIMENT_NAME"
    "--output_path" "./results"
)

# Add input file arguments
[ -n "$T1CE_FILE" ] && CMD_ARGS+=("--input_t1ce" "$T1CE_FILE")
[ -n "$T1_FILE" ] && CMD_ARGS+=("--input_t1" "$T1_FILE")
[ -n "$T2_FILE" ] && CMD_ARGS+=("--input_t2" "$T2_FILE")
[ -n "$FLAIR_FILE" ] && CMD_ARGS+=("--input_flair" "$FLAIR_FILE")

# Run prediction
echo "Running volume prediction..."
echo "Command: python $ENGINE_DIR/test_volume.py ${CMD_ARGS[*]}"
echo ""

cd "$ENGINE_DIR"
python test_volume.py "${CMD_ARGS[@]}"

echo ""
echo "Prediction completed successfully!"
echo "Output saved to: $OUTPUT_DIR/predicted_${TARGET_MODALITY,,}.nii.gz"
