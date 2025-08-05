#!/bin/bash
# Example usage of MU-Diff volume prediction tools
# This script demonstrates different ways to predict complete 3D volumes

# Set common paths
EXPERIMENT_CONFIG="experiments/cfg/local.yaml"
INPUT_DIR="/path/to/your/input/nifti/files"
OUTPUT_DIR="/path/to/your/output/directory"
RESULTS_PATH="./results"

echo "=== MU-Diff Volume Prediction Examples ==="
echo ""

# Example 1: Using configuration-based approach (recommended)
echo "Example 1: Configuration-based prediction"
echo "python tools/predict_volume_wrapper.py -c $EXPERIMENT_CONFIG -e synthesize_T1CE -i $INPUT_DIR -o $OUTPUT_DIR/t1ce_prediction"
echo ""

# Example 2: Direct specification approach  
echo "Example 2: Direct specification"
echo "python tools/predict_volume_wrapper.py --target T1CE --exp synthesize_T1CE -i $INPUT_DIR -o $OUTPUT_DIR/t1ce_direct"
echo ""

# Example 3: Using the low-level test_volume.py script directly
echo "Example 3: Direct test_volume.py usage"
echo "python engine/test_volume.py \\"
echo "    --target_modality T1CE \\"
echo "    --input_t1ce $INPUT_DIR/t1ce.nii.gz \\"
echo "    --input_t1 $INPUT_DIR/t1.nii.gz \\"
echo "    --input_t2 $INPUT_DIR/t2.nii.gz \\"
echo "    --input_flair $INPUT_DIR/flair.nii.gz \\"
echo "    --output_dir $OUTPUT_DIR/t1ce_lowlevel \\"
echo "    --exp synthesize_T1CE \\"
echo "    --num_channels 4 \\"
echo "    --image_size 128 \\"
echo "    --num_channels_dae 128"
echo ""

# Example 4: Using the bash convenience script
echo "Example 4: Bash convenience script"
echo "tools/predict_volume.sh $INPUT_DIR $OUTPUT_DIR/t1ce_bash synthesize_T1CE T1CE"
echo ""

# Example 5: Predicting multiple modalities
echo "Example 5: Predicting all modalities"
echo "# T1CE"
echo "python tools/predict_volume_wrapper.py -c $EXPERIMENT_CONFIG -e synthesize_T1CE -i $INPUT_DIR -o $OUTPUT_DIR/all_modalities"
echo ""
echo "# FLAIR"  
echo "python tools/predict_volume_wrapper.py -c $EXPERIMENT_CONFIG -e synthesize_FLAIR -i $INPUT_DIR -o $OUTPUT_DIR/all_modalities"
echo ""
echo "# T2"
echo "python tools/predict_volume_wrapper.py -c $EXPERIMENT_CONFIG -e synthesize_T2 -i $INPUT_DIR -o $OUTPUT_DIR/all_modalities"
echo ""
echo "# T1"
echo "python tools/predict_volume_wrapper.py -c $EXPERIMENT_CONFIG -e synthesize_T1 -i $INPUT_DIR -o $OUTPUT_DIR/all_modalities"
echo ""

echo "=== Input File Requirements ==="
echo ""
echo "Your input directory should contain NIfTI files with these naming patterns:"
echo "- T1CE: t1ce.nii.gz, t1c.nii.gz, T1CE.nii.gz, or T1C.nii.gz"
echo "- T1:   t1.nii.gz, t1n.nii.gz, T1.nii.gz, or T1N.nii.gz"
echo "- T2:   t2.nii.gz, t2w.nii.gz, T2.nii.gz, or T2W.nii.gz"
echo "- FLAIR: flair.nii.gz, t2f.nii.gz, FLAIR.nii.gz, or T2F.nii.gz"
echo ""

echo "=== Expected Outputs ==="
echo ""
echo "All scripts will save:"
echo "- predicted_[modality].nii.gz: The synthesized volume"
echo "- (Optional) Individual slice PNGs for visualization"
echo ""

echo "=== Notes ==="
echo "- Replace paths with your actual data paths"
echo "- Make sure you have trained models in the results directory"
echo "- GPU memory requirements depend on volume size and model complexity"
echo "- The wrapper scripts handle preprocessing automatically"
