#!/usr/bin/env python3
# --------------------------------------------------------
# Simplified Custom Organisms Model Inference Script
# Use original panoptic_inference.py to load fine-tuned model
# --------------------------------------------------------

import subprocess
import os

def run_simple_custom_inference():
    """
    Use original panoptic_inference.py script with custom model
    """
    
    # ===============================
    # Configuration Parameters
    # ===============================
    
    # Input image path - Please modify to your actual image path
    INPUT_IMAGE = os.environ.get("INPUT_IMAGE", "./sample-img/000039.jpg")  # Can be set via environment variable INPUT_IMAGE
    
    # Output results directory
    OUTPUT_DIR = "simple_custom_results"
    
    # Fine-tuned model path
    CUSTOM_MODEL_PATH = "../pretrained/model_state_dict.pt"
    
    # Use standard demo config file
    CONFIG_FILE = "configs/seem/focall_unicl_lang_demo_low_threshold.yaml" #"configs/seem/focall_unicl_lang_demo.yaml"
    
    # ===============================
    # Run Inference
    # ===============================
    
    print("Simplified Custom Organisms Model Inference")
    print("=" * 60)
    print(f"Input image: {INPUT_IMAGE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Fine-tuned model: {CUSTOM_MODEL_PATH}")
    print(f"Config file: {CONFIG_FILE}")
    print("=" * 60)
    
    # Check if files exist
    if not os.path.exists(INPUT_IMAGE):
        print(f"Error: Input image not found {INPUT_IMAGE}")
        return False
    
    if not os.path.exists(CUSTOM_MODEL_PATH):
        print(f"Error: Model file not found {CUSTOM_MODEL_PATH}")
        return False
    
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: Config file not found {CONFIG_FILE}")
        return False
    
    # Build command - use original panoptic_inference.py
    cmd = [
        "python", "panoptic_inference.py",
        "--input_image", INPUT_IMAGE,
        "--output_dir", OUTPUT_DIR,
        "--conf_files", CONFIG_FILE,
        "--model_path", CUSTOM_MODEL_PATH
    ]
    
    print("Using original panoptic_inference.py script")
    print("Although COCO class names are used, your fine-tuned weights are actually applied")
    print("Fine-tuned model should be able to recognize your trained organism categories")
    
    try:
        # Run inference script
        print("\nStarting inference...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("Inference completed!")
        print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Inference failed: {e}")
        print("Error output:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print("Error: Cannot find panoptic_inference.py script")
        return False

def main():
    """Main function"""
    print("Simplified Custom Organisms Inference")
    print("Description:")
    print("   - Use standard panoptic_inference.py script")
    print("   - Load your fine-tuned model weights")
    print("   - Avoid complex configuration matching issues")
    print("   - Output standard segmentation results")
    print("\n" + "=" * 60)
    
    success = run_simple_custom_inference()
    
    if success:
        print("\nInference completed!")
        print("Please check the results in the output directory:")
        print("   *_panoptic_mask.png - Pure color segmentation results")
        print("   *_individual_masks/ - Individual instance masks")
        print("   *_panoptic_info.txt - Detailed segmentation information")
        print("\nNote:")
        print("   - Class names are displayed as COCO categories")
        print("   - But actual detection uses your fine-tuned weights")
        print("   - Detection results should target your trained organisms")
    else:
        print("\nInference failed")
        print("Suggestions:")
        print("   1. Check model file path")
        print("   2. Confirm image path is correct")
        print("   3. Check if GPU memory is sufficient")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 