#!/usr/bin/env python3
# --------------------------------------------------------
# Batch Custom Organisms Model Inference Script
# Process all images in a directory
# --------------------------------------------------------

import subprocess
import os
import glob
from pathlib import Path
import time

def get_image_files(directory):
    """Get all image files in the directory"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for ext in image_extensions:
        pattern = os.path.join(directory, ext)
        image_files.extend(glob.glob(pattern))
        # Also search for uppercase extensions
        pattern = os.path.join(directory, ext.upper())
        image_files.extend(glob.glob(pattern))
    
    return sorted(list(set(image_files)))  # Remove duplicates and sort

def run_single_inference(image_path, output_dir, model_path, config_file):
    """Run inference on a single image"""
    
    # Build command
    cmd = [
        "python", "panoptic_inference.py",
        "--input_image", image_path,
        "--output_dir", output_dir,
        "--conf_files", config_file,
        "--model_path", model_path
    ]
    
    try:
        # Run inference script
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, result.stdout
        
    except subprocess.CalledProcessError as e:
        return False, f"Error: {e}\nStderr: {e.stderr}"
    except FileNotFoundError:
        return False, "Error: Cannot find panoptic_inference.py script"

def run_batch_custom_inference():
    """
    Batch process all images in the directory
    """
    
    # ===============================
    # Configuration Parameters
    # ===============================
    
    # Input image directory - Please modify to your actual input directory path
    INPUT_DIR = os.environ.get("INPUT_DIR", "./sample-first-img")  # Can be set via environment variable INPUT_DIR
    
    # Output results directory
    OUTPUT_DIR = "simple_custom_results"
    
    # Fine-tuned model path - Please modify to your actual model path
    CUSTOM_MODEL_PATH = os.environ.get("CUSTOM_MODEL_PATH", "./finetune_output/model_state_dict.pt")  # Can be set via environment variable CUSTOM_MODEL_PATH
    
    # Use standard demo config file
    CONFIG_FILE = "configs/seem/focall_unicl_lang_demo_low_threshold.yaml"
    print("Using standard config file")
    
    # ===============================
    # Scan Image Files
    # ===============================
    
    print("Batch Custom Organisms Model Inference")
    print("=" * 80)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Fine-tuned model: {CUSTOM_MODEL_PATH}")
    print(f"Config file: {CONFIG_FILE}")
    print("=" * 80)
    
    # Check input directory
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory not found: {INPUT_DIR}")
        return False
    
    # Check model file
    if not os.path.exists(CUSTOM_MODEL_PATH):
        print(f"Error: Model file not found: {CUSTOM_MODEL_PATH}")
        return False
    
    # Check config file
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: Config file not found: {CONFIG_FILE}")
        return False
    
    # Get all image files
    image_files = get_image_files(INPUT_DIR)
    
    if not image_files:
        print(f"Error: No image files found in directory: {INPUT_DIR}")
        return False
    
    print(f"Found {len(image_files)} images")
    for i, img_path in enumerate(image_files, 1):
        img_name = os.path.basename(img_path)
        print(f"   {i:2d}. {img_name}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ===============================
    # Batch Processing
    # ===============================
    
    print(f"\nStarting batch inference...")
    print("Using original panoptic_inference.py script")
    print("Although COCO class names are displayed, your fine-tuned weights are actually used")
    
    successful_count = 0
    failed_count = 0
    failed_images = []
    
    start_time = time.time()
    
    for i, image_path in enumerate(image_files, 1):
        image_name = os.path.basename(image_path)
        print(f"\nProcessing image {i}/{len(image_files)}: {image_name}")
        
        # Run single image inference
        success, output = run_single_inference(image_path, OUTPUT_DIR, CUSTOM_MODEL_PATH, CONFIG_FILE)
        
        if success:
            print(f"   Success")
            successful_count += 1
        else:
            print(f"   Failed")
            print(f"   Error message: {output}")
            failed_count += 1
            failed_images.append(image_name)
        
        # Show progress
        progress = (i / len(image_files)) * 100
        elapsed_time = time.time() - start_time
        estimated_total = elapsed_time * len(image_files) / i
        remaining_time = estimated_total - elapsed_time
        
        print(f"   Progress: {progress:.1f}% | Time used: {elapsed_time:.1f}s | Estimated remaining: {remaining_time:.1f}s")
    
    # ===============================
    # Statistics
    # ===============================
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("Batch inference completed!")
    print(f"Processing statistics:")
    print(f"   Total images: {len(image_files)}")
    print(f"   Successfully processed: {successful_count}")
    print(f"   Processing failed: {failed_count}")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Average per image: {total_time/len(image_files):.1f}s")
    
    if failed_images:
        print(f"\nFailed images:")
        for img in failed_images:
            print(f"   - {img}")
    
    print(f"\nResults saved in: {OUTPUT_DIR}")
    print("Output file types:")
    print("   *_panoptic_mask.png - Pure color segmentation results")
    print("   *_individual_masks/ - Individual instance masks")
    print("   *_panoptic_info.txt - Detailed segmentation information")
    
    return successful_count > 0

def main():
    """Main function"""
    print("Batch Custom Organisms Inference")
    print("Function description:")
    print("   - Automatically process all images in a directory")
    print("   - Use your fine-tuned model weights")
    print("   - Maintain original output format and paths")
    print("   - Display detailed processing progress and statistics")
    print("   - Automatically detect and use low threshold config if available")
    
    success = run_batch_custom_inference()
    
    if success:
        print("\nBatch processing completed successfully!")
        print("Tips:")
        print("   - Check the results of each image in the output directory")
        print("   - Focus on the detection effectiveness of whitemouse and drosophila")
        print("   - If detection results are not ideal, you can create a low threshold config file")
    else:
        print("\nBatch processing failed")
        print("Suggestions:")
        print("   1. Check model and config file paths")
        print("   2. Ensure sufficient GPU memory")
        print("   3. Check if the image directory is correct")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 