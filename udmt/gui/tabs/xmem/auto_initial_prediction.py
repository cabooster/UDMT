#!/usr/bin/env python3
# --------------------------------------------------------
# Auto Initial Prediction Module
# Use instance segmentation from initial_masks_inference to predict the first frame
# --------------------------------------------------------

import os
import sys
import numpy as np
from PIL import Image
import torch
import subprocess
import tempfile
from pathlib import Path

# Add initial masks inference module path
script_dir = os.path.dirname(os.path.abspath(__file__))
initial_masks_path = os.path.join(script_dir, '../initial_masks_inference')
sys.path.insert(0, initial_masks_path)

try:
    from panoptic_inference import setup_model, load_and_preprocess_image, run_panoptic_inference
except ImportError:
    print("Warning: Cannot import panoptic_inference module, will use fallback method")

def run_auto_initial_prediction(image_dir, mask_dir, config_file=None, model_path=None, target_size=None, start_point_save_path=None, resize_scale=1.0):
    """
    Run automatic initial prediction on the first frame of video
    
    Args:
        image_dir: Image directory path (from resource_manager.image_dir)
        mask_dir: Mask directory path (from resource_manager.mask_dir)
        config_file: Configuration file path (optional)
        model_path: Model weights path (optional)
        target_size: Target size (height, width) (optional)
        start_point_save_path: Path to save start_pos_array.txt (optional)
        resize_scale: Scale factor for coordinate conversion (like manual clicks)
    
    Returns:
        bool: Whether prediction succeeded
        int: Number of detected instances
    """
    
    print("🤖 Starting automatic initial prediction...")
    
    # Check first frame image
    first_frame_name = "0000000.jpg"
    first_frame_path = os.path.join(image_dir, first_frame_name)
    
    if not os.path.exists(first_frame_path):
        print(f"❌ Cannot find first frame image: {first_frame_path}")
        return False, 0
    
    # Get original image size (if target_size not provided)
    if target_size is None:
        from PIL import Image as PILImage
        with PILImage.open(first_frame_path) as img:
            original_width, original_height = img.size
            target_size = (original_height, original_width)  # (height, width)
            print(f"📏 Detected original image size: {original_width}x{original_height}")
    else:
        print(f"📏 Using specified target size: {target_size[1]}x{target_size[0]}")
    
    # Set default configuration
    if config_file is None:
        config_file = os.path.join(script_dir, "../initial_masks_inference/configs/seem/focall_unicl_lang_demo_low_threshold.yaml")
    
    if model_path is None:
        model_path = os.path.join(script_dir, "../../pretrained/model_state_dict.pt")
    
    try:
        # Method 1: Direct call to panoptic_inference functions (recommended)
        return _run_direct_inference(first_frame_path, mask_dir, config_file, model_path, target_size, start_point_save_path, resize_scale)
    except Exception as e:
        print(f"⚠️ Direct call failed: {e}")
        print("🔄 Trying script call method...")
        
        # Method 2: Call panoptic_inference.py script
        return _run_script_inference(first_frame_path, mask_dir, config_file, model_path, target_size, start_point_save_path, resize_scale)

def _run_direct_inference(image_path, mask_dir, config_file, model_path, target_size, start_point_save_path, resize_scale):
    """Direct call to panoptic_inference functions"""
    try:
        print("📊 Setting up model...")
        model = setup_model(config_file, model_path)
        
        print("📷 Loading and preprocessing image...")
        original_image, processed_image, image_tensor = load_and_preprocess_image(image_path)
        
        print("🔄 Running inference...")
        result_image, pano_seg_info, pano_seg_numpy = run_panoptic_inference(model, image_tensor, processed_image)
        
        # Save prediction results
        return _save_prediction_results(pano_seg_info, pano_seg_numpy, processed_image.size, mask_dir, target_size, start_point_save_path, resize_scale)
        
    except Exception as e:
        print(f"❌ Direct inference failed: {e}")
        raise

def _run_script_inference(image_path, mask_dir, config_file, model_path, target_size, start_point_save_path, resize_scale):
    """Call panoptic_inference.py script"""
    try:
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"📁 Temporary directory: {temp_dir}")
            
            # Build command
            script_dir = os.path.dirname(os.path.abspath(__file__))
            initial_masks_path = os.path.join(script_dir, "../initial_masks_inference")
            cmd = [
                "python", 
                os.path.join(initial_masks_path, "panoptic_inference.py"),
                "--input_image", image_path,
                "--output_dir", temp_dir,
                "--conf_files", config_file,
                "--model_path", model_path
            ]
            
            print(f"🚀 Executing command: {' '.join(cmd)}")
            
            # Run inference
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=initial_masks_path)
            
            print("✅ Script executed successfully!")
            if result.stdout:
                print("Output:", result.stdout)
            
            # Process results
            return _process_script_results(temp_dir, mask_dir, image_path, target_size, start_point_save_path, resize_scale)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Script execution failed: {e}")
        if e.stderr:
            print("Error output:", e.stderr)
        return False, 0
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return False, 0

def _process_script_results(temp_dir, mask_dir, image_path, target_size, start_point_save_path, resize_scale):
    """Process script output results"""
    try:
        # Find output files
        image_name = Path(image_path).stem
        masks_dir = os.path.join(temp_dir, f"{image_name}_individual_masks")
        
        if not os.path.exists(masks_dir):
            print(f"❌ Cannot find output mask directory: {masks_dir}")
            return False, 0
        
        # Read all mask files
        mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.png')]
        
        if not mask_files:
            print("⚠️ No instances detected")
            return True, 0
        
        print(f"🎯 Found {len(mask_files)} instances")
        
        # Combine all masks into a multi-instance mask
        combined_mask = None
        instance_id = 1
        
        for mask_file in sorted(mask_files):
            mask_path = os.path.join(masks_dir, mask_file)
            binary_mask = np.array(Image.open(mask_path))
            
            # Binarize to ensure 0 or 255
            binary_mask = (binary_mask > 127).astype(np.uint8)
            
            if combined_mask is None:
                combined_mask = np.zeros_like(binary_mask, dtype=np.uint8)
            
            # Mark current instance as instance_id
            combined_mask[binary_mask > 0] = instance_id
            instance_id += 1
            
            print(f"✅ Processed instance: {mask_file}")
        
        # Resize mask to target size
        if combined_mask.shape[:2] != target_size:
            print(f"🔄 Resizing mask: {combined_mask.shape[:2]} -> {target_size}")
            combined_mask_pil = Image.fromarray(combined_mask, mode='P')
            # Use nearest neighbor interpolation to preserve category IDs
            combined_mask_pil = combined_mask_pil.resize((target_size[1], target_size[0]), Image.Resampling.NEAREST)
            combined_mask = np.array(combined_mask_pil)
        
        # Convert to grayscale mask (foreground=1, background=0)
        grayscale_mask = np.zeros_like(combined_mask, dtype=np.uint8)
        grayscale_mask[combined_mask > 0] = 1  # All instances become 1
        
        # Save grayscale mask
        output_mask_path = os.path.join(mask_dir, "0000000.png")
        mask_image = Image.fromarray(grayscale_mask, mode='L')  # Grayscale mode
        mask_image.save(output_mask_path)
        print(f"💾 Saved grayscale mask: {output_mask_path} (size: {grayscale_mask.shape[:2]})")
        
        # Calculate and save centroids for GUI display
        centroids = _calculate_centroids(combined_mask)
        if start_point_save_path:
            _save_centroids_to_start_pos(centroids, start_point_save_path, resize_scale)
        print(f"📍 Calculated {len(centroids)} centroids")
        
        return True, len(mask_files)
        
    except Exception as e:
        print(f"❌ Processing results failed: {e}")
        return False, 0

def _save_prediction_results(pano_seg_info, pano_seg_numpy, image_size, mask_dir, target_size, start_point_save_path, resize_scale):
    """Save prediction results as first frame mask"""
    try:
        if not pano_seg_info:
            print("⚠️ No instances detected")
            return True, 0
        
        print(f"🎯 Found {len(pano_seg_info)} instances")
        
        # Create combined mask (each instance with different ID)
        height, width = image_size[1], image_size[0]  # PIL uses (width, height)
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        
        instance_id = 1
        for segment in pano_seg_info:
            segment_id = segment.get('id', 0)
            # Find all pixels of current instance
            instance_mask = (pano_seg_numpy == segment_id)
            combined_mask[instance_mask] = instance_id
            instance_id += 1
        
        # Resize mask to target size
        if combined_mask.shape[:2] != target_size:
            print(f"🔄 Resizing mask: {combined_mask.shape[:2]} -> {target_size}")
            combined_mask_pil = Image.fromarray(combined_mask, mode='P')
            # Use nearest neighbor interpolation to preserve category IDs
            combined_mask_pil = combined_mask_pil.resize((target_size[1], target_size[0]), Image.Resampling.NEAREST)
            combined_mask = np.array(combined_mask_pil)
        
        # Convert to grayscale mask (foreground=1, background=0)
        grayscale_mask = np.zeros_like(combined_mask, dtype=np.uint8)
        grayscale_mask[combined_mask > 0] = 1  # All instances become 1
        
        # Save grayscale mask
        output_mask_path = os.path.join(mask_dir, "0000000.png")
        mask_image = Image.fromarray(grayscale_mask, mode='L')  # Grayscale mode
        mask_image.save(output_mask_path)
        print(f"💾 Saved grayscale prediction mask: {output_mask_path} (size: {grayscale_mask.shape[:2]})")
        
        # Calculate and save centroids for GUI display
        centroids = _calculate_centroids(combined_mask)
        if start_point_save_path:
            _save_centroids_to_start_pos(centroids, start_point_save_path, resize_scale)
        print(f"📍 Calculated {len(centroids)} centroids")
        
        return True, len(pano_seg_info)
        
    except Exception as e:
        print(f"❌ Saving results failed: {e}")
        return False, 0



def _calculate_centroids(combined_mask):
    """Calculate centroids for each instance in the combined mask"""
    centroids = []
    unique_ids = np.unique(combined_mask)
    
    for instance_id in unique_ids:
        if instance_id == 0:  # Skip background
            continue
            
        # Find all pixels belonging to this instance
        instance_pixels = np.where(combined_mask == instance_id)
        
        if len(instance_pixels[0]) > 0:
            # Calculate centroid
            centroid_y = np.mean(instance_pixels[0])
            centroid_x = np.mean(instance_pixels[1])
            
            centroids.append({
                'id': int(instance_id),
                'x': float(centroid_x),
                'y': float(centroid_y)
            })
    
    return centroids

def _save_centroids_to_start_pos(centroids, start_point_save_path, resize_scale):
    """Save centroids in start_pos_array.txt format with same processing as manual clicks"""
    if not centroids:
        print("⚠️ No centroids to save")
        return
    
    # Convert centroids to numpy array format (n_points, 2) with [x, y] coordinates
    pos_clicks_array = np.array([[c['x'], c['y']] for c in centroids])
    
    # Apply the same coordinate transformation as manual clicks
    # This mimics: start_pos_array = pos_clicks_list/resize_scale
    start_pos_array = pos_clicks_array / resize_scale
    
    print(f'-------------Auto-predicted Points (x, y) in frame 0-------------')
    for idx, row in enumerate(start_pos_array, start=1):
        formatted_coords = ', '.join(f'({num:.2f})' for num in row)
        print(f"Point {idx}: {formatted_coords}")
    
    # Save in the same format as manual clicks
    start_pos_file = os.path.join(start_point_save_path, "start_pos_array.txt")
    os.makedirs(start_point_save_path, exist_ok=True)
    
    np.savetxt(start_pos_file, start_pos_array, fmt='%.2f', delimiter=',')
    print(f"💾 Saved {len(centroids)} auto-predicted centroids to {start_pos_file} (processed with resize_scale={resize_scale:.2f})")

def check_dependencies():
    """Check if dependencies are available"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check configuration file
        config_file = os.path.join(script_dir, "../initial_masks_inference/configs/seem/focall_unicl_lang_demo_low_threshold.yaml")
        if not os.path.exists(config_file):
            print(f"⚠️ Configuration file does not exist: {config_file}")
            return False
        
        # Check model file
        model_path = os.path.join(script_dir, "../../pretrained/model_state_dict.pt")
        if not os.path.exists(model_path):
            print(f"⚠️ Model file does not exist: {model_path}")
            return False
        
        print("✅ Dependencies check passed")
        return True
        
    except Exception as e:
        print(f"❌ Dependencies check failed: {e}")
        return False

if __name__ == "__main__":
    # Test code
    print("🧪 Testing auto initial prediction module")
    
    if not check_dependencies():
        print("❌ Dependencies check failed")
        sys.exit(1)
    
    # Test code can be added here
    print("✅ Module testing completed") 