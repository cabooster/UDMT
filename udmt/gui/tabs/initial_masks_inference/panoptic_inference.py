#!/usr/bin/env python3
# --------------------------------------------------------
# SEEM Panoptic Inference Script
# Based on demo/seem/app.py panoptic mode
# --------------------------------------------------------

import os
import sys
import warnings
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES
from utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def parse_arguments():
    parser = argparse.ArgumentParser('SEEM Panoptic Inference', description='Run panoptic segmentation on images')
    parser.add_argument('--input_image', type=str, required=True, 
                        help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Directory to save results (default: results)')
    parser.add_argument('--conf_files', type=str, default="configs/seem/focall_unicl_lang_demo.yaml", 
                        help='Path to config file')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to custom model weights (optional)')
    return parser.parse_args()

def setup_model(conf_files, model_path=None):
    """Setup SEEM model for panoptic inference"""
    print("🔧 Setting up SEEM model...")
    
    # Load configuration
    opt = load_opt_from_config_files([conf_files])
    opt = init_distributed(opt)
    
    # Determine model path
    if model_path and os.path.exists(model_path):
        pretrained_pth = model_path
        print(f"📁 Using custom model: {pretrained_pth}")
    else:
        # Auto-download default models
        if 'focalt' in conf_files:
            pretrained_pth = "seem_focalt_v0.pt"
            model_name = 'Focal-T'
        elif 'focal' in conf_files:
            pretrained_pth = "seem_focall_v0.pt"
            model_name = 'Focal-L'
        else:
            pretrained_pth = "seem_focall_v0.pt"
            model_name = 'Focal-L'
            
        if not os.path.exists(pretrained_pth):
            download_url = f"https://huggingface.co/xdecoder/SEEM/resolve/main/{pretrained_pth}"
            print(f"⬇️ Downloading {model_name} model...")
            os.system(f"wget {download_url}")
        
        print(f"✅ Using {model_name} model: {pretrained_pth}")
    
    # Build and load model
    print("🔄 Building model...")
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    
    # Initialize text embeddings for COCO classes
    print("📝 Initializing text embeddings...")
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            COCO_PANOPTIC_CLASSES + ["background"], is_eval=True
        )
    
    print("✅ Model setup complete!")
    return model

def load_and_preprocess_image(image_path):
    """Load and preprocess input image"""
    print(f"📷 Loading image: {image_path}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    print(f"📏 Original image size: {original_size}")
    
    # Preprocess image (resize to 512 as in the original demo)
    transform = transforms.Compose([
        transforms.Resize(512, interpolation=Image.BICUBIC)
    ])
    image_resized = transform(image)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).cuda().float()
    
    print(f"📏 Processed image size: {image_resized.size}")
    return image, image_resized, image_tensor

def run_panoptic_inference(model, image_tensor, image_resized):
    """Run panoptic segmentation inference"""
    print("🔄 Running panoptic inference...")
    
    # Prepare input data
    height, width = image_resized.size[1], image_resized.size[0]
    data = {
        "image": image_tensor, 
        "height": height, 
        "width": width
    }
    batch_inputs = [data]
    
    # Set up metadata for panoptic segmentation
    metadata = MetadataCatalog.get('coco_2017_train_panoptic')
    model.model.metadata = metadata
    
    # Run inference
    with torch.no_grad():
        print("⚙️ Processing...")
        results = model.model.evaluate(batch_inputs)
        
        # Extract panoptic segmentation results
        pano_seg = results[-1]['panoptic_seg'][0]
        pano_seg_info = results[-1]['panoptic_seg'][1]
        
        print("🎨 Generating visualization...")
        print(f"🔍 Found {len(pano_seg_info)} segments")
        if len(pano_seg_info) > 0:
            print(f"🔍 First segment keys: {list(pano_seg_info[0].keys())}")
        
        # Create solid color mask image
        mask_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Generate random colors for each instance
        for segment in pano_seg_info:
            segment_id = segment.get('id', 0)
            category_id = segment.get('category_id', 0)
            
            # Generate random color (avoid too dark or too bright)
            color = np.random.randint(64, 192, size=3)
            
            # Find all pixels of the current instance
            mask = (pano_seg.cpu().numpy() == segment_id)
            
            # Set these pixels to the corresponding color
            mask_image[mask] = color
        
        print("✅ Inference complete!")
        return mask_image, pano_seg_info, pano_seg.cpu().numpy()

def save_results(result_image, pano_seg_info, pano_seg_numpy, input_image_path, output_dir):
    """Save inference results"""
    print(f"💾 Saving results to: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get input filename
    input_name = Path(input_image_path).stem
    
    # Save solid color mask image
    result_pil = Image.fromarray(result_image)
    result_path = os.path.join(output_dir, f"{input_name}_panoptic_mask.png")
    result_pil.save(result_path)
    print(f"✅ Mask visualization saved: {result_path}")
    
    # Save binary mask separately for each category
    print("💾 Saving individual masks...")
    masks_dir = os.path.join(output_dir, f"{input_name}_individual_masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    for segment in pano_seg_info:
        segment_id = segment.get('id', 0)
        category_id = segment.get('category_id', 0)
        
        # Get category name
        if category_id < len(COCO_PANOPTIC_CLASSES):
            category_name = COCO_PANOPTIC_CLASSES[category_id].replace('/', '_')
        else:
            category_name = "unknown"
        
        # Create binary mask
        binary_mask = (pano_seg_numpy == segment_id).astype(np.uint8) * 255
        
        # Save binary mask
        mask_filename = f"{category_name}_{segment_id}.png"
        mask_path = os.path.join(masks_dir, mask_filename)
        Image.fromarray(binary_mask).save(mask_path)
    
    print(f"✅ Individual masks saved in: {masks_dir}")
    
    # Save segmentation info
    info_path = os.path.join(output_dir, f"{input_name}_panoptic_info.txt")
    with open(info_path, 'w') as f:
        f.write("Panoptic Segmentation Results\n")
        f.write("=" * 40 + "\n\n")
        
        for i, segment in enumerate(pano_seg_info):
            # Safely get segment information with default values
            category_id = segment.get('category_id', 0)
            area = segment.get('area', 0)
            is_thing = segment.get('isthing', False)
            segment_id = segment.get('id', 0)
            
            # Get category name
            if category_id < len(COCO_PANOPTIC_CLASSES):
                category_name = COCO_PANOPTIC_CLASSES[category_id]
            else:
                category_name = "unknown"
            
            f.write(f"Segment {i+1}:\n")
            f.write(f"  Category: {category_name} (ID: {category_id})\n")
            f.write(f"  Segment ID: {segment_id}\n")
            f.write(f"  Area: {area} pixels\n")
            f.write(f"  Type: {'Thing' if is_thing else 'Stuff'}\n")
            f.write(f"  Mask file: {category_name}_{segment_id}.png\n")
            f.write("\n")
    
    print(f"✅ Segmentation info saved: {info_path}")
    
    return result_path, info_path, masks_dir

def main():
    """Main inference function"""
    print("🚀 SEEM Panoptic Inference")
    print("=" * 50)
    
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Setup model
        model = setup_model(args.conf_files, args.model_path)
        
        # Load and preprocess image
        original_image, processed_image, image_tensor = load_and_preprocess_image(args.input_image)
        
        # Run inference
        result_image, pano_seg_info, pano_seg_numpy = run_panoptic_inference(model, image_tensor, processed_image)
        
        # Save results
        result_path, info_path, masks_dir = save_results(result_image, pano_seg_info, pano_seg_numpy, args.input_image, args.output_dir)
        
        print("\n🎉 Inference completed successfully!")
        print(f"📁 Results saved in: {args.output_dir}")
        print(f"🖼️ Visualization: {result_path}")
        print(f"📄 Segmentation info: {info_path}")
        print(f"🖼️ Individual masks: {masks_dir}")
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"   Input image: {args.input_image}")
        print(f"   Detected segments: {len(pano_seg_info)}")
        print(f"   Output directory: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 