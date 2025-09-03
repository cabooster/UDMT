# SEEM Lightweight Inference Environment

## 📁 Directory Structure
```
seem_lightweight_inference/
├── run_custom_organisms_batch.py          # Batch inference script (main usage)
├── run_custom_organisms_simple.py         # Single image inference script
├── panoptic_inference.py                  # Core inference engine
├── modeling/                               # Model architecture code
├── utils/                                  # Utility functions
├── configs/                               # Configuration files
│   └── seem/
│       └── focall_unicl_lang_demo.yaml    # Standard configuration
└── README.md                              # This file
```

## 🚀 Usage

### 1. Batch Inference (Recommended)
```bash
cd /data1/liyixin/seem_lightweight_inference
python run_custom_organisms_batch.py
```

### 2. Single Image Inference
```bash
cd /data1/liyixin/seem_lightweight_inference  
python run_custom_organisms_simple.py
```

## ⚙️ Configuration

- **Standard Configuration**: `configs/seem/focall_unicl_lang_demo.yaml`

## 🎯 Output Results

Results are saved in the `simple_custom_results/` directory:
- `*_panoptic_mask.png` - Solid color segmentation results
- `*_individual_masks/` - Individual instance masks
- `*_panoptic_info.txt` - Detailed segmentation information

## 📋 Dependencies

Ensure the following Python packages are installed:
- torch
- torchvision  
- detectron2
- PIL
- numpy
- yaml

## 💡 Tips

- Use your fine-tuned model weights for inference
- Manually edit configuration files if detection threshold adjustment is needed
- Suitable for batch processing multiple images
