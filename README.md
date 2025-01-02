# UDMT: **Unsupervised multi-animal tracking for** quantitative ethology

<center><img src="images/logo_blue.png" width="700" align="middle" /></center>

### [Project page](https://cabooster.github.io/UDMT/) | [Paper](https://www.nature.com/articles/s41587-022-01450-8)

## Contents

- [Overview](#overview)
- [Directory structure](#directory-structure)
- [Pytorch code](#pytorch-code)
- [GUI](#gui)
- [Results](#results)
- [License](./LICENSE)
- [Citation](#citation)

## Overview

Animal behavior is closely related to their internal state and external environment. **Quantifying animal behavior is a fundamental step in ecology, neuroscience, psychology, and various other fields.** However, there exist enduring challenges impeding multi-animal tracking advancing towards higher accuracy, larger scale, and more complex scenarios, especially the similar appearance and frequent interactions of animals of the same species.

Growing demands in quantitative ethology has motivated concerted efforts to develop high-accuracy and generalized tracking methods. **Here, we present <u>the first</u> unsupervised multi-animal tracking method that achieves state-of-the-art performance without requiring any human annotations.** The only thing users need to do is to click the animals in the first frame to specify the individuals they want to track. 

We demonstrate the state-of-the-art performance of UDMT on five different kinds of model animals, including mice, rats, *Drosophila*, *C. elegans*, and *Betta splendens*. Combined with a head-mounted miniaturized microscope, we recorded the calcium transients synchronized with mouse locomotion to decipher the correlations between animal locomotion and neural activity. 

For more details, please see the companion paper where the method first appeared: 
["*Unsupervised multi-animal tracking for quantitative ethology*"](https://www.nature.com/articles/s41587-022-01450-8).

<img src="images/schematic.png" width="800" align="middle">

## Directory structure

<details>
  <summary>Click to unfold the directory tree</summary>

```
UDMT
|---UDMT_pytorch #Pytorch implementation of DeepCAD-RT#
|---UDMT_GUI #Python GUI of DeepCAD-RT#
```
- **DeepCAD_RT_pytorch** contains the Pytorch implementation of DeepCAD-RT (Python scripts, Jupyter notebooks, Colab notebook)
- **DeepCAD_RT_GUI** contains all C++ and Matlab files for the real-time implementation of DeepCAD-RT

</details>



## Pytorch code
### Our environment 

* Ubuntu 16.04 
* Python 3.9
* Pytorch 1.8.0
* NVIDIA GPU (GeForce RTX 3090) + CUDA (11.1)

### Environment configuration

1. Create a virtual environment and install PyTorch. **In the 3rd step, please select the correct Pytorch version that matches your CUDA version from [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/).** 

   ```
   $ conda create -n deepcadrt python=3.9
   $ conda activate deepcadrt
   $ pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
   ```
   
      *Note:  `pip install` command is required for Pytorch installation.*
  
2. We made a installable pip release of DeepCAD-RT [[pypi](https://pypi.org/project/deepcad/)]. You can install it by entering the following command:

   ```
   $ pip install deepcad
   ```

### Download the source code

```
$ git clone https://github.com/cabooster/DeepCAD-RT
$ cd DeepCAD-RT/DeepCAD_RT_pytorch/
```

### Demos

To try out the Python code, please activate the `deepcadrt` environment first:

```
$ source activate deepcadrt
$ cd DeepCAD-RT/DeepCAD_RT_pytorch/
```

**Example training**

To train a DeepCAD-RT model, we recommend starting with the demo script `demo_train_pipeline.py`. One demo dataset will be downloaded to the `DeepCAD_RT_pytorch/datasets` folder automatically. You can also download other data from [the companion webpage](https://cabooster.github.io/DeepCAD-RT/Datasets/) or use your own data by changing the training parameter `datasets_path`. 

```
python demo_train_pipeline.py
```

**Example testing**

To test the denoising performance with pre-trained models, you can run the demo script `demo_test_pipeline.py` . A demo dataset and its denoising model will be automatically downloaded to `DeepCAD_RT_pytorch/datasets` and `DeepCAD_RT_pytorch/pth`, respectively. You can change the dataset and the model by changing the parameters `datasets_path` and `denoise_model`.

```
python demo_test_pipeline.py
```

### Jupyter notebook

We provide simple and user-friendly Jupyter notebooks to implement DeepCAD-RT. They are in the `DeepCAD_RT_pytorch/notebooks` folder. Before you launch the notebooks, please configure an environment following the instruction in [Environment configuration](#environment-configuration) . And then, you can launch the notebooks through the following commands:

```
$ source activate deepcadrt
$ cd DeepCAD-RT/DeepCAD_RT_pytorch/notebooks
$ jupyter notebook
```

<center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/deepcad8.png?raw=true" width="800" align="middle"></center> 

### Colab notebook

We also provide a cloud-based notebook implemented with Google Colab. You can run DeepCAD-RT directly in your browser using a cloud GPU without configuring the environment. 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cabooster/DeepCAD-RT/blob/main/DeepCAD_RT_pytorch/notebooks/DeepCAD_RT_demo_colab.ipynb)

*Note: The Colab notebook needs much longer time to train and test because of the limited GPU performance offered by Colab.*

<center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/deepcad7.png?raw=true" width="800" align="middle"></center> 

## GUI

To achieve real-time denoising, DeepCAD-RT was optimally deployed on GPU using TensorRT (Nvidia) for further acceleration and memory reduction. We also designed a sophisticated time schedule for multi-thread processing. Based on a two-photon microscope, real-time denoising has been achieved with our Matlab GUI of DeepCAD-RT (tested on a Windows desktop with Intel i9 CPU and 128 GB RAM).  **Tutorials** on installing and using the GUI has been moved to [**this page**](https://github.com/cabooster/DeepCAD-RT/tree/main/DeepCAD_RT_GUI).  

<center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/GUI2.png?raw=true" width="950" align="middle"></center> 

## Results

### 1. Tracking the movement of 10 mice simultaneously with UDMT.

[![IMAGE ALT TEXT](images/sv1_video.png)](https://youtu.be/yFT3AdmNVg8 "Video Title")

### 2. Neuroethology analysis of multiple mice combined with a head-mounted microscope.

[![IMAGE ALT TEXT](images/sv5_video.png)]( https://youtu.be/zufYK1ovlLU "Video Title")

### 3. Analyzing the aggressive behavior of betta fish with UDMT.

[![IMAGE ALT TEXT](images/sv8_video.png)](https://youtu.be/z724dDa0CRM "Video Title")

More demo videos are presented on [our website](https://cabooster.github.io/UDMT/Gallery/).

## Citation

If you use this code, please cite the companion paper where the original method appeared: 

- Xinyang Li, Yixin Li, Yiliang Zhou, et al. Real-time denoising enables high-sensitivity fluorescence time-lapse imaging beyond the shot-noise limit. Nat. Biotechnol. (2022). [https://doi.org/10.1038/s41587-022-01450-8](https://www.nature.com/articles/s41587-022-01450-8)



```
@article {li2022realtime,
  title = {Real-time denoising enables high-sensitivity fluorescence time-lapse imaging beyond the shot-noise limit},
  author = {Li, Xinyang and Li, Yixin and Zhou, Yiliang and Wu, Jiamin and Zhao, Zhifeng and Fan, Jiaqi and Deng, Fei and Wu, Zhaofa and Xiao, Guihua and He, Jing and Zhang, Yuanlong and Zhang, Guoxun and Hu, Xiaowan and Chen, Xingye and Zhang, Yi and Qiao, Hui and Xie, Hao and Li, Yulong and Wang, Haoqian and Fang, Lu and Dai, Qionghai},
  journal={Nature Biotechnology},
  year={2022},
  publisher={Nature Publishing Group}
}
```
