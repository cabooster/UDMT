<center><img src="https://github.com/cabooster/UDMT/blob/page/images/logo_blue_v2.png?raw=true" width="750" align="middle" /></center>
<h1 align="center">UDMT: Unsupervised Multi-animal Tracking for Quantitative Ethology</h1>

### [Project page](https://cabooster.github.io/UDMT/) | [Paper](https://www.nature.com/articles/s41587-022-01450-8)

## Contents

- [Overview](#overview)
- [Installation](#Installation)
- [GUI Tutorial](#gui-tutorial)
- [Results](#results)
- [License](./LICENSE)
- [Citation](#citation)

## Overview

Animal behavior is closely related to their internal state and external environment. **Quantifying animal behavior is a fundamental step in ecology, neuroscience, psychology, and various other fields.** However, there exist enduring challenges impeding multi-animal tracking advancing towards higher accuracy, larger scale, and more complex scenarios, especially the similar appearance and frequent interactions of animals of the same species.

Growing demands in quantitative ethology have motivated concerted efforts to develop high-accuracy and generalized tracking methods. **Here, we present UDMT, the first unsupervised multi-animal tracking method that achieves state-of-the-art performance without requiring any human annotations.** The only thing users need to do is to click the animals in the first frame to specify the individuals they want to track. 

We demonstrate the state-of-the-art performance of UDMT on five different kinds of model animals, including mice, rats, *Drosophila*, *C. elegans*, and *Betta splendens*. Combined with a head-mounted miniaturized microscope, we recorded the calcium transients synchronized with mouse locomotion to decipher the correlations between animal locomotion and neural activity. 

For more details, please see the companion paper where the method first appeared: 
["*Unsupervised multi-animal tracking for quantitative ethology*"](https://www.nature.com/articles/s41587-022-01450-8).

<img src="https://github.com/cabooster/UDMT/blob/page/images/udmt_schematic.png?raw=true" width="800" align="middle">

## Installation
### 1. For Linux (Recommended)

#### Our environment 

* Ubuntu 20.04 (or newer)
* Python 3.8
* Pytorch 1.7.1
* NVIDIA GPU (GeForce RTX 3090) + CUDA (11.7)

#### Environment configuration 

1. Create a virtual environment and install PyTorch. **In the 3rd step, please select the correct Pytorch version that matches your CUDA version from [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/).** 

   ```
   $ conda create -n udmt python=3.8
   $ conda activate udmt
   $ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
   $ sudo apt-get install ninja-build
   $ sudo apt-get install libturbojpeg
   ```

2. We made a installable pip release of UDMT [[pypi](https://pypi.org/project/udmt-pip/)]. You can install it by entering the following command:

   ```
   $ pip install udmt-pip
   ```

### Download the source code

```
$ git clone https://github.com/cabooster/UDMT
$ cd UDMT/
```

### 2. For Windows

#### Our environment 

* Windows 10
* Python 3.8
* Pytorch 1.7.1
* NVIDIA GPU (GeForce RTX 3090) + CUDA (11.0)

#### Environment configuration 

1. Create a virtual environment and install PyTorch. **In the 3rd step, please select the correct Pytorch version that matches your CUDA version from [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/).** 

   ```
   $ conda create -n udmt python=3.8
   $ conda activate udmt
   $ pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. We made a installable pip release of UDMT [[pypi](https://pypi.org/project/udmt-pip/)]. You can install it by entering the following command:

   ```
   $ pip install udmt-pip     
   ```

3. Install Precise ROI pooling: If your environment is the same as ours, directly copy `<UDMT_install_path>\udmt\env_file\prroi_pool.pyd` to `<Anaconda_install_path>\anaconda3\envs\udmt\Lib\site-packages`.  Otherwise, build `prroi_pool.pyd` file with Visual Studio with the [tutorial](https://github.com/visionml/pytracking/blob/master/INSTALL_win.md#build-precise-roi-pooling-with-visual-studio-optional).

4. Install libjpeg-turbo: You can download installer from the official libjpeg-turbo [Sourceforge](https://sourceforge.net/projects/libjpeg-turbo/files/3.0.1/libjpeg-turbo-3.0.1-vc64.exe/download) repository, install it and copy `<libjpeg-turbo_install_path>\libjpeg-turbo64\bin\turbojpeg.dll` to the directory from the system PATH `C:\Windows\System32`.

## GUI Tutorial

We have released the Python source code and a user-friendly GUI of UDMT to make it an easily accessible tool for quantitative ethology and neuroethology. Below is the tutorial video for the GUI. For detailed instructions on installing and using the GUI, please visit [**our project page**](https://cabooster.github.io/UDMT/Tutorial/).

[![IMAGE ALT TEXT](https://github.com/cabooster/UDMT/blob/page/images/GUI-video2.png)](https://youtu.be/7rkpVTawpBU "Video Title")

#### Tips: 

- **Quick Start with Demo Data**:
  If you would like to try the GUI with a smaller dataset first, we provide a **demo video** ([Download Here](https://drive.google.com/file/d/13_sjRsFoZQ4kmnNKzdiDpxza0Gwu8fQn/view?usp=drive_link)) and a pre-trained **model** ([Download Here](https://drive.google.com/file/d/179JZLUcPfhE_B9XYt0YmIM1Jy6RK9fv8/view?usp=sharing)).
  - When creating a project, you can select the folder containing the demo video to import it.
  - If you want to skip the **Network Training** process, place the downloaded model into the `your_project_path/models` folder before running the **Analyze Video** step.

## Results

### 1. Tracking the movement of 10 mice simultaneously with UDMT.

[![IMAGE ALT TEXT](https://github.com/cabooster/UDMT/blob/page/images/sv1_video.png)](https://youtu.be/yFT3AdmNVg8 "Video Title")

### 2. Neuroethology analysis of multiple mice combined with a head-mounted microscope.

[![IMAGE ALT TEXT](https://github.com/cabooster/UDMT/blob/page/images/sv5_video.png)]( https://youtu.be/zufYK1ovlLU "Video Title")

### 3. Analyzing the aggressive behavior of betta fish with UDMT.

[![IMAGE ALT TEXT](https://github.com/cabooster/UDMT/blob/page/images/sv8_video.png)](https://youtu.be/z724dDa0CRM "Video Title")

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
