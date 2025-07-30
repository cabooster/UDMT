<center><img src="https://github.com/cabooster/UDMT/blob/page/images/logo_blue_v2.png?raw=true" width="750" align="middle" /></center>
<h1 align="center">UDMT: Unsupervised Multi-animal Tracking for Quantitative Ethology</h1>

### [Project page](https://cabooster.github.io/UDMT/) | [Paper](https://www.biorxiv.org/content/10.1101/2025.01.23.634625v1)

## Updates
<details>
  <summary>:triangular_flag_on_post:2025/07/30: Optimize the initialization method. </summary>
  <summary>:triangular_flag_on_post:2025/03/06: Added a log window and tooltips in the GUI. </summary>
  
A log window has been added at the bottom of the GUI to display runtime messages.  
Tooltips have been added for buttons and the property panel to improve usability.  

  <summary>:triangular_flag_on_post:2025/02/16: Fixed some bugs to improve stability. </summary>
</details>

## Contents

- [Overview](#overview)
- [Installation](#Installation)
- [GUI Tutorial](#gui-tutorial)
- [Q&A](#qa)
- [Results](#results)
- [License](./LICENSE)
- [Citation](#citation)

## Overview

Animal behavior is closely related to their internal state and external environment. **Quantifying animal behavior is a fundamental step in ecology, neuroscience, psychology, and various other fields.** However, there exist enduring challenges impeding multi-animal tracking advancing towards higher accuracy, larger scale, and more complex scenarios, especially the similar appearance and frequent interactions of animals of the same species.

Growing demands in quantitative ethology have motivated concerted efforts to develop high-accuracy and generalized tracking methods. **Here, we present UDMT, the first unsupervised multi-animal tracking method that achieves state-of-the-art performance without requiring any human annotations.** The only thing users need to do is to click the animals in the first frame to specify the individuals they want to track. 

We demonstrate the state-of-the-art performance of UDMT on five different kinds of model animals, including mice, rats, *Drosophila*, *C. elegans*, and *Betta splendens*. Combined with a head-mounted miniaturized microscope, we recorded the calcium transients synchronized with mouse locomotion to decipher the correlations between animal locomotion and neural activity. 

For more details, please see the companion paper where the method first appeared: 
["*Unsupervised multi-animal tracking for quantitative ethology*"](https://www.biorxiv.org/content/10.1101/2025.01.23.634625v1).

<img src="https://github.com/cabooster/UDMT/blob/page/images/udmt_schematic.png?raw=true" width="800" align="middle">

## Installation
If you encounter any issues during installation or usage, please refer to the [Q&A section](#qa) for common solutions.
### 1. For Linux (Recommended)

#### Our Environment 

* Ubuntu 20.04 + (required)
* Python 3.8
* Pytorch 2.1.1
* NVIDIA GPU (GeForce RTX 4090) + CUDA (12+)

#### Environment Configuration 

1. Create a virtual environment and install PyTorch.

   ```
   $ conda create -n udmt python=3.8
   $ conda activate udmt
   $ sudo apt-get install ninja-build
   $ sudo apt-get install libturbojpeg
   ```
    If your CUDA version is **12.x**, run:
    ```
    pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
    ```
    If your CUDA version is **11.x**, run:
    ```
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
    ```
2. Install other dependencies.

   ```
   $ conda activate udmt
   $ git clone https://github.com/cabooster/UDMT.git
   $ cd UDMT/
   $ pip install -r requirements.txt
   ```

### 2. For Windows

#### Environment 

* Windows 10
* Python 3.8
* Pytorch 1.7.1
* NVIDIA GPU (GeForce RTX 3090) + CUDA (11.0)

#### Environment Configuration 

1. Create a virtual environment and install PyTorch.

   ```
   $ conda create -n udmt python=3.8
   $ conda activate udmt
   ```
    If your CUDA version is **12.x**, run:
    ```
    pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
    ```
    If your CUDA version is **11.x**, run:
    ```
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
    ```
2. Install other dependencies.

   ```
   $ conda activate udmt
   $ git clone https://github.com/cabooster/UDMT.git
   $ cd UDMT/
   $ pip install -r requirements.txt
   ```

3. Install Precise ROI pooling: If your environment is the same as ours, directly copy `<UDMT_install_path>\udmt\env_file\prroi_pool.pyd` to `<Anaconda_install_path>\anaconda3\envs\udmt\Lib\site-packages`.  Otherwise, build `prroi_pool.pyd` file with Visual Studio with the [tutorial](https://github.com/visionml/pytracking/blob/master/INSTALL_win.md#build-precise-roi-pooling-with-visual-studio-optional).

4. Install libjpeg-turbo: You can download installer from the official libjpeg-turbo [Sourceforge](https://sourceforge.net/projects/libjpeg-turbo/files/3.0.1/libjpeg-turbo-3.0.1-vc64.exe/download) repository, install it and copy `<libjpeg-turbo_install_path>\libjpeg-turbo64\bin\turbojpeg.dll` to the directory from the system PATH `C:\Windows\System32`.

## GUI Tutorial

We have released the Python source code and a user-friendly GUI of UDMT to make it an easily accessible tool for quantitative ethology and neuroethology. 

<center><img src="https://github.com/cabooster/UDMT/blob/page/images/GUI-home-page2.png?raw=true" width="800" align="middle"></center>

1. Once you have UDMT installed, start by opening a terminal. Activate the environment and download the codes with:

   ```
   $ conda activate udmt
   $ git clone https://github.com/cabooster/UDMT.git
   $ cd UDMT/
   ```
   Causion: To install the library, please git clone the repository instead of downloading the zip file, since source files inside the folder are symbol-linked. Downloading the repository as a zip file will break these symbolic links. 
2. To launch the GUI, simply enter in the terminal:

   ```
   $ python -m udmt.gui.launch_script
   ```
   If your server has multiple GPUs and you want to control which one is used when launching the GUI, you can use the `CUDA_VISIBLE_DEVICES` environment variable to specify the GPU index. For example, to run the GUI using **GPU 3** (the fourth GPU), use the following command:
    ```
    $ CUDA_VISIBLE_DEVICES=3 python -m udmt.gui.launch_script
    ```
#### **Quick Start with Demo Data**:

If you would like to try the GUI with a smaller dataset first, we provide **demo videos** ([5-mice video](https://zenodo.org/records/14689184/files/5-mice-1min.mp4?download=1) & [7-mice video](https://zenodo.org/records/14709082/files/7-mice-1min.mp4?download=1)) and pre-trained **models** (model for [5-mice](https://zenodo.org/records/14689184/files/DiMPnet_ep0020.pth.tar?download=1) and [7-mice](https://zenodo.org/records/14709082/files/DiMPnet_ep0020.pth.tar?download=1)).

- When creating a project, you can select the folder containing the demo video to import it.
- If you want to skip the **Network Training** process, place the downloaded model into the `your_project_path/models` folder before running the **Analyze Video** step.

Below is the tutorial video for the GUI. For detailed instructions on installing and using the GUI, please visit [**our website**](https://cabooster.github.io/UDMT/Tutorial/).

[![IMAGE ALT TEXT](https://github.com/cabooster/UDMT/blob/page/images/GUI-video2.png)](https://youtu.be/7rkpVTawpBU "Video Title")

## Q&A

### Q1: I get the error `ValueError: Unknown CUDA arch (8.9) or GPU not supported` when using an RTX 4090. How can I fix this?
### A1:

This error occurs when PyTorch attempts to compile a CUDA extension for Ada Lovelace GPUs (e.g., RTX 4090), which use compute capability `8.9`, but does not recognize this architecture in its internal list.
You can fix this by **manually patching PyTorch’s CUDA architecture list**.

1. Locate and open the following file in your Python environment:

   ```
   <your_conda_env>/lib/pythonX.X/site-packages/torch/utils/cpp_extension.py
   ```
2. Inside the _get_cuda_arch_flags() function, find the supported_arches list and add '8.9':
  ```
  supported_arches = ['3.5', '3.7', '5.0', '5.2', '5.3', '6.0', '6.1', '6.2',
                      '7.0', '7.2', '7.5', '8.0', '8.6', '8.9']
  ```
3. Also add support for the "Ada" name in named_arches:
  ```
  named_arches = collections.OrderedDict([
      ('Kepler+Tesla', '3.7'),
      ('Kepler', '3.5+PTX'),
      ('Maxwell+Tegra', '5.3'),
      ('Maxwell', '5.0;5.2+PTX'),
      ('Pascal', '6.0;6.1+PTX'),
      ('Volta', '7.0+PTX'),
      ('Turing', '7.5+PTX'),
      ('Ampere', '8.0;8.6+PTX'),
      ('Ada', '8.9+PTX')  # ← Add this line
  ])
  ```
Save the file and re-run your code. PyTorch will now be able to compile CUDA extensions for compute capability 8.9 (e.g., RTX 4090).
### Q2: I can't open the GUI when using VSCode or a terminal. What should I do?
### A2:
If you're trying to run the GUI from VSCode or a regular SSH terminal and encounter errors like `QXcbConnection: Could not connect to display` or the window simply doesn't appear, it's likely because the Linux server does not have a display environment or your SSH session lacks X11 forwarding.

To resolve this, we recommend using [MobaXterm](https://mobaxterm.mobatek.net/download.html) — a powerful SSH client with built-in X11 server support that allows you to run GUI applications on a remote Linux server seamlessly.


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

- Yixin Li, Xinyang Li, Qi Zhang, et al. Unsupervised multi-animal tracking for quantitative ethology. bioRxiv (2025). [https://doi.org/10.1101/2025.01.23.634625](https://www.biorxiv.org/content/10.1101/2025.01.23.634625v1)

```
@article {Li2025.01.23.634625,
 title = {Unsupervised multi-animal tracking for quantitative ethology},
 author = {Li, Yixin and Li, Xinyang and Zhang, Qi and Zhang, Yuanlong and Fan, Jiaqi and Lu, Zhi and Li, Ziwei and Wu, Jiamin and Dai, Qionghai},
 journal = {bioRxiv}
 year = {2025},
 publisher = {Cold Spring Harbor Laboratory},
 doi = {10.1101/2025.01.23.634625}
}
```
