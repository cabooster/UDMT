<center><img src="https://github.com/cabooster/UDMT/blob/page/images/logo_blue_v2.png?raw=true" width="750" align="middle" /></center>
<h1 align="center">UDMT: Unsupervised transfer learning enables multi-animal tracking without training annotation</h1>

### [Project page](https://cabooster.github.io/UDMT/) | [Paper](https://www.nature.com/articles/s41592-026-03051-8)

## Updates
<details>
  <summary>:triangular_flag_on_post:2026/09/08: Version 1.2.0. Replaced SAM ViT-B with SAM3 for first-frame click segmentation and forward propagation. Updated the recommended environment to Python 3.12 + PyTorch 2.7 (CUDA 12.6). Create training datasets can be stopped early and still export labels from the best completed parameters.<br>
:triangular_flag_on_post:2026/06/19: Version Update Summary: 1. Added `File > Add Video...` to add new videos to an existing project 2. Improved support for low-frame-rate videos by normalizing FPS values before frame-based tracking operations.<br>
:triangular_flag_on_post:2026/05/23: We have uploaded the corresponding frame-level manual annotations for the UDMT behavioral recording dataset to Zenodo: https://zenodo.org/records/20355567.<br>
:triangular_flag_on_post:2026/05/06: Our paper has been published in Nature Methods.</summary>
:triangular_flag_on_post:2025/07/30: Optimize the initialization method.<br>
:triangular_flag_on_post:2025/03/06: Added a log window and tooltips in the GUI.<br>
&emsp; A log window has been added at the bottom of the GUI to display runtime messages. <br>
&emsp; Tooltips have been added for buttons and the property panel to improve usability. <br>
:triangular_flag_on_post:2025/02/16: Fixed some bugs to improve stability.<br>
</details>

## Contents

- [Overview](#overview)
- [Installation](#Installation)
- [SAM 3 weights](#sam-3-weights)
- [GUI Tutorial](#gui-tutorial)
- [Q&A](#qa)
- [Results](#results)
- [License](./LICENSE)
- [Citation](#citation)

## Overview

Animal behavior is closely related to their internal state and external environment. **Quantifying animal behavior is a fundamental step in ecology, neuroscience, psychology, and various other fields.** However, there exist enduring challenges impeding multi-animal tracking advancing towards higher accuracy, larger scale, and more complex scenarios, especially the similar appearance and frequent interactions of animals of the same species.

Growing demands in quantitative ethology have motivated concerted efforts to develop high-accuracy and generalized tracking methods. **Here, we present UDMT, an unsupervised multi-animal tracking method that achieves state-of-the-art performance without requiring any human annotations.** The only thing users need to do is to click the animals in the first frame to specify the individuals they want to track. 

We demonstrate the state-of-the-art performance of UDMT on five different kinds of model animals, including mice, rats, *Drosophila*, *C. elegans*, and *Betta splendens*. Combined with a head-mounted miniaturized microscope, we recorded the calcium transients synchronized with mouse locomotion to decipher the correlations between animal locomotion and neural activity. 

For more details, please see the companion paper where the method first appeared: 
["*Unsupervised transfer learning enables multi-animal tracking without training annotation*"](https://www.nature.com/articles/s41592-026-03051-8).

<img src="https://github.com/cabooster/UDMT/blob/page/images/udmt_schematic.png?raw=true" width="700" align="middle">

## Installation
If you encounter any issues during installation or usage, please refer to the [Q&A section](#qa) for common solutions.
### 1. For Linux (Recommended)

#### Our Environment 

* Ubuntu 20.04 + (required)
* Python 3.12
* Pytorch 2.7.0
* NVIDIA GPU (GeForce RTX 4090) + CUDA (12.6+)

#### Environment Configuration 

1. Create a virtual environment and install PyTorch.

   ```
   $ conda create -n udmt python=3.12
   $ conda activate udmt
   $ sudo apt-get install ninja-build
   $ sudo apt-get install libturbojpeg
   ```
    If your CUDA version is **12.x**, run:
    ```
    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
    ```
    If your CUDA version is **11.x**, run:
    ```
    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
    ```
    **If you are not sure about your CUDA version, please refer to the [Q&A section](#qa).**
2. Install other dependencies.

   ```
   $ conda activate udmt
   $ git clone https://github.com/cabooster/UDMT.git
   $ cd UDMT/
   $ pip install -r requirements.txt
   $ pip install -r requirements_custom.txt
   $ conda install -c conda-forge xcb-util-cursor
   ```

3. Download SAM 3 weights (required, **not auto-downloaded**). See [SAM 3 weights](#sam-3-weights) below.

### 2. For Windows

> **Not recommended.**  
> The Windows installation is provided only as a reference for users who cannot access a Linux machine.  
> Since UDMT relies on specific PyTorch/CUDA configurations and several system-dependent packages, the Windows environment can be difficult to set up and debug.  
> For reproducible and stable use, please use the recommended Linux environment above.

#### Environment 

* Windows 10
* Python 3.12
* Pytorch 2.7.0
* NVIDIA GPU (GeForce RTX 3090) + CUDA (12.6+)

#### Environment Configuration 

1. Create a virtual environment and install PyTorch.

   ```
   $ conda create -n udmt python=3.12
   $ conda activate udmt
   ```
    If your CUDA version is **12.x**, run:
    ```
    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
    ```
    If your CUDA version is **11.x**, run:
    ```
    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
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

5. Download SAM 3 weights (required, **not auto-downloaded**). See [SAM 3 weights](#sam-3-weights) below.

### SAM 3 weights

SAM 3 is already integrated in UDMT for first-frame click segmentation and forward propagation. **SAM 3 weights are not auto-downloaded.** You must download `sam3.pt` yourself from the official SAM 3 repository and place it at:

```
./udmt/gui/tabs/xmem/sam_model/sam3.pt
```

Download steps (from [facebookresearch/sam3](https://github.com/facebookresearch/sam3)):

1. Request access to the checkpoints on the [SAM 3 Hugging Face repo](https://huggingface.co/facebook/sam3) and wait until you are approved.
2. Authenticate with a Hugging Face access token ([create a token](https://huggingface.co/settings/tokens) if you do not have one):

   ```
   $ hf auth login
   ```

3. Download `sam3.pt` into the UDMT folder above:

   ```
   $ cd UDMT/
   $ hf download facebook/sam3 sam3.pt --local-dir ./udmt/gui/tabs/xmem/sam_model
   ```

   You can also open https://huggingface.co/facebook/sam3/tree/main in a browser, download `sam3.pt`, and copy it to that directory.

4. Optional: if the checkpoint is stored elsewhere, set `UDMT_SAM3_CHECKPOINT=/path/to/sam3.pt` before launching the GUI.

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
3. Other pre-trained models (not SAM 3) will be downloaded automatically before launching the GUI. Alternatively, you can manually download them and place them in the specified location.

   **SAM 3 weights are not auto-downloaded.** Follow [SAM 3 weights](#sam-3-weights) to obtain `sam3.pt` from the official SAM 3 repository before using Tracking Initialization.

   | Model name                                                   | Location                         |
   | ------------------------------------------------------------ | -------------------------------- |
   | [trdimp_net_ep.pth.tar](https://zenodo.org/records/14671891/files/trdimp_net_ep.pth.tar?download=1) | `./udmt/gui/pretrained`          |
   | [XMem.pth](https://zenodo.org/records/14671891/files/XMem.pth?download=1) | `./udmt/gui/tabs/xmem/saves`     |
   | [model_state_dict.pt](https://zenodo.org/records/16625810/files/model_state_dict.pt?download=1) | `./udmt/gui/pretrained`   |

4. After **Forward Propagate** in Tracking Initialization, scrub through the video and check that every animal still has a mask. Crowded scenes can lose one or two masks later in the clip.

   If some masks are missing:
   - Pause on the **last frame that still has a complete set of masks**.
   - Click the missing animals to restore their masks.
   - Click **Forward Propagate** again from that frame.

   These recovery clicks are **not** written to `start_pos_array.txt`. Only clicks on the **first frame** (frame 0) are saved as start points. Do not go back to frame 0 to re-click, or those extra points will be recorded and can mislead later tracking.

#### **Quick Start with Demo Data**:

If you would like to try the GUI with a smaller dataset first, we provide **demo videos** ([5-mice video](https://zenodo.org/records/14689184/files/5-mice-1min.mp4?download=1) & [7-mice video](https://zenodo.org/records/14709082/files/7-mice-1min.mp4?download=1)) and pre-trained **models** (model for [5-mice](https://zenodo.org/records/14689184/files/DiMPnet_ep0020.pth.tar?download=1) and [7-mice](https://zenodo.org/records/14709082/files/DiMPnet_ep0020.pth.tar?download=1)).

- When creating a project, you can select the folder containing the demo video to import it.
- If you want to skip the **Network Training** process, place the downloaded model into the `your_project_path/models` folder before running the **Analyze Video** step.

Below is the tutorial video for the GUI. For detailed instructions on installing and using the GUI, please visit [**our website**](https://cabooster.github.io/UDMT/Tutorial/).

[![IMAGE ALT TEXT](https://github.com/cabooster/UDMT/blob/page/images/GUI-video2.png)](https://youtu.be/7rkpVTawpBU "Video Title")

## Q&A
### Q1: How can I check which CUDA version is installed on my system?
### A1: 
You can list the installed CUDA versions by checking the /usr/local directory:
   ```
   ls /usr/local | grep cuda
   ```
This will show all CUDA-related directories, such as:
   ```
   cuda-12
   cuda-12.4
   cuda-12.9
   ```
To check which version is currently set as default:
   ```
   ls -l /usr/local/cuda
   ```
   Example output:
   ```
   cuda -> /usr/local/cuda-12.1
   ```

### Q2: I get the error `ValueError: Unknown CUDA arch (8.9) or GPU not supported` when using an RTX 4090. How can I fix this?
### A2:

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
### Q3: I can't open the GUI when using VSCode or a terminal. What should I do?
### A3:
If you're trying to run the GUI from VSCode or a regular SSH terminal and encounter errors like `QXcbConnection: Could not connect to display` or the window simply doesn't appear, it's likely because the Linux server does not have a display environment or your SSH session lacks X11 forwarding.

To resolve this, we recommend using [MobaXterm](https://mobaxterm.mobatek.net/download.html) — a powerful SSH client with built-in X11 server support that allows you to run GUI applications on a remote Linux server seamlessly.

### Q4: The GUI says `SAM3 checkpoint not found: .../sam3.pt`. What should I do?
### A4:
SAM 3 is already integrated in the code, but **SAM 3 weights are not auto-downloaded**. Download `sam3.pt` from the official SAM 3 repository and place it at `./udmt/gui/tabs/xmem/sam_model/sam3.pt`. See [SAM 3 weights](#sam-3-weights).


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

- Li, Y., Zhang, Q., Zhang, Y. et al. Unsupervised transfer learning enables multi-animal tracking without training annotation. Nat Methods (2026). https://doi.org/10.1038/s41592-026-03051-8

```
@article {Li2026.05.06,
 title = {Unsupervised transfer learning enables multi-animal tracking without training annotation},
 author = {Li, Yixin and Zhang, Qi and Zhang, Yuanlong and Fan, Jiaqi and Lu, Zhi and Xu, Xinhong and Li, Xinyang and Li, Ziwei and Wu, Jiamin and Dai, Qionghai},
 journal = {Nature Methods}
 year = {2026},
 publisher = {Springer},
 doi = {10.1038/s41592-026-03051-8}
}
```
