---
layout: page
title: DeepCAD-RT tutorial
---
<img src="https://github.com/cabooster/UDMT/blob/main/images/logo_blue_v2.png?raw=true" width="700" align="right" />



## GUI Tutorial

### 1. Start UDMT in the terminal

1. Once you have UDMT installed, start by opening a terminal. If you installed via the recommended method, activate the environment with:

   ```
   $ source activate udmt
   $ cd UDMT/
   ```

2. To launch the GUI, simply enter in the terminal:

   ```
   $ python -m udmt.gui.launch_script
   ```


3. Pre-trained models will be downloaded automatically before launching the GUI. Alternatively, you can manually download the model and place it in the specified location.

   | Model name                                                   | Location                                        |
   | ------------------------------------------------------------ | ----------------------------------------------- |
   | [trdimp_net_ep.pth.tar](https://zenodo.org/records/14671891/files/sam_vit_b_01ec64.pth?download=1) | `.\UDMT-main-pyqt\udmt\gui\pretrained`          |
   | [XMem.pth](https://zenodo.org/records/14671891/files/XMem.pth?download=1) | `.\UDMT-main-pyqt\udmt\gui\tabs\xmem\saves`     |
   | [sam_vit_b_01ec64.pth](https://zenodo.org/records/14671891/files/sam_vit_b_01ec64.pth?download=1) | `.\UDMT-main-pyqt\udmt\gui\tabs\xmem\sam_model` |

### 2. Create a project

### 3. Tracking initialization

<img src="M:\11-LYX\25-code-release\images\GUI-ini1.png" alt="description" style="box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.5); border-radius: 8px;">

### 4. Create training dataset

<img src="M:\11-LYX\25-code-release\images\GUI-create.png" alt="description" style="box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.5); border-radius: 8px;">

### 5. Train network

<img src="M:\11-LYX\25-code-release\images\GUI-train.png" alt="description" style="box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.5); border-radius: 8px;">

### 6. Analyze videos

<img src="M:\11-LYX\25-code-release\images\GUI-analyze.png" alt="description" style="box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.5); border-radius: 8px;">

## 