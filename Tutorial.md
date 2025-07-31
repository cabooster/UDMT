---
layout: page
title: UDMT tutorial
---
<img src="https://github.com/cabooster/UDMT/blob/page/images/logo_blue_v2.png?raw=true" width="400" align="right" />

## Installation
If you encounter any issues during installation or usage, please refer to the [Q&A](https://github.com/cabooster/UDMT#qa) for common solutions.
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

## GUI tutorial

### 1. Start UDMT in the terminal

1. Once you have UDMT installed, start by opening a terminal. Activate the environment and download the codes with:

   ```
   $ source activate udmt
   $ git clone https://github.com/cabooster/UDMT
   $ cd UDMT/
   ```

2. To launch the GUI, simply enter in the terminal:

   ```
   $ python -m udmt.gui.launch_script
   ```


3. Pre-trained models will be downloaded automatically before launching the GUI. Alternatively, you can manually download the model and place it in the specified location.

   | Model name                                                   | Location                         |
   | ------------------------------------------------------------ | -------------------------------- |
   | [trdimp_net_ep.pth.tar](https://zenodo.org/records/14671891/files/trdimp_net_ep.pth.tar?download=1) | `./udmt/gui/pretrained`          |
   | [XMem.pth](https://zenodo.org/records/14671891/files/XMem.pth?download=1) | `./udmt/gui/tabs/xmem/saves`     |
   | [sam_vit_b_01ec64.pth](https://zenodo.org/records/14671891/files/sam_vit_b_01ec64.pth?download=1) | `./udmt/gui/tabs/xmem/sam_model` |
   | [model_state_dict.pt](https://zenodo.org/records/16625810/files/model_state_dict.pt?download=1) | `./udmt/gui/pretrained`          |

### 2. Create a new project

1. Click the **Create New Project** button.

   <center><img src="https://github.com/cabooster/UDMT/blob/page/images/GUI-start1.png?raw=true" width="700" align="middle" style="box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.5); border-radius: 8px;"></center>

2. Set the **Project Path** and **Project Name**.

3. Add videos by clicking the **Browse Videos** button, then select the folders containing your videos. By default, the selected videos will be copied to the project path.

4. Finally, click the **Create** button to complete this process. The project you created will open automatically.

   <center><img src="https://github.com/cabooster/UDMT/blob/page/images/GUI-start2.png?raw=true" width="700" align="middle" style="box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.5); border-radius: 8px;"></center>

**Open an existing project**

1. To reopen a project, click the **Load Project** button on the home page.

2. Navigate to your project folder and select the **config.yaml** file.

#### Directory structure

<details>
  <summary>Click to unfold the directory tree</summary>
<pre><code>
UDMT
|-- config.yaml
|-- videos
    |-- video1.mp4
    |-- video2.mp4
|-- training-datasets
    |-- video1
        |-- img
        	|-- ...
        |-- label
        	|-- ...
|-- models
    |-- video1
        |-- DiMPnet_ep0018.pth.tar
        |-- DiMPnet_ep0019.pth.tar
        |-- DiMPnet_ep0020.pth.tar
|-- tracking-results
    |-- video1
        |-- video1-whole-filter5.mp4
        |-- video1-whole-filter5.npy
        |-- tmp-videos
        	|-- ...
|-- tmp
    |-- video1
        |-- evaluation_metric.json
        |-- evaluation_metric_for_test.json
        |-- evaluation_metric_for_train.json
        |-- extracted-images
            |-- start_pos_array.txt
            |-- 00000.jpg
			|-- ...
        |-- images
        	|-- ...
        |-- masks
        	|-- ...
        |-- test_set_results
        	|-- ...
        |-- train_set_results
        	|-- ...

</code></pre>

</details>

#### Folder description

`project_name/videos`: Path to save the videos to be processed.

`project_name/training-datasets`: Path to save the training datasets.

`project_name/models`: Path to save the models.

`project_name/tracking-results`: Path to save the tracking results, including `.npy` files and `.MP4` files showing the tracking trajectories.

`project_name/tmp`: Path to save temporary files, including images used during tracking initialization, extracted frames, and files generated during automatic parameter tuning.

### 3. Tracking initialization

1. Click the **Select Videos** button to choose a video you want to process (you can select videos from the `videos` folder in your project file).

   <center><img src="https://github.com/cabooster/UDMT/blob/page/images/GUI-ini1.png?raw=true" width="700" align="middle" style="box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.5); border-radius: 8px;"></center>

2. After selecting the video, click the **Launch Tracking Initialization GUI** button to open the sub-GUI.

   <center><img src="https://github.com/cabooster/UDMT/blob/page/images/GUI-ini2.png?raw=true" width="700" align="middle" style="box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.5); border-radius: 8px;"></center>

3. In the sub-GUI, click on the centroids of all the animals you want to track.

4. Click **Forward Propagate** to start the foreground extraction process.

5. Wait until foreground extraction is completed for all frames. After foreground extraction, check whether all animals are marked with a red mask. If many consecutive frames fail to extract an animal (which usually happens with small animals like flies), please increase 'memory frame every' to 10,000. Once finished, close the sub-GUI to proceed to the next tab.

### 4. Create training dataset

1. Click the **Select Videos** button to choose a video you want to process (if you have already selected videos in the previous tab, you don't need to select again).
 
2. After selecting the video, choose the parameters for preprocessing the videos.

3. Click the **Create Training Dataset** button to start the creation process.

4. During the process, you can click the **Show Video** button at any time to visualize the creation progress.

5. Once the creation is complete, a popup message will appear, notifying you to proceed to the next tab.

<center><img src="https://github.com/cabooster/UDMT/blob/page/images/GUI-create.png?raw=true"  width="700" align="middle" style="box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.5); border-radius: 8px;"></center>

### 5. Train network

1. Click the **Select Videos** button to choose a video you want to process (if you have already selected videos in the previous tab, you don't need to select them again).

2. Set the training parameters:

   **Batch Size**: Select a larger batch size if you have multiple GPUs. If you encounter a "CUDA out of memory" error, reduce the batch size.

   **Number of Workers**: Set to 0 on Windows; on Linux, you can set it to 8.

   **Max Training Epochs**: The default is 20 epochs. This value doesn't need to be changed unless necessary.

3. Click the **Train Network** button to start the training process. Note: Due to thread occupation issues during network training, the training phase output logs will be displayed in the console instead of the GUI log window.

<center><img src="https://github.com/cabooster/UDMT/blob/page/images/GUI-train.png?raw=true"  width="700" align="middle"  style="box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.5); border-radius: 8px;"></center>

### 6. Analyze videos

1. Select a video you want to process.
 
2. Choose the parameters for preprocessing the videos and set the **Filter Size** for post-processing smooth trajectory filtering.

3. Click the **Analyze Videos** button to start the tracking process.

4. During the tracking, you can click the **Show Video** button at any time to visualize the tracking process. (This includes **automatic parameter tuning** and the final tracking process.)

5. Once the tracking is complete, the result files will be automatically saved in the **tracking-results** folder.

6. A popup message will appear, notifying you to proceed to the next tab.


<center><img src="https://github.com/cabooster/UDMT/blob/page/images/GUI-analyze.png?raw=true" width="700" align="middle" style="box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.5); border-radius: 8px;"></center>

### 7. Visualize tracks

1. Select a video you want to process.

2. Choose the parameters for post-processing the videos.

   <center><img src="https://github.com/cabooster/UDMT/blob/page/images/GUI-vis1.png?raw=true" width="700" align="middle" style="box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.5); border-radius: 8px;"></center>

3. Click the **Launch Track Visualization GUI** button to start the visualization.

4. In the pop-up sub-GUI: The left side shows the current frame and the right side displays the trajectory XY coordinates that change with each frame.

5. You can click the **Frame** bar to visualize each frame.

6. Adjust the **Dot Size** bar to change the size of the dots marking positions in the image.

<center><img src="https://github.com/cabooster/UDMT/blob/page/images/GUI-vis2.png?raw=true" width="700" align="middle" style="box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.5); border-radius: 8px;"></center>

## GUI demo video

<center><iframe width="800" height="500" src="https://www.youtube.com/embed/7rkpVTawpBU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> </center>
