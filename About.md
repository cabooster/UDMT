---
layout: page
title: About
---
<img src="https://github.com/cabooster/UDMT/blob/main/images/logo_blue.png?raw=true" width="700" align="right" />
### [GitHub](https://github.com/cabooster/UDMT) | [Paper](https://www.nature.com/articles/s41587-022-01450-8)
## Content

- [Introduction](#introduction)
- [Results](#results)
- [Citation](#citation)

## Introduction

### Background
Animal behavior is closely related to their internal state and external environment. Quantifying animal behavior is a fundamental step in ecology, neuroscience, psychology, and various other fields. As the most basic representation of behavior, the position of animals can reflect the locomotion of individuals and is an indispensable metric for behavioral analysis, especially for population-level studies involving multiple animals. Over the past few decades, the technology for animal tracking evolves continuously and recent advances have catalyzed a series of scientific discoveries. Specifically, tracking insects in laboratories permits the identification of neural circuits and genes involved in visual navigation and locomotion control. In the wild, statistics on individual and population-level animal migration allows the revelation of disruption and ecotoxicology caused by chemical pollution.


However, there exist enduring challenges impeding multi-animal tracking advancing towards higher accuracy, larger scale, and more complex scenarios, especially the similar appearance and frequent interactions of animals of the same species.
- **High Annotation Workload:** Supervised-learning-based tracking methods achieve good performance but require large amounts of manual annotations, making them time-consuming and labor-intensive, especially with increasing animal numbers and behavioral diversity.

- **Limitations of Semi-Automatic Annotation:** While semi-automatic annotation through GUIs reduces human intervention, it struggles in complex environments and low-contrast conditions, where threshold-based segmentation becomes ineffective.

- **Tracking Accuracy Challenges:** Frequent animal interactions and occlusions lead to identity switches and trajectory errors in existing methods, resulting in accumulated inaccuracies that degrade overall tracking performance.



<center><img src="https://github.com/cabooster/UDMT/blob/main/images/udmt_background.png?raw=true" width="1000" align="middle" /></center>


### Unsupervised Learning

Recently, unsupervised learning shows great potential to eliminate the reliance on human annotation or ground-truth labels by constructing supervisory relationships directly from data, instead of relying on external labels. Latest research has demonstrated that unsupervised learning can perform better than supervised methods when applied to the test dataset, providing a feasible methodology for achieving higher accuracy with minimal annotation costs. Moreover, theory and practice have shown that unsupervised learning can eliminate annotation bias inherent in supervised methods, which is caused by human variability and mistakes, or by insufficient labeling diversity that fails to represent the entire dataset. Despite these expected advantages, the benefits of unsupervised learning have not yet been realized in multi-animal tracking, and comprehensive efforts are still needed to achieve high-accuracy tracking without requiring human annotation.


### Our Contribution

Here, we present an **unsupervised deep-learning method for multi-animal tracking (UDMT)** that outperforms existing tracking methods. UDMT does not require any human annotations for training. The only thing users need to do is to click the animals in the first frame to specify the individuals they want to track. UDMT is grounded in a bidirectional closed-loop tracking strategy that visual tracking can be conducted equivalently in both forward and backward directions. The network is trained in a completely unsupervised way by optimizing the network parameters to make the forward tracking and the backward tracking consistent. To better capture the spatiotemporal evolution of animal features more effectively, we incorporated a spatiotemporal transformer network (ST-Net) to utilize self-attention and cross-attention mechanisms for feature extraction, leading to a threefold reduction in IDSW compared with convolutional neural networks (CNNs). For identity correction, we designed a sophisticated module based on bidirectional tracking to relocate missing targets caused by crowding and occlusion, achieving a 2.7-fold improvement in tracking accuracy. We demonstrate the state-of-the-art performance of UDMT on five different kinds of model animals, including mice, rats, Drosophila, C. elegans, and Betta splendens. Combined with a head-mounted miniaturized microscope, we recorded the calcium transients synchronized with mouse locomotion to decipher the correlations between animal locomotion and neural activity. We have released the Python source code and a user-friendly GUI of UDMT to make it an easily accessible tool for quantitative ethology and neuroethology.


## Results

<center><h3>1. Qualitative evaluation of UDMT in various demanding recording conditions.  </h3></center>

<center><img src="https://github.com/cabooster/UDMT/blob/main/images/udmt_result1.png?raw=true" width="850" align="middle"></center>
<i>Numbers near the boxes represent the IOU, which is the intersection over union between ground-truth bounding box and the predicted bounding box of target animals.</i>
<center><h3>2. Comparative analysis of tracking accuracy across different variables</h3></center>

<center><img src="https://github.com/cabooster/UDMT/blob/main/images/udmt_result2.png?raw=true" width="850" align="middle"></center>
<i>The 5-mouse dataset (33 Hz frame rate, 18,000 frames, N=5) was used in this experiment for quantitative evaluation.</i>

<center><h3>3.Visualization of animal localization accuracy through tracking snapshots</h3></center>

<center><img src="https://github.com/cabooster/UDMT/blob/main/images/udmt_result3.png?raw=true" width="850" align="middle"></center>
<i>Example frames and magnified views of a 7-mouse video at three different time points. Tracking results and corresponding IOU metrics of UDMT, DLC and SLEAP are shown. Scale bars, 50 mm for all images.</i>
<br>
More demo images and videos are demonstrated in <a href='https://cabooster.github.io/UDMT/Gallery/'>Gallery</a>. More details please refer to <a href='https://www.nature.com/articles/s41587-022-01450-8'>the companion paper</a>.


## Citation

If you use this code please cite the companion paper where the original method appeared: 

- Xinyang Li, Yixin Li, Yiliang Zhou, et al. Real-time denoising enables high-sensitivity fluorescence time-lapse imaging beyond the shot-noise limit. Nat Biotechnol (2022). [https://doi.org/10.1038/s41587-022-01450-8](https://www.nature.com/articles/s41587-022-01450-8)




