# Visualizing the Smart Environment in AR: An Approach Based on Visual Geometry Matching

<div align='center'>
 Ming Xia, Min Huang, Qiuqi Pan, Yunhan Wang, Xiaoyan Wang, and Kaikai Chi


 <strong><a>Zhejiang University of Technology</a>, <a>University of Wisconsin-Madison</a> </strong>
</div>

## Introduction

This is the repository for our paper "Visualizing the Smart Environment in AR: An Approach Based on Visual Geometry Matching". This repository provides the instructions for setting up the environment, training the model, and testing the model to reproduce the results reported in the paper.

## Dependencies

Create the environment by following the steps below:

```shell
git clone https://github.com/overlappredator/OverlapPredator.git;
conda create --name Insight;
conda activate Insight;
cd Insight; pip install -r requirements.txt
cd cpp_wrappers; sh compile_wrappers.sh; cd ..
```

## Training

```shell
bash train
```

To change the weights, modify the pretrain field in the configs/train/indoor.yaml file.

## Testing

First run:

```shell
bash clear_last_test_data
```

Then run:

```shell
bash eve
```

To change the weights, modify the pretrain field in the configs/test/indoor.yaml file. The result files are saved under the snapshot/indoor/est_traj directory.

## Credits

If you find this work useful, please consider citing:

```
@ARTICLE{10763439,
author={Xia, Ming and Huang, Min and Pan, Qiuqi and Wang, Yunhan and Wang, Xiaoyan and Chi, Kaikai},
journal={IEEE Transactions on Mobile Computing}, 
title={Visualizing the Smart Environment in AR: An Approach Based on Visual Geometry Matching}, 
year={2025},
volume={24},
number={4},
pages={2900-2916},
doi={10.1109/TMC.2024.3504960}}
```
