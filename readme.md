# Neural Polarizer: A Lightweight and Effective Backdoor Defense via Purifying Poisoned Features

This is the official MindSpore implementation of [Neural Polarizer: A Lightweight and Effective Backdoor Defense via Purifying Poisoned Features](https://openreview.net/pdf?id=VFhN15Vlkj)

## Requirements
see file `mindspore.yaml`

## Datasets and state-of-the-art backdoor attack and defense methods.
We test our method on CIFAR-10, Tiny ImageNet and GTSRB datasets.
For CIFAR-10, the dataset will be download automatically. We follow [BackdoorBench](https://github.com/SCLBD/BackdoorBench) on the implementation of SOTA attack and defense methods.

## Running the code
Before run the defense method, a backdoored model should be generated first. We provide the script for defense on CIFAR-10 dataset.

### Step 1 Prepare a poisoned dataset.
    python attack/data_poison.py
### Step 2 Train a backdoored model
    python attack/train_backdoor.py

### Step 3 Run the defense
    python defense/npd.py

If you use this paper/code in your research, please consider citing us:

```
@inproceedings{
anonymous2023neural,
title={Neural Polarizer: A Lightweight and Effective Backdoor Defense via Purifying Poisoned Features},
author={Anonymous},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=VFhN15Vlkj}
}
```

## Acknowledgment
Our project references the codes in the following repos.
- [BackdoorBench](https://github.com/SCLBD/BackdoorBench)

