# LFASM_pytroch
---
### Introduction

- Implementation of LFASM

### Requirements

- Linux
- CUDA 8.0 or higher
- Python3
- Pytorch 1.0+

### Install LFASM_pytorch

1.Clone the LFASM_pytorch repository

```python
git clone https://github.com/WangHonglie/LFASM_pytroch
```

### Get Started

---

1.Download the dataset([VeRi776](https://github.com/VehicleReId/VeRidataset)/[VehicleID](https://medusa.fit.vutbr.cz/traffic/datasets/)/[PKU_VD](https://pkuml.org/resources/pku-vehicleid.html))

2.Train

```
python train.py --gpu_ids <gpu_ids> \\
				--name <model name> \\
				--train_data_root <path_to_train_data>\\
				--val_data_root <path_to_val_data>\\
				--batchsize <batchsize>
```

​	3.Test 

```
python test.py --gpu_ids <gpu_ids> \\
			   --name <model name> \\
			   --which_epoch <select the i-th model>\\
			   --test_dir <path_to_testing_data>
```

4.Evaluation

```
python evaluate_gpu.py
```
It will output Rank@1,Rank@5,Rand@10 and mAP
5.Complexity calculation
```
python flops.py
```
It will output the number of parameters and print per-layer computational cost of a given network.



### Acknowlegement

---

- Some code is provided by [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch)
- Complexity calculation by [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch)
