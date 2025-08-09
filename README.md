# Official Code for "[Elucidating the Design Choice of Probability Paths in Flow Matching for Forecasting](https://openreview.net/forum?id=JApMDLwbLR)" (TMLR 2025)

This repository contains the official implementation of the methods and experiments described in the TMLR paper. The arxiv version is available [here](https://arxiv.org/abs/2410.03229).

### **Setup**

First, build the container image from `ours.def` (installs packages from `requirements.txt`):
```
apptainer build ours.sif ours.def
```

### **Running Examples**
You can then run the provided simple examples inside the container:

```
apptainer exec ours.sif python train_ae.py \
  --run-name github_simpleflow_ae \
  --dataset simpleflow \
  --ae_option ae \
  --ae_epochs 2000 \
  --snapshots-per-sample 25 \
  --ae_lr_scheduler cosine \
  --ae_learning_rate 0.001
```

```
apptainer exec ours.sif python train.py \
  --run-name github_simpleflow_ours_sigma0.1_samplingsteps10_rk4_separate \
  --dataset simpleflow \
  --train_option separate \
  --probpath_option ours \
  --epochs 2000 \
  --sampling_steps 10 \
  --sigma 0.01 \
  --solver rk4 \
  --snapshots-per-sample 25 \
  --snapshots-to-generate 20 \
  --path_to_ae_checkpoints checkpoints/ \
  --path_to_results results/
```

### Access to Datasets 
The dataset for the simple fluid flow task is included in this repository. The other datasets are provided by [PDEBench](https://github.com/pdebench/PDEBench) and can be downloaded from their official repository.

### Reproducibility Disclaimer
The quantitative results in the paper may not be precisely reproducible because the packages listed in `requirements.txt` install the latest versions, whereas the experiments in the paper were conducted using older versions of some dependencies. This can cause minor variations in results. For more consistent reproduction, we recommend using the specific package versions listed in `requirements_exact.txt`.

### **Citation**
If you find our work useful for your research, please consider citing our paper:
```
@article{lim2025elucidating,
  title={Elucidating the design choice of probability paths in flow matching for forecasting},
  author={Lim, Soon Hoe and Wang, Yijin and Yu, Annan and Hart, Emma and Mahoney, Michael W and Li, Xiaoye S and Erichson, N Benjamin},
  journal={Transaction on Machine Learning Research},
  year={2025}
}
```
