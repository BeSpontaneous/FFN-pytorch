# 2D Network for Video Recognition

## Requirements
- python 3.7
- torch 1.11.0
- torchvision 0.12.0

## Implementation Details
1. We uniformly sample 4/8/16 frames for `num_segments_L`, `num_segments_M` and `num_segments_H` during training, and use `num_segments_H` to specify the number of frames during inference. 
2. We enable Any-Frame-Inference for 2D network so that the model can be evaluated at frames which are not used in training.
3. We use 1-clip 1-crop evaluation for 2D network with the resolution of 224x224.
4. `lambda_act` denotes the coefficient $\lambda$ in the loss function and we set it as 1 without further fine-tuning the hyperparameter.
5. We train 2D network TSM, TEA with 2 NVIDIA Tesla V100 (32GB) cards and the model is pretrained on ImageNet.

## Training
1. Specify the directory of datasets with `ROOT_DATASET` in `ops/dataset_config.py`.
2. Simply run the training scripts in [exp](exp) as followed:

   ```
   bash exp/tsm_sthv1/run.sh  ## baseline training
   bash exp/tsm_sthv1_FFN/run.sh   ## FFN training
   ```

## Inference
1. Specify the directory of datasets with `ROOT_DATASET` in `ops/dataset_config.py`.
2. Please download pretrained models from [Google Drive](https://drive.google.com/drive/folders/1anktOMWzoWiZA3rvb9Tax4Y26ULoGU16?usp=sharing).
3. Specify the directory of the pretrained model with `resume` in `test.sh`.
4. Run the inference scripts in [exp](exp) as followed:

   ```
   bash exp/tsm_sthv1/test.sh  ## baseline inference
   bash exp/tsm_sthv1_FFN/test.sh   ## FFN inference
   ```