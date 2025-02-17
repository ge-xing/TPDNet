# TPDNet
Temporal Prompt Learning with Depth Memory for Video Mirror Detection

## Trained model and predicted results

We provide the trained model and predicted results on VMD-D dataset in [here](https://hkustgz-my.sharepoint.com/:f:/g/personal/zxing565_connect_hkust-gz_edu_cn/Epn74ueGmP9LvIl8jQ1F63gBRWjwvQqzbrX_-Vc7d5CJiA?e=XHHeih).


## 1. Data Download

The dataset using in this project is VMD-D dataset, which can be downloaded from [here](https://jiaying.link/cvpr2023-vmd/).

The downloaded data should be put in the `./VMD` folder.

## 2. Train

The training script is `train.py`, where we set env to `pytorch` and num_gpus to 1.

If you want to train the model on multiple GPUs, you can set env to `DDP` and num_gpus to the number of GPUs you want to use.


## 3. Predict results

The predict script is `validate.py`.
In this script, we load the trained model (`final_model_0.6581.pt`) and predict the results on the test set.
The predicted results will be saved at `./results/tqdm`.


## 4. Evaluation

After running the predict script, we can evaluate the results by running the `eval.py` script.

For example:

```bash 
python eval.py --pred=./results/tqdm

```