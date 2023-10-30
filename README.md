# LDCNet
the code for LDCNet
## Note
we propose a LDCNet(lightweight dynamic convolution Network) that has the same superior segmentation performance as the transformer network while running at the speed of a careful convolutional neural network.

## Model zoo
- kvasir dataset pretrain
[google](https://drive.google.com/file/d/174CPEQCv_dEkfnMoXa7oLnrwh-Yh80XY/view?usp=drive_link)
- Tatme dataset pretrain
[google](https://drive.google.com/file/d/1_msRG2X7XnMXSO9YHmIJlPiV0byzFhId/view?usp=drive_link)
## Install
- Clone this repo:
```bash
git clone git@github.com:yinyiyang416/LDCNet.git
```

- Create a conda virtual environment and activate it:

```bash
conda create -n LDCNet python=3.8
conda activate LDCNet
```

- Install `CUDA==11.6` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

- Install `PyTorch==1.13.0` with `CUDA==11.6`:

```bash
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
```
- The python packages we use are placed in the requirements.txt
```bash
pip install requirements.txt
```

- Build the ddf operation for DDMF module:

```bash
cd ddf
python setup.py install
mv build/lib*/* .
```

## Dataset
Please download dataset and put it into dataset, and change the DATAPATH in config
- https://drive.google.com/file/d/1lLLQCs7-Hpbf0yFEXmkI7KyYPWqJbsyb/view?usp=drive_link

## Training

To train a model, check the config in [train config](configs\train_config.py) and change the img_dataset_root (cuda id), then run:

```bash
python Train.py
```

## Evaluation

To evaluate a pre-trained model,check the config in [test config](configs\test_config.py) and change the img_dataset_root and model_path(cuda id), then run:

```bash
python Test.py
```
if you want to eval more, you can see the eval dir.

## Acknowledgement

this work is bulit from [ddfnet](https://github.com/thefoxofsky/ddfnet).



