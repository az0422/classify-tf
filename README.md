This project is a sandbox project! This projects will be used for experiments!

# Image Classifier using TensorFlow
## How to Use
### Installation
1. This project can be run only linux environment.
2. This project is recommended to run on virtual environment include docker.
3. Command `pip install -r requirements.txt` to install dependency.

### Commands
* Train: `python3 train.py option=<configuration file (optional)>`
  - Used to obtain weights.
  - The `option` argument for indivisual training options.
  
* Predict: `python3 classify.py weights=path/to/checkpoint_path/checkpoint_name image=path/to/image(s/dir) epoch=[best|last|<an epoch number>](optional) output_format=[csv|stdout](optional)`

* Export: `python3 export.py path=path/to/checkpoint_path/checkpoint_name epoch=[best|last|<an epoch number>] image_size=<image_size>`
  - Export weights to saved_model and TFLite
  - The weights files will be saved onto `path/to/checkpoint_path/checkpoint_name/export`.

* Select specific GPU (example): `CUDA_VISIBLE_DEVICES=[cpu|<GPU numbers (example: 0,1,2,3)>] python3 train.py option=example.yaml`
  - The default uses all GPUs.
  - If you have multiple GPUs and want to select specific GPU, you have to use this command form or set the `CUDA_VISIBLE_DEVICES` variable.

### Options
#### Option application order
1. `cfg/settings.yaml` (default settings)
2. `option.yaml` (user global settings)
3. configuration file (copied from `option.yaml`)

#### option.yaml
* Dataloader options
  - Image crop options
  - Image translate options
  - Color adjustment options
  - Quality adjustment options
  - Other dataloader options
* Train options
  - Checkpoint path
  - Training options (optimizer, loss, etc.)
  - Learning rate scheduler options
  - Model option

## Model Configurations
* ResNet Models
  - `resnet/18.yaml`
  - `resnet/34.yaml`
  - `resnet/50.yaml`
  - `resnet/110.yaml`
  - `resnet/152.yaml`

* Other Models
  - `vgg16.yaml`
  - `googlenet.yaml`

## Experiment
### Experimental Environment
* CPU: AMD Ryzen 7900
* Memory: DDR5 128GB
* GPU: RTX 5060 Ti 16GB @ 2.0GHz x2
* OS: Ubuntu 24.04 (docker image: nvidia/cuda:12.8.1-cudnn8-devel-ubuntu24.04)
* Python: CPython 3.12
* TensorFlow version: 2.20.0

### Experimetnal Results
| Model     | Accuracy | Params | FLOPs |
|-----------|----------|--------|-------|
| ResNet18  | 67.15%   | 11.70M | 3.63G |
| ResNet34  | 69.34%   | 21.81M | 7.34G |
| ResNet50  | 72.36%   | 26.73M | 8.64G |
| ResNet101 | 72.76%   | 44.65M | 15.64G |
| ResNet152 | 73.17%   | 60.34M | 23.08G |
