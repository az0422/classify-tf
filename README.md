This project is a sandbox project! This projects will be used for experiments!

# Image Classifier using TensorFlow
## How to Use
### Installation
1. This project can be run only linux environment.
2. This project is recommended to run on virtual environment include docker.
3. Command `pip install -r requirements.txt` to install dependency.

### Commands
* Train: `python3 train.py option=<configuration file (optional)>`
  - For obtaining weights.
  - The `option` argument for indivisual training options.

* Plot graph: `python3 plot.py path=path/to/checkpoint_path/checkpoint_name`
  - For plotting graphs on training.
  - The graphs will be saved onto `path/to/checkpoint_path/checkpoint_name` with `plot.png`.

* Export: `python3 export.py path=path/to/checkpoint_path/checkpoint_name epoch=[best|last|<an epoch number>] image_size=<image_size>`
  - Export weights to saved_model and TFLite
  - The weights files will be saved onto `path/to/checkpoint_path/checkpoint_name/export`.

* Select specific GPU (example): `CUDA_VISIBLE_DEVICES=[cpu|<GPU numbers (example: 0,1,2,3)>] python3 train.py option=example.yaml`
  - The default is use all GPUs.
  - If you have multiple GPUs and want to select specific GPU, you should use this command form.

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
  - Color space adjustment
  - Other dataloader options
* Train options
  - Checkpoint path
  - Training options (optimizer, loss, etc.)
  - Learning rate scheduler options
  - Model option

## Model Configurations
* ResNet Models
  - `resnet/18.yaml`
  - `resnet/24.yaml`
  - `resnet/33.yaml`
  - `resnet/54.yaml`
  - `resnet/78.yaml`
  - `resnet/102.yaml`
  - `resnet/150.yaml`

* CSPResNet Models
  - `cspresnet/18.yaml`
  - `cspresnet/24.yaml`
  - `cspresnet/33.yaml`
  - `cspresnet/54.yaml`
  - `cspresnet/78.yaml`
  - `cspresnet/102.yaml`
  - `cspresnet/150.yaml`

## Experiment
### Experimental Environment
* CPU: AMD Ryzen 7900 @ 4.5GHz; 90W
* Memory: DDR5 64GB
* GPU: RTX 4060 Ti 16GB @ 2.5GHz; 125W x2
* OS: Ubuntu 24.04 (docker image: nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04)
* Python: CPython 3.10

### Model Performance
Experimenting now...
