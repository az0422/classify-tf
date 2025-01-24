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

* Train resume: `python3 train.py resume=path/to/checkpoint_path/checkpotin_name`
  - Used to resume training.

* Predict: `python3 classify.py weights=path/to/checkpoint_path/checkpoint_name image=path/to/image(s/dir) epoch=[best|last|<an epoch number>](optional) output_format=[csv|stdout](optional)`

* Plot graph: `python3 plot.py path=path/to/checkpoint_path/checkpoint_name`
  - Used to plot graph of train result.
  - The graphs will be saved onto `path/to/checkpoint_path/checkpoint_name` with `plot.png`.

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
  - `resnet/54.yaml`
  - `resnet/102.yaml`
  - `resnet/150.yaml`
  - `resnet2l/15.yaml`
  - `resnet2l/54.yaml`
  - `resnet2l/110.yaml`
  - `resnet2l/166.yaml`

* CSPResNet Models
  - `cspresnet/18.yaml`
  - `cspresnet/54.yaml`
  - `cspresnet/102.yaml`
  - `cspresnet/150.yaml`
  - `cspresnet2c/18.yaml`
  - `cspresnet2c/54.yaml`
  - `cspresnet2c/102.yaml`
  - `cspresnet2c/150.yaml`
  - `cspresnet2l2c/15.yaml`
  - `cspresnet2l2c/54.yaml`
  - `cspresnet2l2c/110.yaml`
  - `cspresnet2l2c/166.yaml`
  - `cspresnet2l3c/15.yaml`
  - `cspresnet2l3c/54.yaml`
  - `cspresnet2l3c/110.yaml`
  - `cspresnet2l3c/166.yaml`

* Other Models
  - `vgg16.yaml`
  - `googlenet.yaml`

* Details
  - The ResNet models are implemented of 3-layers ResNet blocks with bottleneck.
  - The ResNet2L models are implemented of 2-layers ResNet blocks without bottleneck.
  - The CSPResNet and CSPResNet2L3C models are implemented of 3-Conv layers to implementing CSPNet structure.
  - The CSPResNet2C and CSPResNet2L2C models are implemented of 2-Conv layers to implementing CSPNet structure.

## Experiment
### Experimental Environment
* CPU: AMD Ryzen 7900 @ 4.5GHz; 90W
* Memory: DDR5 64GB
* GPU: RTX 4060 Ti 16GB @ 2.5GHz; 130W x2
* OS: Ubuntu 24.04 (docker image: nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04)
* Python: CPython 3.10
* TensorFlow version: 2.17.0

### Model Performance
Experimenting now...
