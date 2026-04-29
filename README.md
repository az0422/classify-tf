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
  - The `option` means argument for indivisual training options.

* Resume: `python3 train.py resume=<checkpoint path>`
  - The &lt;checkpoint path&gt; means a path of `checkpoint_path` and `checkpoint_name` in `option.yaml`
  
* Predict: `python3 classify.py weights=path/to/checkpoint_path/checkpoint_name image=path/to/image(s/dir) epoch=[best|last|<an epoch number>](optional) output_format=[csv|stdout](optional)`

* Export: `python3 export.py path=path/to/checkpoint_path/checkpoint_name epoch=[best|last|<an epoch number>] image_size=<image_size>`
  - Export weights to saved_model and TFLite
  - The weights files will be saved onto `path/to/checkpoint_path/checkpoint_name/export`.

* Select specific GPU (example): `CUDA_VISIBLE_DEVICES=[cpu|<GPU numbers (example: 0,1,2,3)>] python3 train.py option=example.yaml`
  - The default uses all GPUs.
  - If you have multiple GPUs and want to select specific GPU, you have to use this command form or set the `CUDA_VISIBLE_DEVICES` variable.

### Dataset File Structure
```
dataset/
 ├ train/
 │   ├ class1/
 |   |   └ image1.xxx
 │   └ class2/
 |       └ image2.xxx
 └ val/
     ├ class1/
     |   └ image3.xxx
     └ class2/
         └ image4.xxx
```

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

* CSPResNetS models
  - `cspresnets/18.yaml`
  - `cspresnets/34.yaml`
  - `cspresnets/50.yaml`
  - `cspresnets/101.yaml`
  - `cspresnets/152.yaml`

* CSPResNetP models
  - `cspresnetp/18.yaml`
  - `cspresnetp/34.yaml`
  - `cspresnetp/50.yaml`
  - `cspresnetp/101.yaml`
  - `cspresnetp/152.yaml`

* DenseNet Models
  - `densenet/121.yaml`
  - `densenet/169.yaml`
  - `densenet/201.yaml`
  - `densenet/264.yaml`

* S/P suffix of CSPResNet is meaning of implementation method. S is split by channels, P is projection by channales

* Other Models
  - `vgg16.yaml`
  - `googlenet.yaml`

## Experiment
### Experimental Environment
* CPU: AMD Ryzen 7900
* Memory: DDR5 128GB
* GPU: RTX 5060 Ti 16GB @ 2.0GHz x2
* OS: Ubuntu 26.04 (docker image: nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04)
* Python: CPython 3.12
* TensorFlow version: 2.21.0

### Experimetnal Results
| Model     | Accuracy | Params | FLOPs |
|-----------|----------|--------|-------|
| ResNet18  | 67.15%   | 11.70M | 3.63G |
| ResNet34  | 69.34%   | 21.81M | 7.34G |
| ResNet50  | 72.36%   | 26.73M | 8.64G |
| ResNet101 | 72.76%   | 44.65M | 15.64G |
| ResNet152 | 73.17%   | 60.34M | 23.08G |
|                                        |
| CSPResNetS18 | 68.11%  | 7.64M | 2.10G |
| CSPResNetS34 | 69.98%  | 10.18M | 3.03G |
| CSPResNetS50 | 70.31%  | 25.33M | 7.00G |
| CSPResNetS101 | 73.84%  | 29.83M | 8.76G |
| CSPResNetS152 | 72.11%  | 33.78M | 10.62G |
|                                          |
| CSPResNetP18 | 68.16%  | 7.64M | 2.10G |
| CSPResNetP34 | 69.85%  | 10.18M | 3.03G |
| CSPResNetP50 | 71.03%  | 25.33M | 7.00G |
| CSPResNetP101 | 71.59%  | 29.83M | 8.76G |
| CSPResNetP152 | 71.91%  | 33.78M | 10.62G |

Trained and evaluated in ImageNet2012 dataset.
