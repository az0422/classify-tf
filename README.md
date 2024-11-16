This project is alpha version!

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

### Configurations
#### Apply Sequence
1. cfg/settings.yaml (default options)
2. option.yaml (global user options)
3. configuration file (copied from option.yaml)

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
