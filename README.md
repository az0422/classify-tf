This project is alpha version!

# Image Classifier using TensorFlow
## How to Use
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
 * option.yaml
 - Global user options.
 - Can copy to create indivisual options for training with `option` argument.
