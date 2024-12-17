# Scripts for Datasets
## Resize
Resize the all image in dataset.
Run with `python3 resize.py path=<path/to/dataset> export=<path/to/export> image_size=<image size ex: 224>`

## Split
Split train-validation dataset.
Run with `python3 split.py path=<path/to/dataset/to/split> ratio=<validation ratio ex: 0.2>`

## Decode
Convert image format to NumPy file.
Run with `python3 decode.py path=<path/to/dataset> export=<path/to/export>`