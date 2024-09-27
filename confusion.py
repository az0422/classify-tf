import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from modules.nn.model import ClassifyModel
from modules.dataloader import DataLoader
from modules.utils import parse_cfg

from sklearn.metrics import confusion_matrix

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, False)

model = ClassifyModel("runs/classify-medium/model.yaml", 1000, 224)
model.load_weights("runs/classify-medium/weights/best.weights.h5")
cfg = parse_cfg("runs/classify-medium/cfg.yaml")
dataloader = DataLoader("../datasets/imagenet/val", cfg, False)
dataloader.startAugment()

labels = []
preds = []

for _ in range(dataloader.__len__()):
    image, label = dataloader.__getitem__()
    pred = model.predict(image, batch_size=256)
    pred = np.argmax(pred, axis=-1)
    label = np.argmax(label, axis=-1)

    print(pred.shape, label.shape)

    labels.append(label.reshape(-1, 1))
    preds.append(pred.reshape(-1, 1))

labels = np.vstack(labels)
preds = np.vstack(preds)

print(preds.shape, labels.shape)

confusion = confusion_matrix(labels, preds)

def plot_confusion_matrix(con_mat, labels, title='Confusion Matrix', cmap=plt.get_cmap("Blues"), normalize=True):
    plt.figure(figsize=(16, 16))
    plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    marks = np.arange(len(labels))
    nlabels = []
    for k in range(len(con_mat)):
        n = sum(con_mat[k])
        nlabel = '{0}(n={1})'.format(labels[k],n)
        nlabels.append(nlabel)
    plt.xticks(marks, labels)
    plt.yticks(marks, nlabels)

    thresh = con_mat.max() / 2.
    if normalize:
        for i, j in zip(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, ' ', horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
    else:
        for i, j in zip(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, con_mat[i, j], horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion.png")

plot_confusion_matrix(confusion, range(1000))

dataloader.stopAugment()