import time
import random
import math

from .utils import load_filelist
from .augment import DataAugment

class DataLoader():
    def __init__(self, images: str, cfg: dict, augment_flag=True):
        super().__init__()
        random.seed(time.time())

        self.cfg = cfg
        self.augment_flag = augment_flag
        images, labels, classes_name = load_filelist(images, self.cfg["file_checkers"])
        self.classes_name = classes_name
        self.classes = len(classes_name)

        self.images = list(zip(images, labels))
        self.data_length = math.ceil(len(images) / self.cfg["batch_size"]) if not self.augment_flag or self.cfg["data_length"] is None else self.cfg["data_length"]

        self.augment_rotate = 0

        if augment_flag:
            self.augments = [
                DataAugment(
                    random.randrange(-2**31, 2**31 - 1),
                    self.images,
                    self.classes,
                    cfg,
                    True,
                    classes_name,
                ) for _ in range(self.cfg["loaders"])
            ]

        else:
            self.augments = [
                DataAugment(
                    random.randrange(-2**31, 2**31 - 1),
                    self.images,
                    self.classes,
                    cfg,
                    False,
                    classes_name
                ) for _ in range(self.cfg["loaders"])
            ]
    
    def startAugment(self):
        for augment in self.augments:
            augment.start()
    
    def stopAugment(self):
        for augment in self.augments:
            augment.terminate()

    def __len__(self):
        return self.data_length
    
    def __getitem__(self, index=0):
        images, labels = self.augments[self.augment_rotate].getData()
        self.augment_rotate = (self.augment_rotate + 1) % len(self.augments)
        
        return images, labels