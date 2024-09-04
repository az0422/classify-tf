import os
import yaml

import cv2
import numpy as np
import threading

def parse_cfg(cfg):
    cfg_default = yaml.full_load(open(cfg, "r"))
    if cfg_default["user_option"] is None or not os.path.isdir(cfg_default["user_option"]):
        return cfg_default
    
    user_option = yaml.full_load(open(cfg_default["user_option"], "r"))
    
    for key in user_option.keys():
        if user_option[key] is None: continue
        cfg_default[key] = user_option[key]
    
    return cfg_default

def resize_stretch(image, target_size=640):
    return cv2.resize(image, (target_size, target_size))

def resize_contain(image, target_size=640):
    height, width, _ = image.shape

    scale = target_size / max(height, width)
    scaled_width, scaled_height = int(width * scale), int(height * scale)

    image = cv2.resize(image, (scaled_width, scaled_height))

    pad_width = target_size - scaled_width
    pad_height = target_size - scaled_height

    top, bottom = pad_height // 2, pad_height - pad_height // 2
    left, right = pad_width // 2, pad_width - pad_width // 2

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return image

def checker_log(ok, none):
    print("checking files... ok: %-16d\terror: %-16d" % (ok, none), end="\r")

def load_filelist(image, loaders):
    images = []
    labels = []

    images_test = []
    labels_test = []

    categories = sorted(os.listdir(image))

    for id, category in enumerate(categories):
        category_path = os.path.join(image, category)
        if os.path.isfile(category_path): continue

        files = os.listdir(category_path)
        for file in files:
            file_path = os.path.join(image, category, file)
            if os.path.isdir(file_path): continue
            images_test.append(file_path)
            labels_test.append(id)
    
    none = 0
    ok = 0
    
    for mass in range(0, len(images_test), loaders):
        threads = []
        images_none_test = [-1 for _ in range(loaders)]
        for index, image in enumerate(images_test[mass:mass + loaders]):
            threads.append(TestImage(image, images_none_test, index))
            threads[-1].start()
        
        for thread in threads:
            thread.join()
        
        for index, image in enumerate(images_none_test):
            if image == -1: continue
            if not image:
                none += 1
                checker_log(ok, none)
                continue
            images.append(images_test[mass + index])
            labels.append(labels_test[mass + index])
            ok += 1
            checker_log(ok, none)
    
    print()
    return images, labels, categories

class TestImage(threading.Thread):
    def __init__(self, image, arr, index):
        super().__init__()
        self.image = image
        self.arr = arr
        self.index = index
    
    def run(self):
        if type(self.image) in (bytes, bytearray):
            image_np = np.frombuffer(self.image, np.uint8)
            test = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            self.arr[self.index] = test is not None
        elif type(self.image) is str:
            test = cv2.imread(self.image, cv2.IMREAD_COLOR)
            self.arr[self.index] = test is not None
        elif type(self.image) is np.ndarray:
            test = cv2.imread(self.image, cv2.IMREAD_COLOR)
            self.arr[self.index] = test is not None

class LoadImage(threading.Thread):
    def __init__(self, image, arr, image_size, index, resize_method, dtype):
        super().__init__()
        self.image = image
        self.arr = arr
        self.image_size = image_size
        self.index = index
        self.resize_method = resize_method
        self.dtype = dtype
    
    def _resize(self, image):
        if self.resize_method in ("default", "contain"):
            return resize_contain(image, self.image_size)
        if self.resize_method in ("stretch",):
            return resize_stretch(image, self.image_size)
        raise Exception("invalid resize method %s" % self.resize_method)
    
    def run(self):
        if type(self.image) in (bytes, bytearray):
            image_np = np.frombuffer(self.image, np.uint8)
            self.arr[self.index] = self._resize(cv2.imdecode(image_np, cv2.IMREAD_COLOR)).astype(self.dtype)
        elif type(self.image) is str:
            self.arr[self.index] = self._resize(cv2.imread(self.image, cv2.IMREAD_COLOR)).astype(self.dtype)
        elif type(self.image) is np.ndarray:
            self.arr[self.index] = self._resize(self.image).astype(self.dtype)