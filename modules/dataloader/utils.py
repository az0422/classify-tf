import os

import cv2
import numpy as np
import threading

def resize(image, target_size=640):
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

def load_filelist(image):
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
    
    images_none_test = [False for _ in images_test]
    threads = []
    for i, image in enumerate(images_test):
        threads.append(TestImage(image, images_none_test, i))
        threads[-1].start()
    
    for thread in threads:
        thread.join()
    
    for index, image in enumerate(images_none_test):
        if not image: continue
        images.append(images_test[index])
        labels.append(labels_test[index])
    
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
    def __init__(self, image, arr, image_size, index):
        super().__init__()
        self.image = image
        self.arr = arr
        self.image_size = image_size
        self.index = index
    
    def run(self):
        if type(self.image) in (bytes, bytearray):
            image_np = np.frombuffer(self.image, np.uint8)
            self.arr[self.index] = resize(cv2.imdecode(image_np, cv2.IMREAD_COLOR), self.image_size).astype(np.float32) / 255.
        elif type(self.image) is str:
            self.arr[self.index] = resize(cv2.imread(self.image, cv2.IMREAD_COLOR), self.image_size).astype(np.float32) / 255.
        elif type(self.image) is np.ndarray:
            self.arr[self.index] = resize(self.image, self.image_size).astype(np.float32) / 255.