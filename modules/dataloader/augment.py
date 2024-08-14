import cv2
import numpy as np
import multiprocessing
import random

from .utils import LoadImage

class DataAugment(multiprocessing.Process):
    def __init__(self,
                 seed=0,
                 flip_vertical=0.5,
                 flip_horizontal=0.5,
                 rotate_degree=0,
                 zoom=1,
                 translate_vertical=0,
                 translate_horizontal=0,
                 hsv_h=1,
                 hsv_s=1,
                 hsv_v=1,
                 noise=0,
    ):
        super().__init__()
        np.random.seed(seed)
        random.seed(seed)

        self.flip_vertical = flip_vertical
        self.flip_horizontal = flip_horizontal
        self.rotate_degree = rotate_degree
        self.zoom = zoom
        self.translate_vertical = translate_vertical
        self.translate_horizontal = translate_horizontal
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.noise = noise

        self.originals = None
        self.results = None
        self.lock = None
        self.batch_size = 16
        self.image_size = 320
    
    def _flip(self, image):
        if np.random.rand() < self.flip_vertical:
            image = image[::-1]
        
        if np.random.rand() < self.flip_horizontal:
            image = image[:, ::-1]
        
        return image
    
    def _translate(self, image):
        height, width, _ = image.shape
        vertical = (np.random.rand() * self.translate_vertical * 2 - self.translate_vertical) * height
        horizontal = (np.random.rand() * self.translate_horizontal * 2 - self.translate_horizontal) * width
        degree = np.random.rand() * self.rotate_degree * 2 - self.rotate_degree

        zoom = np.random.rand() * (self.zoom - 1) + 1
        if np.random.rand() > 0.5: zoom = 1 / zoom

        matrix = cv2.getRotationMatrix2D((width // 2, height // 2), degree, zoom)
        matrix[:2, 2] += [horizontal, vertical]

        image = cv2.warpAffine(image, matrix, [width, height])

        return image
    
    def _hsv(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        h = np.random.rand() * (self.hsv_h - 1) + 1
        s = np.random.rand() * (self.hsv_s - 1) + 1
        v = np.random.rand() * (self.hsv_v - 1) + 1
 
        image[..., :] *= [h, s, v]
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        return image
    
    def _noise(self, image):
        noise = np.random.rand(*image.shape) * self.noise
        image += noise

        return image
    
    def _clip(self, image):
        over = image > 255.
        under = image < 0.

        image[over] = 255.
        image[under] = 0.

        return image
    
    def data(self, images, results, lock, batch_size=16, image_size=320):
        self.images = images
        self.results = results
        self.lock = lock
        self.batch_size = batch_size
        self.image_size = image_size
    
    def run(self):
        assert self.images is not None
        assert self.results is not None
        assert self.lock is not None

        image_taked = [None for _ in range(self.batch_size)]
        label_taked = [None for _ in range(self.batch_size)]
        none_index = list(range(self.batch_size))

        none_flag = True

        while none_flag:
            threads = []
            for index in none_index:
                take_index = random.randint(0, len(self.images) - 1)
                image, label = self.images[take_index]

                threads.append(LoadImage(image, image_taked, self.image_size, index))
                threads[-1].start()

                label_taked[index] = label
            
            for thread in threads:
                thread.join()
            
            none_index = []
            none_flag = False
            for i, image in enumerate(image_taked):
                if image is not None: continue
                none_index.append(i)
                none_flag = True
        
        images = np.zeros([self.batch_size, *image_taked[0].shape], dtype=np.float16)
        labels = np.zeros([self.batch_size], dtype=np.float16)
        
        for index, (image, label) in enumerate(zip(image_taked, label_taked)):
            image = self._flip(image)
            image = self._translate(image)
            image = self._hsv(image)
            image = self._noise(image)
            image = self._clip(image)

            images[index] = image.astype(np.float16)
            labels[index] = label
        
        try:
            self.lock.acquire()
            self.results.put([images, labels])
            self.lock.release()
        except:
            self.join()
            self.close()