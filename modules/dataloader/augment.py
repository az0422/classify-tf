import cv2
import numpy as np
import multiprocessing
import random
import sys

from .utils import LoadImage, resize_contain, resize_stretch

class DataAugment(multiprocessing.Process):
    def __init__(
            self,
            seed=0,
            flip_vertical=0.5,
            flip_horizontal=0.5,
            rotate_degree=0,
            zoom=1,
            translate_vertical=0,
            translate_horizontal=0,
            hsv_h=0,
            hsv_s=0,
            hsv_v=0,
            noise=0,
            dequality=25,
            crop_ratio=0.5,
    ):
        super().__init__()
        np.random.seed((~seed) & 0xFFFFFFFF)
        random.seed(seed if np.random.rand() > 0.5 else seed * -1)

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
        self.dequality = dequality
        self.crop_ratio = crop_ratio

        self.images = None
        self.classes = 0
        self.store = None
        self.batch_size = 16
        self.image_size = 320
        self.resize_method = "default"
        self.dtype = "float32"
    
    def _resize(self, image):
        if self.resize_method in ("default", "contain"):
            return resize_contain(image, self.image_size)
        if self.resize_method in ("stretch"):
            return resize_stretch(image, self.image_size)
        raise Exception("invalid resize method %s" % self.resize_method)
    
    def _flip(self, image):
        if np.random.rand() < self.flip_vertical:
            image = image[::-1]
        
        if np.random.rand() < self.flip_horizontal:
            image = image[:, ::-1]
        
        return image
    
    def _crop(self, image):
        height, width, _ = image.shape

        crop_width, crop_height = ((1 - np.random.rand(2) * self.crop_ratio) * [width, height]).astype(np.int32)
        padding = [width - crop_width, height - crop_height]
        offset_x, offset_y = (np.random.rand(2) * padding).astype(np.int32)

        crop_index = np.ones([height, width], dtype=np.bool_)
        crop_index[offset_y:offset_y + crop_height, offset_x:offset_x + crop_width] = False
        image[crop_index] = 0

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

        image = cv2.warpAffine(image.astype(np.float32), matrix, [width, height])

        return image.astype(np.float32)
    
    def _hsv(self, image):
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)

        hsv = np.array([self.hsv_h, self.hsv_s, self.hsv_v], dtype=np.float32)
        hsv[hsv > 1.] = 1.
        hsv[hsv < 0.] = 0.

        hsv_mul = np.random.rand(3) * hsv * 2 - hsv + 1.
 
        image[..., :] *= hsv_mul
        image[..., 0] = image[..., 0] % 180
        image[image > 255.] = 255.
        image[image < 0.] = 0.
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

        return image
    
    def _noise(self, image):
        noise = (np.random.rand(*image.shape) * 255. * self.noise).astype(np.float32)
        image += noise

        return image
    
    def _clip(self, image):
        over = image > 255.
        under = image < 0.

        image[over] = 255.
        image[under] = 0.

        return image
    
    def _dequality(self, image):
        quality = int(100 - np.random.rand() * self.dequality)
        image_np = cv2.imencode(".jpeg", image.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR).astype(np.float32)

        return image
    
    def data(self, images, classes, store, batch_size=16, image_size=320, resize_method="default", color_space="bgr", dtype="float32"):
        color_space_dict = {"bgr": None, "rgb": cv2.COLOR_BGR2RGB, "hsv": cv2.COLOR_BGR2HSV}

        self.images = images
        self.classes = classes
        self.store = store
        self.batch_size = batch_size
        self.image_size = image_size
        self.resize_method = resize_method
        self.color_space = color_space_dict[color_space.lower()]
        self.dtype = dtype
    
    def run(self):
        assert self.images is not None
        assert self.store is not None

        while True:
            image_taked = [None for _ in range(self.batch_size)]
            label_taked = [None for _ in range(self.batch_size)]

            for index in range(self.batch_size):
                take_index = random.randint(0, len(self.images) - 1)
                image, label = self.images[take_index]
                
                image = cv2.imread(image, cv2.IMREAD_COLOR)
                
                if self.color_space is not None:
                    image = cv2.cvtColor(image, self.color_space)
                
                image_taked[index] = self._resize(image).astype(np.float32)
                label_taked[index] = label
            
            images = np.zeros(
                [
                    self.batch_size,
                    self.image_size,
                    self.image_size,
                    3,
                ], dtype=self.dtype
            )
            labels = np.zeros(
                [
                    self.batch_size,
                    self.classes,
                ], dtype=np.uint8
            )
            
            for index, (image, label) in enumerate(zip(image_taked, label_taked)):
                image = self._flip(image)
                image = self._translate(image)
                image = self._crop(image)
                image = self._hsv(image)
                image = self._noise(image)
                image = self._clip(image)
                image = self._dequality(image)

                images[index] = image.astype(self.dtype) / 255.
                labels[index][label] = 1

            try:
                self.store.put([images, labels])
            except:
                sys.exit()