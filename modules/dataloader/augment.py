import cv2
import numpy as np
import random
import multiprocessing
import queue
import gc

from .utils import LoadImage, resize_contain, resize_stretch

class DataAugment(multiprocessing.Process):
    def __init__(self, seed, images, classes, batch_size, cfg):
        super().__init__()
        np.random.seed((~seed) & 0xFFFFFFFF)
        random.seed(seed if np.random.rand() > 0.5 else seed * -1)

        self.images = images
        self.classes = classes
        self.cfg = cfg
        self.stop = False

        self.queue = multiprocessing.Queue(self.cfg["queue_size"])
        self.batch_size = batch_size
        self.color_space = self.cfg["color_space"].lower()

    def _resize(self, image):
        if self.cfg["resize_method"] in ("default", "contain"):
            return resize_contain(image, self.cfg["image_size"])
        if self.cfg["resize_method"] in ("stretch",):
            return resize_stretch(image, self.cfg["image_size"])
        raise Exception("invalid resize method %s" % self.cfg["resize_method"])
    
    def _flip(self, image):
        if np.random.rand() < self.cfg["flip_vertical"]:
            image = image[::-1]
        
        if np.random.rand() < self.cfg["flip_horizontal"]:
            image = image[:, ::-1]
        
        return image
    
    def _crop(self, image):
        height, width, _ = image.shape

        crop_width, crop_height = ((1 - np.random.rand(2) * self.cfg["crop_ratio"]) * [width, height]).astype(np.int32)
        padding = [width - crop_width, height - crop_height]
        offset_x, offset_y = (np.random.rand(2) * padding).astype(np.int32)

        crop_index = np.ones([height, width], dtype=np.bool_)
        crop_index[offset_y:offset_y + crop_height, offset_x:offset_x + crop_width] = False
        image[crop_index] = 0

        return image
    
    def _translate(self, image):
        height, width, _ = image.shape
        vertical = (np.random.rand() * self.cfg["translate_vertical"] * 2 - self.cfg["translate_vertical"]) * height
        horizontal = (np.random.rand() * self.cfg["translate_horizontal"] * 2 - self.cfg["translate_horizontal"]) * width
        degree = np.random.rand() * self.cfg["rotate_degree"] * 2 - self.cfg["rotate_degree"]

        zoom = np.random.rand() * (self.cfg["zoom"] - 1) + 1
        if np.random.rand() > 0.5: zoom = 1 / zoom

        matrix = cv2.getRotationMatrix2D((width // 2, height // 2), degree, zoom)
        matrix[:2, 2] += [horizontal, vertical]

        image = cv2.warpAffine(image, matrix, [width, height])

        return image
    
    def _hsv(self, image):
        if self.color_space != "hsv":
            image = cv2.cvtColor(image.astype(np.uint8), {"bgr": cv2.COLOR_BGR2HSV, "rgb": cv2.COLOR_RGB2HSV}[self.color_space]).astype(np.float32)

        hsv = np.array([self.cfg["hsv_h"], self.cfg["hsv_s"], self.cfg["hsv_v"]], dtype=np.float32)
        hsv[hsv > 1.] = 1.
        hsv[hsv < 0.] = 0.

        hsv_mul = np.random.rand(3) * hsv * 2 - hsv + 1.
 
        image[..., :] *= hsv_mul
        image[..., 0] = image[..., 0] % 180
        image[image > 255.] = 255.
        image[image < 0.] = 0.

        if self.color_space != "hsv":
            image = cv2.cvtColor(image.astype(np.uint8), {"bgr": cv2.COLOR_HSV2BGR, "rgb": cv2.COLOR_HSV2RGB}[self.color_space]).astype(np.float32)

        return image
    
    def _noise(self, image):
        noise = (np.random.rand(*image.shape) * 255. * self.cfg["noise"]).astype(np.float32)

        if self.color_space == "hsv":
            noise = cv2.cvtColor(noise, cv2.COLOR_BGR2HSV)
        
        image = np.clip(image, 0, 255.)

        return image
    
    def _dequality(self, image):
        quality = int(100 - np.random.rand() * self.cfg["dequality"])
        
        if self.color_space != "bgr":
            image = cv2.cvtColor(image, {"rgb": cv2.COLOR_RGB2BGR, "hsv": cv2.COLOR_HSV2BGR}[self.color_space])
        
        image_np = cv2.imencode(".jpeg", image.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR).astype(np.float32)

        if self.color_space != "bgr":
            image = cv2.cvtColor(image, {"rgb": cv2.COLOR_BGR2RGB, "hsv": cv2.COLOR_BGR2HSV}[self.color_space])

        return image
    
    def run(self):
        gc_count = 0
        while not self.stop:
            taked_indices = [random.randrange(0, len(self.images) - 1) for _ in range(self.batch_size)]
            images = np.zeros(
                [
                    self.batch_size,
                    self.cfg["image_size"],
                    self.cfg["image_size"],
                    3
                ], dtype=np.float32
            )
            labels = np.zeros(
                [
                    self.batch_size,
                    self.classes,
                ], dtype=np.uint8
            )

            for index, taked_index in enumerate(taked_indices):
                image, label = self.images[taked_index]
                image = cv2.imread(image, cv2.IMREAD_COLOR)

                if self.color_space != "bgr":
                    image = cv2.cvtColor(image, {"rgb": cv2.COLOR_BGR2RGB, "hsv": cv2.COLOR_BGR2HSV}[self.color_space])
                
                image = self._resize(image)
                image = self._flip(image)
                image = self._translate(image)
                image = self._hsv(image)
                image = self._noise(image)
                image = self._dequality(image)

                images[index] = image / 255.
                labels[index][label] = 1

            self.queue.put([images, labels])

            del (
                images,
                labels,
            )

            gc_count = (gc_count + 1) % len(self.images) // self.batch_size

            if gc_count == len(self.images) // self.batch_size - 1:
                gc.collect()

        
    def getData(self):
        return self.queue.get()