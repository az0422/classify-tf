import cv2
import numpy as np
import random
import multiprocessing

from .utils import resize_contain, resize_stretch

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
        
        random_ratio = self.cfg["crop_ratio"] + (1 - self.cfg["crop_ratio"]) * np.random.rand(2)

        crop_width, crop_height = (np.array([width, height]) * random_ratio).astype(np.int32)

        pad_x = np.random.randint(0, width - crop_width + 1)
        pad_y = np.random.randint(0, height - crop_height + 1)
        
        if self.cfg["crop_method"] == "blank":
            flags = np.ones(image.shape[:2], dtype=bool)
            flags[pad_y:pad_y + crop_height, pad_x:pad_x + crop_width] = False
            image[flags] = 0.
        
        elif self.cfg["crop_method"] == "resize":
            image_crop = image[pad_y:pad_y + crop_height, pad_x:pad_x + crop_width]
            image = self._resize(image_crop)

        return image
    
    def _translate(self, image):
        height, width, _ = image.shape
        vertical = random.uniform(-self.cfg["translate_vertical"], self.cfg["translate_vertical"]) * height
        horizontal = random.uniform(-self.cfg["translate_horizontal"], self.cfg["translate_horizontal"]) * width
        degree = random.uniform(-self.cfg["rotate_degree"], self.cfg["rotate_degree"])

        zoom = np.sqrt(np.random.rand() * (self.cfg["zoom"] - 1) + 1)
        if np.random.rand() > 0.5: zoom = 1 / zoom

        matrix = cv2.getRotationMatrix2D((width // 2, height // 2), degree, zoom,)
        matrix[:2, 2] += [horizontal, vertical]

        image = cv2.warpAffine(image, matrix, [width, height], borderMode=cv2.BORDER_REPLICATE)

        return image
    
    def _hsv(self, image):
        hsv = np.array([self.cfg["hsv_h"], self.cfg["hsv_s"], self.cfg["hsv_v"]], dtype=np.float32)
        hsv[hsv > 1.] = 1.
        hsv[hsv < 0.] = 0.

        hsv_mul = np.random.rand(3) * hsv * 2 - hsv + 1.

        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        image[..., :] *= hsv_mul
        image[..., 0] = image[..., 0] % 180
        image = np.clip(image, 0, 255.)
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

        return image
    
    def _dequality(self, image):
        noise = (np.random.rand(*image.shape) * 255.).astype(np.float32)
        opacity = np.random.rand() * self.cfg["noise"]
        dequality = 1 - np.random.rand() * self.cfg["dequality"]

        image = image * (1 - opacity) + noise * opacity

        if dequality > 0:
            _, image = cv2.imencode(".jpg", image.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, round(dequality * 100)])
            image = cv2.imdecode(image, cv2.IMREAD_COLOR).astype(np.float32)

        return image
    
    def run(self):
        while True:
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
                
                image = self._resize(image).astype(np.float32)
                image = self._crop(image)
                image = self._flip(image)
                image = self._translate(image)
                image = self._hsv(image)
                image = self._dequality(image)

                image = image.astype(np.uint8)

                if self.cfg["color_space"].lower() == "rgb":
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                elif self.cfg["color_space"].lower() == "hsv":
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                
                image = image.astype(np.float32)

                images[index] = image / 255.
                labels[index][label] = 1

            self.queue.put([images, labels])
        
    def getData(self):
        return self.queue.get()