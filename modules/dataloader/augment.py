import cv2
import numpy as np
import random
import multiprocessing

from .utils import resize_contain, resize_stretch

LOSS_FORMAT = [
    [".jpeg", cv2.IMWRITE_JPEG_QUALITY],
    [".webp", cv2.IMWRITE_WEBP_QUALITY],
    #[".avif", cv2.IMWRITE_AVIF_QUALITY],
]

class DataAugment(multiprocessing.Process):
    def __init__(self, seed, images, classes, cfg, augment=True):
        super().__init__()
        np.random.seed((~seed) & 0xFFFFFFFF)
        random.seed(seed if np.random.rand() > 0.5 else seed * -1)

        self.images = images
        self.classes = classes
        self.cfg = cfg
        self.augment = augment

        self.queue = multiprocessing.Queue(self.cfg["queue_size"])

    def _resize(self, image):
        if self.cfg["resize_method"] in ("default", "contain"):
            return resize_contain(image, self.cfg["image_size"])
        if self.cfg["resize_method"] in ("stretch",):
            return resize_stretch(image, self.cfg["image_size"])
        raise Exception("invalid resize method %s" % self.cfg["resize_method"])
    
    def _flip(self, image):
        if np.random.rand() < self.cfg["flip_vertical"]:
            image = np.flip(image, axis=0)
        
        if np.random.rand() < self.cfg["flip_horizontal"]:
            image = np.flip(image, axis=1)
        
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
        hsv = np.clip(
            np.array([self.cfg["hsv_h"], self.cfg["hsv_s"], self.cfg["hsv_v"]], dtype=np.float16),
            0.0,
            1.0
        )
        hsv_offset = np.clip(
            np.array([self.cfg["hsv_offset_h"], self.cfg["hsv_offset_s"], self.cfg["hsv_offset_v"]], dtype=np.float16),
            -1.0,
            1.0,
        )

        hsv = np.random.rand(3) * hsv * 2 - hsv + 1.

        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float16)
        image_hsv = image_hsv + image_hsv * hsv_offset
        image_hsv = image_hsv * hsv
        image_hsv[..., 0] = image_hsv[..., 0] % 180
        image = np.clip(image_hsv, 0, 255.).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        return image
    
    def _noise(self, image):
        noise_cfg = self.cfg["noise"]
        noise_mean = np.random.rand() * abs(noise_cfg["mean"]) * 2 - abs(noise_cfg["mean"])
        noise_std = np.random.rand() * (noise_cfg["std"][1] - noise_cfg["std"][0]) + noise_cfg["std"][0]

        noise = np.random.normal(noise_mean, noise_std, image.shape)
        image = image.astype(np.int16) + noise.astype(np.int16)
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image
    
    def _dequality(self, image):
        quality = 1 - np.random.rand() * self.cfg["dequality"]

        if quality < 0.9:
            fmt, qlt = random.choices(LOSS_FORMAT)[0]
            _, image = cv2.imencode(fmt, image, [qlt, round(quality * 100)])
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        return image
    
    def run(self):
        while True:
            images = []
            labels = []
            images_list = [random.choice(self.images) for _ in range(self.cfg["batch_size"])]

            for image, label in images_list:
                if image.endswith(".npy"):
                    image = np.load(image)
                else:
                    image = cv2.imread(image, cv2.IMREAD_COLOR)
                
                image = self._resize(image)

                if self.augment:
                    image = self._crop(image)
                    image = self._flip(image)
                    image = self._translate(image)
                    image = self._hsv(image)
                    image = self._noise(image)
                    image = self._dequality(image)

                if self.cfg["color_space"].lower() == "rgb":
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                elif self.cfg["color_space"].lower() == "hsv":
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                
                label_np = np.zeros([self.classes], dtype=np.uint8)
                label_np[label] = 1
                
                images.append(np.expand_dims(image, axis=0))
                labels.append(np.expand_dims(label_np, axis=0))
            
            images = np.concatenate(images, axis=0)
            labels = np.concatenate(labels, axis=0)
            
            self.queue.put([images, labels])
        
    def getData(self):
        return self.queue.get()