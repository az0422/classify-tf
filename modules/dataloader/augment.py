import cv2
import numpy as np
import random
import multiprocessing
import copy
import tensorflow as tf

from multiprocessing import shared_memory

from .utils import resize_contain, resize_stretch

LOSS_FORMAT = [
    [".jpeg", cv2.IMWRITE_JPEG_QUALITY],
    [".webp", cv2.IMWRITE_WEBP_QUALITY],
    #[".avif", cv2.IMWRITE_AVIF_QUALITY],
]

class Augmentor():
    def __init__(self, cfg, augment_flag):
        self.cfg = cfg
        self.augment_flag = augment_flag
        self.buffer = np.zeros(
            [
                self.cfg["image_size"],
                self.cfg["image_size"],
                3
            ],
            dtype=np.uint8,
        )
    
    def __call__(self, image):
        np.copyto(self.buffer, 0)
        np.copyto(self.buffer, self._resize(image))

        if self.augment_flag:
            np.copyto(self.buffer, self._crop(self.buffer))
            np.copyto(self.buffer, self._flip(self.buffer))
            np.copyto(self.buffer, self._translate(self.buffer))
            np.copyto(self.buffer, self._hsv(self.buffer))
            np.copyto(self.buffer, self._noise(self.buffer))
            np.copyto(self.buffer, self._dequality(self.buffer))

        return self.buffer
    
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

class DataAugment(multiprocessing.Process):
    def __init__(self, seed, images, classes, cfg, augment=True, categories=None):
        super().__init__()
        np.random.seed((~seed) & 0xFFFFFFFF)
        random.seed(seed if np.random.rand() > 0.5 else seed * -1)

        self.images = images
        self.classes = classes
        self.cfg = cfg
        self.augment = augment
        self.categories = categories

        image_data_length = cfg["batch_size"] * (self.cfg["image_size"] ** 2) * 3
        label_data_length = cfg["batch_size"] * cfg["classes"]

        self.writer_index_queue = multiprocessing.Queue(self.cfg["queue_size"])
        self.reader_index_queue = multiprocessing.Queue(self.cfg["queue_size"])

        for i in range(self.cfg["queue_size"]):
            self.writer_index_queue.put(i)
        
        self.images_buff = shared_memory.SharedMemory(create=True, size=image_data_length * self.cfg["queue_size"])
        self.labels_buff = shared_memory.SharedMemory(create=True, size=label_data_length * self.cfg["queue_size"])

        self.images_buff_np = np.ndarray(
            [
                self.cfg["queue_size"],
                self.cfg["batch_size"],
                self.cfg["image_size"],
                self.cfg["image_size"],
                3
            ],
            dtype=np.uint8,
            buffer=self.images_buff.buf
        )

        self.labels_buff_np = np.ndarray(
            [
                self.cfg["queue_size"],
                self.cfg["batch_size"],
                self.cfg["classes"],
            ],
            dtype=np.uint8,
            buffer=self.labels_buff.buf
        )

    def terminate(self):
        self.images_buff.unlink()
        self.labels_buff.unlink()

        super().terminate()
    
    def run(self):
        buff_images_shm = np.ndarray(
            [
                self.cfg["queue_size"],
                self.cfg["batch_size"],
                self.cfg["image_size"],
                self.cfg["image_size"],
                3
            ],
            dtype=np.uint8,
            buffer=self.images_buff.buf
        )
        buff_labels_shm = np.ndarray(
            [
                self.cfg["queue_size"],
                self.cfg["batch_size"],
                self.cfg["classes"],
            ],
            dtype=np.uint8,
            buffer=self.labels_buff.buf
        )

        buff_images_batch = np.zeros(
            [
                self.cfg["batch_size"],
                self.cfg["image_size"],
                self.cfg["image_size"],
                3,
            ],
            dtype=np.uint8
        )
        buff_labels_batch = np.zeros(
            [
                self.cfg["batch_size"],
                self.cfg["classes"]
            ],
            dtype=np.uint8
        )

        images = copy.deepcopy(self.images)
        cfg = copy.deepcopy(self.cfg)
        augmentor = Augmentor(cfg, self.augment)
        
        while True:
            taked_images_list = random.sample(images, self.cfg["batch_size"])
            np.copyto(buff_images_batch, 0)
            np.copyto(buff_labels_batch, 0)

            for sub_index, (image, label) in enumerate(taked_images_list):
                image = cv2.imread(image, cv2.IMREAD_COLOR)
                image = augmentor(image)
                
                if cfg["color_space"].lower() == "rgb":
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif cfg["color_space"].lower() == "hsv":
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.uint8)
                
                np.copyto(buff_images_batch[sub_index], image)
                buff_labels_batch[sub_index][label] = 1

            data_index = self.writer_index_queue.get()
            np.copyto(buff_images_shm[data_index], buff_images_batch)
            np.copyto(buff_labels_shm[data_index], buff_labels_batch)
            self.reader_index_queue.put(data_index)
        
    def getData(self):
        index = self.reader_index_queue.get()
        images = tf.convert_to_tensor(self.images_buff_np[index], tf.float32) / 255.
        labels = tf.convert_to_tensor(self.labels_buff_np[index], tf.float32)
        self.writer_index_queue.put(index)

        return images, labels