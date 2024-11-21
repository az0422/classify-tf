import cv2
import os
import threading
import sys

class Save(threading.Thread):
    def __init__(self, src, dest, size):
        super().__init__()
        self.src = src
        self.dest = dest
        self.size = size

    def run(self):
        image = cv2.imread(self.src, cv2.IMREAD_COLOR)
        if image is None: return
        height, width, _ = image.shape
        ratio = max(height, width) / self.size
        image = cv2.resize(image, (round(width/ratio), round(height/ratio)))
        cv2.imwrite(self.dest, image)

def convert(src, export, size):
    categories = os.listdir(src)

    for category in sorted(categories):
        print(category, end="\r")
        category_path = os.path.join(src, category)
        files = os.listdir(category_path)
        threads = []

        for file in files:
            src = os.path.join(category_path, file)
            dest_dir = os.path.join(export, category)

            if not os.path.isdir(dest_dir):
              os.makedirs(dest_dir)

            dest = os.path.join(dest_dir, file)

            threads.append(Save(src, dest, size))
            threads[-1].start()

if __name__ == "__main__":
    path = None
    export = None
    size = 256

    for arg in sys.argv:
        if arg.startswith("path"):
            path = arg.split("=")[1]
            continue
        if arg.startswith("export"):
            export = arg.split("=")[1]
            continue
    
    if path is None:
        print("Please set the `path` argument by dataset's root path")
        sys.exit()
    
    if export is None:
        print("Please set the `path` argument by dataset's root path!!")
        sys.exit()
    
    listdir = os.listdir(path)

    for d in listdir:
        print(d)
        src = os.path.join(path, d)
        dst = os.path.join(export, d)

        convert(src, dst, size)