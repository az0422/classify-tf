import os
import threading
import sys
import random

class Copy(threading.Thread):
    def __init__(self, src, dst):
        super().__init__()
        self.src = src
        self.dst = dst

    def run(self):
        with open(self.src, "br") as src:
            with open(self.dst, "bw") as dst:
                dst.write(src.read())

def split(root, ratio):
    categories = sorted(os.listdir(root))

    for category in categories:
        print(category, end="\r")
        category_path = os.path.join(root, category)

        threads = []
        files = os.listdir(category_path)
        random.shuffle(files)
        length = len(files)

        train = files[round(length * ratio):]
        val = files[:round(length * ratio)]

        os.makedirs(os.path.join(root, "train", category))
        os.makedirs(os.path.join(root, "val", category))

        for file in train:
            src = os.path.join(category_path, file)
            dst = os.path.join(root, "train", category, file)

            threads.append(Copy(src, dst))
            threads[-1].start()

        for file in val:
            src = os.path.join(category_path, file)
            dst = os.path.join(root, "val", category, file)

            threads.append(Copy(src, dst))
            threads[-1].start()

        for thread in threads:
            thread.join()

if __name__ == "__main__":
    path = None
    ratio = 0.2

    for arg in sys.argv:
        if arg.startswith("path"):
            path = arg.split("=")[1]
            continue
        if arg.startswith("ratio"):
            ratio = float(arg.split("=")[1])
            continue
    
    if path is None:
        print("Please set the `path` argument by dataset's root path")
        sys.exit()
    
    split(path, ratio)