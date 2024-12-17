import cv2
import numpy as np
import os
import sys
import threading

class Save(threading.Thread):
    def __init__(self, src, dest):
        super().__init__()
        self.src = src
        self.dest = dest

    def run(self):
        image = cv2.imread(self.src, cv2.IMREAD_COLOR)
        if image is None: return
        np.save(self.dest, image)

def decode(src, export):
    categories = os.listdir(src)

    for category in sorted(categories):
        print(category, end="\r")
        category_path = os.path.join(src, category)
        files = os.listdir(category_path)
        
        for i in range(0, len(files), 64):
            threads = []
            for file in files[i:i+64]:
                src_ = os.path.join(category_path, file)
                dest_dir = os.path.join(export, category)

                if not os.path.isdir(dest_dir):
                    os.makedirs(dest_dir)

                dest_ = os.path.join(dest_dir, file)

                threads.append(Save(src_, dest_))
                threads[-1].start()
            
            for thread in threads:
                thread.join()

if __name__ == "__main__":
    path = None
    export = None

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

        decode(src, dst)