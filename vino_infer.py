from openvino.inference_engine import IENetwork, IEPlugin
import argparse
import collections
import os
from multiprocessing import Queue
import threading
import time

import cv2
import numpy as np
import torch
import tqdm
from PIL import Image
from torch.autograd import Variable

from models import TransformerNet
from utils import *


# bufferless VideoCapture
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except:
                    # pass
                    continue
            self.q.put(frame)

    def read(self):
        return self.q.get()


filepaths = set()
for root, dirs, files in os.walk("deploy_models", topdown=False):
    for f in files:
        filepath = os.path.join(root, f)
        if ".bin" in filepath or ".xml" in filepath:
            filepaths.add(filepath.split('.')[0])

filepaths = list(filepaths)

def load_model(root_name):
    model = root_name + ".xml"
    weights = root_name + ".bin"
    plugin = IEPlugin(device="MYRIAD") 
    net = IENetwork(model=model, weights=weights) 
    exec_net = plugin.load(network=net)
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    return exec_net, input_blob, out_blob

print(filepaths)
if __name__ == "__main__":
    cap = VideoCapture(0)

    exec_net, input_blob, out_blob = load_model(filepaths[1])
    while True:
        # Prepare input frame
        # Stylize image
        img = cap.read()
        if img is None:
            continue
        # cv2.imshow('plm', img)
        if chr(cv2.waitKey(1) & 255) == 'q':
            break

        with torch.no_grad():
            # cv2.imshow('Test', img)
            img = cv2.resize(img, (300, 300))
            img = np.array([img])
            img = np.transpose(img,(0,3,1,2))
            # 
            # print(img.shape)
            res = exec_net.infer(inputs={input_blob: img})
            # print(res)
            res = res[out_blob][0]
            res = np.transpose(res, (1,2,0))
            # print(res.shape)

            # print(stylized_image.shape)
            if res is not None:
                cv2.imshow('Darius Omaj', res)
