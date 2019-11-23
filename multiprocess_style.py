import argparse
import collections
import os
import onnx
from onnx import utils
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
                except Queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_model",
                        type=str,
                        required=True,
                        help="Path to checkpoint model")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = style_transform()

    # Define model and load model checkpoint
    transformer = TransformerNet().to(device)
    transformer.load_state_dict(
        torch.load(args.checkpoint_model, map_location='cpu'))
    transformer.eval()
    dummy_input = torch.randn(1, 3, 128, 128, device='cpu')
    onnx_file = ".".join([args.checkpoint_model.split('.')[0],'onnx'])
    print(f"Exporting as {onnx_file}")
    torch.onnx.export(transformer, dummy_input, onnx_file)
    original_model = onnx.load(onnx_file)
    polished_model = utils.polish_model(original_model)
    onnx.save(polished_model, onnx_file)
    print("Successfully polished model")
    exit()

    cap = VideoCapture(0)

    while True:
        # Prepare input frame
        # Stylize image
        img = cap.read()
        if chr(cv2.waitKey(1) & 255) == 'q':
            break

        with torch.no_grad():
            # cv2.imshow('Test', img)
            img = cv2.resize(img, (128, 128))
            image_tensor = Variable(transform(img)).to(device).unsqueeze(0)
            stylized_image = transformer(image_tensor)
            stylized_image = np.array(stylized_image)[0]
            stylized_image = np.transpose(stylized_image, (1, 2, 0))
            # print(stylized_image.shape)
            cv2.imshow('Darius Omaj', stylized_image)
