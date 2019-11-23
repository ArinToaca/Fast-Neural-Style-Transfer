from models import TransformerNet
import cv2
from utils import *
import torch
from torch.autograd import Variable
import argparse
import os
import tqdm
from PIL import Image

import numpy as np
import cv2

import collections


def get_frame(cap):

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here

        # Display the resulting frame
        # cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        yield frame

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


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

    while True:
        # Prepare input frame
        # Stylize image
        with torch.no_grad():
            for img in yield_frame():
                # cv2.imshow('Test', img)
                image_tensor = Variable(transform(img)).to(device).unsqueeze(0)
                stylized_image = transformer(image_tensor)
                stylized_image = np.array(stylized_image)[0]
                stylized_image = np.transpose(stylized_image, (1, 2, 0))
                print(stylized_image.shape)
                cv2.imshow('Darius Omaj', stylized_image)