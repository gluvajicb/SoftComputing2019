import numpy as np
import cv2
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import sys


def loading_video():

    video = cv2.VideoCapture(sys.argv[1])
    ret, frame = video.read()

    if ret is False:
        return False

    return frame


if __name__ == '__main__':

    frame = loading_video()

    if frame is False:
        print("Can not load video! Exiting...")

    else:
        print("Video loaded!")

