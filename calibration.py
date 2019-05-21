import glob

import numpy as np
import cv2

from parameters import Parameters as P


def run_calibration():
    directory = glob.glob('./CalibrationImages/**')

    for frame in directory:
        img = cv2.imread(frame)
        cv2.imshow('out', img)
        cv2.waitKey()


run_calibration()
