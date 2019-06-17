import glob

import cv2

from utils import read_image_cv2_torch, cv2_img_to_torch_tensor, load_undistortion_parameters
from calibration import undistort_img
from parameters import Parameters as P


class ImageFolderDataLoader(object):

    def __init__(self, folder_name, img_size):
        if folder_name[-1] == '/':
            append = '**'
        else:
            append = '/**'
        self.folder = glob.glob(folder_name + append)
        self.img_size = img_size
        self.last_index = 0
        self.camera_matrix, self.distortion_coefficients, _, _ = load_undistortion_parameters()

    def __getitem__(self, index):
        img, torch_img = read_image_cv2_torch(self.folder[index], self.img_size, self.camera_matrix, self.distortion_coefficients)
        return img, torch_img

    def __next__(self):
        if self.last_index == len(self):
            raise StopIteration()
        img, torch_img = self.__getitem__(self.last_index)
        self.last_index += 1
        return img, torch_img

    def __len__(self):
        return len(self.folder)


class VideoDataLoader(object):

    def __init__(self, video_path, img_size):
        self.video_capture = cv2.VideoCapture(video_path)
        self.img_size = img_size
        self.camera_matrix, self.distortion_coefficients, _, _ = load_undistortion_parameters()

    def __iter__(self):
        return self

    def __next__(self):
        if not self.video_capture.isOpened():
            raise StopIteration()

        ret, frame = self.video_capture.read()
        if ret is False:
            raise StopIteration()
        frame = undistort_img(frame, self.camera_matrix, self.distortion_coefficients, P.DISTORTION.ALPHA)
        torch_frame = cv2_img_to_torch_tensor(frame, self.img_size)
        return frame, torch_frame

    def close(self):
        self.video_capture.release()
