import glob

import cv2
from torch.utils.data import Dataset, DataLoader

from utils import read_image_cv2_torch, cv2_img_to_torch_tensor


class _ImageFolderDataset(Dataset):

    def __init__(self, folder_name, img_size):
        if folder_name[-1] == '/':
            append = '**'
        else:
            append = '/**'
        self.folder = glob.glob(folder_name + append)
        self.img_size = img_size

    def __getitem__(self, index):
        return read_image_cv2_torch(self.folder[index], self.img_size)

    def __len__(self):
        return len(self.folder)


def ImageFolderDataLoader(folder_name, img_size):
    return DataLoader(_ImageFolderDataset(folder_name, img_size))


class VideoDataLoader(object):

    def __init__(self, video_path, img_size):
        self.video_capture = cv2.VideoCapture(video_path)
        self.img_size = img_size

    def __iter__(self):
        return self

    def __next__(self):
        if not self.video_capture.isOpened():
            raise StopIteration()

        ret, frame = self.video_capture.read()
        if ret is False:
            raise Exception()
        torch_frame = cv2_img_to_torch_tensor(frame, self.img_size)
        return frame, torch_frame

    def close(self):
        self.video_capture.release()
