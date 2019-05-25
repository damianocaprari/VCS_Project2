import cv2
import os
import torchvision.transforms as transforms

from yolo_v3.utils.datasets import pad_to_square, resize
from person import *
from yolo_v3.utils.utils import rescale_boxes
import numpy as np
from parameters import Parameters as P


def cv2_img_to_torch_tensor(img, img_size):
    torch_img = transforms.ToTensor()(img)
    torch_img, _ = pad_to_square(torch_img, 0)
    torch_img = resize(torch_img, img_size)
    return torch_img


def frame_extraction_from_video(video_name, folder_name):
    vidcap = cv2.VideoCapture(video_name)
    success, image = vidcap.read()
    count = 0
    os.mkdir(folder_name)
    while success:
        cv2.imwrite("./"+folder_name+"/frame%d.jpg" % count, image)
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


def mog():
    cap = cv2.VideoCapture('./Videos/video1.mp4')
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(nmixtures=3)
    while(1):
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        cv2.imshow('frame',fgmask)
        k = cv2.waitKey(20) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def read_image_cv2_torch(input_img, img_size):
    img = cv2.imread(input_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    torch_img = cv2_img_to_torch_tensor(img, img_size)
    return img, torch_img


class ColorGenerator(object):

    def __init__(self):
        self.__used_colors = np.empty((1, 3), dtype=np.uint8)

    def generate_color(self):
        color = np.random.randint(0, 256, 3, np.uint8)
        r = self.__used_colors[:-1]

        while True:
            if np.any(r[r == color]):
                continue
            self.__used_colors[-1] = color
            self.__used_colors = np.vstack((self.__used_colors, np.empty((1, 3), dtype=np.uint8)))
            return color

    @property
    def colors(self):
        return self.__used_colors[:-1]


def follow_old_person(person, old_persons, tmp_persons):
    # if old_persons_exists is True:
    if old_persons:
        idx_closest = find_closest_person(person, old_persons)
        person.id = old_persons[idx_closest].id
        person.color = old_persons[idx_closest].color

        # person.centroid_past.append(old_persons[idx_closest].centroid.astype(np.int))
        # person.centroid_past.extend(old_persons[idx_closest].centroid_past.astype(np.int))
        # person.centroid_past.append(old_persons[idx_closest].centroid_past.astype(np.int))
        cp = old_persons[idx_closest].centroid_past
        person.centroid_past.extend(cp)

        
        old_persons.remove(old_persons[idx_closest])
        tmp_persons.append(person)
        #if not old_persons:
            # lista vuota
            #old_persons_exists = False
    else:
        tmp_persons.append(person)
        # --------

    return person, old_persons, tmp_persons
