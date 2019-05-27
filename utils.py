import cv2
import os
import torchvision.transforms as transforms

from yolo_v3.utils.datasets import pad_to_square, resize
from person import *
from yolo_v3.utils.utils import rescale_boxes
import numpy as np
from parameters import Parameters as P


def euclidean_distance(p1, p2):
    print(p1,p2)
    return np.sqrt(np.sum(np.square(p1 - p2), axis=0)).astype(np.float)


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
    if old_persons:
        idx_closest = find_closest_person(person, old_persons)
        person.id = old_persons[idx_closest].id
        person.color = old_persons[idx_closest].color

        cp = old_persons[idx_closest].centroid_past
        person.centroid_past.extend(cp)

        old_persons.remove(old_persons[idx_closest])
        tmp_persons.append(person)
    else:
        tmp_persons.append(person)
    return person, old_persons, tmp_persons


def predict_centroid(old_persons):
    for p in old_persons:
        print(p.centroid_past)
        if len(p.centroid_past) > 1:
            t0 = p.centroid_past[0]
            tm1 = p.centroid_past[1]
            tp1x = 2 * t0[0] - tm1[0]   # punto medio
            tp1y = 2 * t0[1] - tm1[1]
            future_point = (tp1x, tp1y)
            p.centroid_future = future_point
            print(p.centroid_past)
            print("future point: ", future_point)


def tracking_centroid(old_persons, img):
    imm = cv2.imread('./Frames1/frame0.jpg', )
    for i in old_persons:
        #for j in i.centroid_past:
        for j in reversed(i.centroid_past):
            print(j[0], j[1])
            cv2.circle(imm, (j[0].astype(np.int), j[1].astype(np.int)), 2, i.color, -1)
            cv2.imshow('out', imm)
            cv2.waitKey(40)
    cv2.imwrite('./out_track/track_video3.jpg', imm)


def match_likelihood(this, other):
    """
    :param this: person
    :param other:  person
    :return: likelihood value
    """
    # check thresholds
    dist = euclidean_distance(this.centroid, other.centroid)
    if dist > P.LIKELIHOOD.DISTANCE_THS:
        return 0

    # calculate likelihood as a function of distance, color, ...
    contributions = []
    contributions.append( min(np.reciprocal(dist), np.finfo(np.float).max))  # distance
    contributions.append( 0 )  # TODO decidere come calcolare la contribution del colore
    # contriutions.append( ... )

    # TODO forse vale la pena imparare la loss function con del ML
    return np.sum(np.multiply( contributions, P.LIKELIHOOD.WEIGHTS ))


def update_persons(persons_new, persons_old, max_used_id):
    persons_tmp = []
    if persons_old:     # forse inutile dato che ho gia fatto il controllo fuori dalla funzione
        if len(persons_new) <= len(persons_old):
            remaining = persons_old
            for person_new in persons_new:
                likelihoods = list(map(lambda x: match_likelihood(person_new, x), remaining))
                if np.amax(likelihoods) <= 0:
                    # no matches found
                    max_used_id += 1
                    person_new.id = max_used_id
                    persons_tmp.append(person_new)
                else:
                    person_matching = remaining.pop(np.argmax(likelihoods))
                    person_matching.centroid_past.insert(0, person_new.centroid)
        else:  # len(persons_new) > len(persons_old):
            remaining = persons_new
            for person_old in persons_old:
                likelihoods = list(map(lambda x: match_likelihood(person_old, x), remaining))
                if np.amax(likelihoods) <= 0:
                    # no matches found
                    continue
                else:
                    person_matching = remaining.pop(np.argmax(likelihoods))
                    person_old.centroid_past.insert(0, person_matching.centroid)

            for p in remaining:
                max_used_id += 1
                p.id = max_used_id
                persons_tmp.append(p)

    else:  # persons_old is empty
        for p in persons_new:
            max_used_id += 1
            p.id = max_used_id
            persons_tmp.append(p)

    return persons_tmp, max_used_id

