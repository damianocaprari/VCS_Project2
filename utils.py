import cv2
import os
import torchvision.transforms as transforms

from yolo_v3.utils.datasets import pad_to_square, resize
from person import *
from yolo_v3.utils.utils import rescale_boxes
import numpy as np
from parameters import Parameters as P
from calibration import undistort_img


def euclidean_distance(p1, p2):
    #print(p1, p2)
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


def read_image_cv2_torch(input_img, img_size, camera_matrix, distortion_coefficients):
    img = cv2.imread(input_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = undistort_img(img, camera_matrix, distortion_coefficients, P.DISTORTION.ALPHA)
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
        # for j in i.centroid_past:
        for j in reversed(i.centroid_past):
            print(j[0], j[1])
            cv2.circle(imm, (j[0].astype(np.int), j[1].astype(np.int)), 2, i.color, -1)
            cv2.imshow('out', imm)
            cv2.waitKey(40)
    cv2.imwrite('./out_track/track_video3.jpg', imm)


# -- NON USATO
def follow_new_SIFT(person, old_persons, tmp_persons, img):
    if old_persons:
        print("current person\n", person.sift_kp, len(person.sift_descriptors))
        #x11, y11 = person.p1
        #x12, y12 = person.p2
        ##x11 = max(0, x11)
        #y11 = max(0, y11)
        #x12 = max(0, x12)
        #y12 = max(0, y12)

        #crop_img = img[y11:y12, x11:x12]
        #cv2.imshow('current person', crop_img)
        #cv2.waitKey()
        #gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        #sift = cv2.xfeatures2d.SIFT_create()

        #kp1 = sift.detect(gray, None)
        #kp1, des1 = sift.compute(gray, kp1)
        bf = cv2.BFMatcher()
        max_matches = 0
        list_good = []
        for idx, p in enumerate(old_persons):
            print("old person ", idx, '\n', p.sift_kp, len(p.sift_descriptors))
            # -- p.sift_descriptors e' una LISTA, io devo avere un array
            des1 = np.asarray(person.sift_descriptors)
            des2 = np.asarray(p.sift_descriptors)
            matches = bf.knnMatch(des1, des2, k=2)
            # matches = sorted(matches, key = lambda x:x.distance)
            good = []
            # if ma
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            list_good.append(len(good))
        if list_good:
            idx_betterSIFT_old_person = list_good.index(max(list_good))
            # idx_closest = find_closest_person(person, old_persons)
            person.id = old_persons[idx_betterSIFT_old_person].id
            person.color = old_persons[idx_betterSIFT_old_person].color
            cp = old_persons[idx_betterSIFT_old_person].centroid_past
            person.centroid_past.extend(cp)
            old_persons.remove(old_persons[idx_betterSIFT_old_person])
            tmp_persons.append(person)

            #x21, y21 = p.p1
            #x22, y22 = p.p2
            #x21 = max(0, x21)
            #y21 = max(0, y21)
            #x22 = max(0, x22)
            #y22 = max(0, y22)

            #crop_img2 = img[y21:y22, x21:x22]
            #cv2.imshow('old person', crop_img2)
            #cv2.waitKey()

            #gray2 = cv2.cvtColor(crop_img2, cv2.COLOR_BGR2GRAY)
            #kp2 = sift.detect(gray2, None)
            #kp2, des2 = sift.compute(gray2, kp2)
            #img = cv2.drawKeypoints(gray, kp1, img)
            #cv2.imshow('a1', img)
            #cv2.waitKey()

            #img2 = cv2.drawKeypoints(gray2, kp2, im2)
            #cv2.imshow('a2', img2)
            #cv2.waitKey()
    else:
        tmp_persons.append(person)
    return person, old_persons, tmp_persons


def sift_contrib(person, other):
    # print("current person\n", person.sift_kp, len(person.sift_descriptors))
    bf = cv2.BFMatcher()
    # print("Other person\n", other.sift_kp, len(other.sift_descriptors))
    des1 = np.asarray(person.sift_descriptors)
    des2 = np.asarray(other.sift_descriptors)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    # if ho solo 1 non farlo
    for i in matches:
        if len(i) <= 1:
            return 0
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return len(good)


def direction_contrib(this, other):
    l_other = other.ground_point_past[0:P.NUMBER_OF_POINTS_CALC_GHOST]
    if len(l_other) >= 2:
        l = [this.ground_point]
        l.extend(l_other)
        other_vector = np.divide(np.subtract(l_other[0], l_other[-1]), len(l_other)-1)
        this_vector = np.divide(np.subtract(l[0], l[-1]), len(l)-1)
        # other_angle = np.arctan(other_vector[1])
        other_angle = np.arctan2(other_vector[1], other_vector[0])
        this_angle = np.arctan2(this_vector[1], this_vector[0])
        return min(np.abs(this_angle - other_angle), np.abs(other_angle - this_angle))

    else:
        print("return: LEN < 2")
        return 999 # 0


def match_likelihood_DICT(this, other):
    """
    :param this: person
    :param other:  person
    :return: likelihood value
    """
    # -- DICTIONARY --
    """
    likel_dict
        0: distance
        1: sift
        2: direction
        3: other.id 
    """
    likel_dict = np.zeros((4), dtype=np.float)
    likel_dict[3] = other.id

    # find points in birdeye view
    pt_this_z = from_camera_to_birdeye(np.float32([this.ground_point]))[0]
    pt_other_z = from_camera_to_birdeye(np.float32([other.ground_point]))[0]

    # -- distance
    likel_dict[0] = euclidean_distance(pt_this_z, pt_other_z)
    # -- sift
    likel_dict[1] = sift_contrib(this, other)              # return number of matches
    # -- direction
    likel_dict[2] = direction_contrib(this, other)    # return angular distance
    return likel_dict



'''
def match_likelihood(this, other):
    """
    :param this: person
    :param other:  person
    :return: likelihood value
    """
    # check thresholds
    pt_this_z = from_camera_to_birdeye(np.float32([this.ground_point]))[0]
    pt_other_z = from_camera_to_birdeye(np.float32([other.ground_point]))[0]
    dist = euclidean_distance(pt_this_z, pt_other_z)
    # print("Dist:  ", dist)
    if dist > P.LIKELIHOOD.DISTANCE_THS:
        return 0
    # -- SIFT
    matches = sift_contrib(this, other)
    # calculate likelihood as a function of distance, sift, color, ...
    contributions = []
    contributions.append( min(np.reciprocal(dist), np.finfo(np.float).max))  # distance
    contributions.append(matches*matches)
    contributions.append(direction_contrib(this, other))
    return np.sum(np.multiply( contributions, P.LIKELIHOOD.WEIGHTS ))
'''

def calc_ghost_point(p, mode='camera'):
    assert mode == 'camera' or mode == 'birdeye', "mode can only be 'camera' or 'birdeye'"

    if mode == 'birdeye':
        if len(p.ground_point_past) >= 2:
            last_pts = p.ground_point_past[p.ghost_detection_count: p.ghost_detection_count + P.NUMBER_OF_POINTS_CALC_GHOST]
            if len(last_pts) >= 2:
                last_pts = from_camera_to_birdeye(np.array(last_pts))
                new_point = np.add(p.ground_point, np.divide(np.subtract(last_pts[0], last_pts[-1]), len(last_pts) - 1))
                return new_point.astype(np.int)
        return from_camera_to_birdeye(np.reshape(p.ground_point, (1, 2)).astype(np.float32))

    else:   # if mode == 'camera':
        if len(p.ground_point_past) >= 2:
            last_pts = p.ground_point_past[p.ghost_detection_count: p.ghost_detection_count + P.NUMBER_OF_POINTS_CALC_GHOST]
            if len(last_pts) >= 2:
                pt = np.divide(np.subtract(last_pts[0], last_pts[-1]), len(last_pts) - 1)
                dist = np.sqrt(pt[0]**2 + pt[1]**2)
                if dist > P.MAX_DISTANCE_FOR_CALC_GHOST:
                    print("old PT: ", pt)
                    pt = pt * P.MAX_DISTANCE_FOR_CALC_GHOST / dist
                    print("scale: ", P.MAX_DISTANCE_FOR_CALC_GHOST / dist)
                    print("new PT: ", pt)
                return np.add(p.ground_point, pt).astype(np.int)
        return p.ground_point


def return_scores(likelihoods):
    idx_dist_min = np.argmin(likelihoods[:, 0])
    idx_sift_max = np.argmax(likelihoods[:, 1])
    idx_dir_min = np.argmin(likelihoods[:, 2])

    scoreboard = np.zeros_like((likelihoods), dtype=np.int)
    scoreboard[idx_dist_min, 0] = 1
    scoreboard[likelihoods[:, 0] >P.LIKELIHOOD.DISTANCE_THS, 0] = -100

    scoreboard[idx_sift_max, 1] = 1

    scoreboard[idx_dir_min, 2] = 1
    scoreboard[likelihoods[:, 2] > P.LIKELIHOOD.DIRECTION_THS , 2] = -1
    scoreboard[likelihoods[:, 2] == 999, 2] = 0

    ret_scores = np.sum(scoreboard, axis=1)
    idx = np.argmax(ret_scores)
    return idx, ret_scores[idx]



def update_persons_DICT(persons_detected, persons_old, max_used_id):
    persons_tmp = []
    if persons_old:     # forse inutile dato che ho gia fatto il controllo fuori dalla funzione
        if len(persons_detected) <= len(persons_old):
            remaining = persons_old.copy()
            for p in persons_detected:
                # print("person detected ", p.id)

                likelihoods = np.array( list(map(lambda x: match_likelihood_DICT(p, x), remaining)) )
                idx, score = return_scores(likelihoods)

                if score <= 0:
                    # no matches found
                    # add to person_old, tramite person_tmp
                    max_used_id += 1
                    p.id = max_used_id
                else:
                    person_matching = remaining.pop(idx)
                    # p.centroid_past.extend(person_matching.centroid_past)
                    p.update_past(person_matching.id, person_matching.centroid_past, person_matching.ground_point_past)
                    # p.id = person_matching.id
                persons_tmp.append(p)

            # prima di aggiornare person_tmp aggiungendo remaining...
            # io prendo questi "fantasmi" e aggiorno i valori, soprattuto il flag real_detection
            for p in remaining:
                if p.ghost_detection_count < P.MAX_GHOST_DETECTION:
                    p.ghost_detection_count += 1
                    new_ghost_point = calc_ghost_point(p)
                    p.follow_moving_ground_point(new_ghost_point)
                    persons_tmp.append(p)

        else:  # len(persons_detected) > len(persons_old):
            remaining = persons_detected
            for p in persons_old:
                print("person old ", p.id)
                likelihoods = np.array(list(map(lambda x: match_likelihood_DICT(x, p), remaining)))

                idx, score = return_scores(likelihoods)

                if score <= 0:
                    # no matches found
                    # la persona old e' USCITA oppure NASCOSTA
                    if p.ghost_detection_count < P.MAX_GHOST_DETECTION:
                        p.ghost_detection_count += 1
                        new_ghost_point = calc_ghost_point(p)
                        p.follow_moving_ground_point(new_ghost_point)
                        persons_tmp.append(p)
                else:
                    person_matching = remaining.pop(idx)
                    # person_matching.centroid_past.extend(p.centroid_past)
                    person_matching.update_past(p.id, p.centroid_past, p.ground_point_past)
                    # person_matching.id = p.id
                    persons_tmp.append(person_matching)

            for p in remaining:
                max_used_id += 1
                p.id = max_used_id
                persons_tmp.append(p)

    else:  # persons_old is empty
        for p in persons_detected:
            max_used_id += 1
            p.id = max_used_id
            persons_tmp.append(p)

    return persons_tmp, max_used_id

'''
def update_persons(persons_detected, persons_old, max_used_id):
    persons_tmp = []
    if persons_old:     # forse inutile dato che ho gia fatto il controllo fuori dalla funzione
        if len(persons_detected) <= len(persons_old):
            remaining = persons_old.copy()
            for p in persons_detected:
                # print("person detected ", p.id)
                likelihoods = list(map(lambda x: match_likelihood(p, x), remaining))
                if np.amax(likelihoods) <= 0:
                    # no matches found
                    # add to person_old, tramite person_tmp
                    max_used_id += 1
                    p.id = max_used_id
                else:
                    person_matching = remaining.pop(np.argmax(likelihoods))
                    # p.centroid_past.extend(person_matching.centroid_past)
                    p.update_past(person_matching.id, person_matching.centroid_past, person_matching.ground_point_past)
                    # p.id = person_matching.id
                persons_tmp.append(p)

            # prima di aggiornare person_tmp aggiungendo remaining...
            # io prendo questi "fantasmi" e aggiorno i valori, soprattuto il flag real_detection
            for p in remaining:
                if p.ghost_detection_count < P.MAX_GHOST_DETECTION:
                    p.ghost_detection_count += 1
                    new_ghost_point = calc_ghost_point(p)
                    p.follow_moving_ground_point(new_ghost_point)
                    persons_tmp.append(p)

        else:  # len(persons_detected) > len(persons_old):
            remaining = persons_detected
            for p in persons_old:
                print("person old ", p.id)
                likelihoods = list(map(lambda x: match_likelihood(x, p), remaining))
                if np.amax(likelihoods) <= 0:
                    # no matches found
                    # la persona old e' USCITA oppure NASCOSTA
                    # ORA la persona scomparsa non viene aggiunta a person_tmp, quindi l'id scompare.
                    # dopo dovremo fare una funzione per tenerla come possibile se e' NASCOSTA
                    # forse DIZIONARIO: a frame 1 ho tot persone, a frame 2 ho tot persone. magari intorno a 50 frrame o altro boh
                    # continue
                    persons_tmp.append(p)
                else:
                    person_matching = remaining.pop(np.argmax(likelihoods))
                    # person_matching.centroid_past.extend(p.centroid_past)
                    person_matching.update_past(p.id, p.centroid_past, p.ground_point_past)
                    # person_matching.id = p.id
                    persons_tmp.append(person_matching)

            for p in remaining:
                max_used_id += 1
                p.id = max_used_id
                persons_tmp.append(p)

    else:  # persons_old is empty
        for p in persons_detected:
            max_used_id += 1
            p.id = max_used_id
            persons_tmp.append(p)

    return persons_tmp, max_used_id
'''

def from_camera_to_birdeye(pts):
    """
    :param pts: list of 2D points
    :return:  np.array of mapped 2D points
    """
    return np.rint(cv2.perspectiveTransform(np.array([pts]), P.HOMOGRAPHY.MAT)[0])


def from_birdeye_to_camera(pts):
    """
    :param pts: list of 2D points
    :return:  np.array of mapped 2D points
    """
    return np.rint(cv2.perspectiveTransform(np.array([pts]), np.linalg.inv(P.HOMOGRAPHY.MAT))[0])


def load_undistortion_parameters():
    camera_matrix = np.load('./calibration_parameters/camera_matrix.npy')
    distortion_coefficients = np.load('./calibration_parameters/distortion_coefficients.npy')
    rotation_vectors = np.load('./calibration_parameters/rotation_vectors.npy')
    translation_vectors = np.load('./calibration_parameters/translation_vectors.npy')
    return camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors
