import torch
import cv2
from data import VideoDataLoader
from yolo_v3 import create_darknet_instance
from utils import *
#from person import *
import numpy as np
from person_old import PersonOLD, set_sift_keypoints
from parameters import Parameters as P
from datetime import datetime


"""
def follow_SIFT(person, old_persons, tmp_persons, img):
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
    else:
        tmp_persons.append(person)
    return person, old_persons, tmp_persons
"""


def draw_points_in_birdeye(z_img, persons):
    for p in persons:
        a = np.float32([p.ground_point])
        pt = from_camera_to_birdeye(a)[0]
        cv2.circle(z_img, (pt[0], pt[1]), 3, p.color, -1)
    cv2.imshow('track', z_img)
    cv2.waitKey(10)
    return z_img


# todo ONLY main function, others in utils
def main_matteo():
    # frame_extraction_from_video('./Videos/video1.mp4', 'Frames1')
    # mog()
    # col = ColorGenerator()

    CUDA = torch.cuda.is_available()
    if CUDA is True:
        Tensor = torch.cuda.FloatTensor
        device = torch.device(P.CUDA.DEVICE)
        IMG_SIZE = P.CUDA.IMG_SIZE
    else:
        Tensor = torch.FloatTensor
        device = torch.device(P.CPU.DEVICE)
        IMG_SIZE = P.CPU.IMG_SIZE

    net = create_darknet_instance(IMG_SIZE, device, P.DARKNET.CONF_THS, P.DARKNET.NMS_THS)
    # loader = VideoDataLoader('./Videos/vid2.mp4', IMG_SIZE)
    loader = VideoDataLoader('./VideosNewCam/v11.mp4', IMG_SIZE)

    colors = P.COLORS
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # writer = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    writer = cv2.VideoWriter('./output_new_cam_DIRECTION/output11.avi', fourcc, 15.0, (854, 480))  # !!! deve essere la stessa dim dell'input dell img !!!!

    persons_old = []
    max_used_id = 0
    startTime = datetime.now()

    z_img = cv2.imread('aligned.jpg')

    for idx, (img, torch_img) in enumerate(loader):
        if img is None or torch_img is None:
            continue
        print('\nFrame ', idx)
        torch_img = torch_img.type(Tensor).to(device)

        persons_detected = []

        detections = net.detect(torch_img)[0]
        if detections is not None:
            detections = detections[detections[:, -1] == 0.]
            detections = rescale_boxes(detections, IMG_SIZE, img.shape[:2])
            persons_detected = []
            tmp_new_id = 100
            for i, detection in enumerate(detections):
                person = PersonOLD(detection[:4].cpu().numpy())

                # person.id = max_used_id
                person.id = tmp_new_id
                tmp_new_id += 1

                sift_img = np.copy(img)
                person = set_sift_keypoints(sift_img, person)
                persons_detected.append(person)

            # print('Persons in the frame:', len(persons_detected))
            # print("Old person", len(persons_old))

            persons_old, max_used_id = update_persons_DICT(persons_detected, persons_old, max_used_id)
            # persons_old = persons_tmp    # solo per stampare

            for p in persons_old:
                p.draw_bounding_box_on_img(img)
                cv2.circle(img, (p.centroid[0].astype(np.int), p.centroid[1].astype(np.int)), 1, p.color, -1)
                cv2.circle(img, (p.ground_point[0], p.ground_point[1]), 3, p.color, -1)

        else:
            persons_old_tmp = []
            for p in persons_old:

                print("\nGROUND POINTS PAST\n", p.ground_point_past)

                p.ghost_detection_count += 1
                if p.ghost_detection_count < P.MAX_GHOST_DETECTION:
                    new_ghost_point = calc_ghost_point(p)

                    print("new ghost point: ", new_ghost_point)

                    p.follow_moving_ground_point(new_ghost_point)
                    p.draw_bounding_box_on_img(img)
                    cv2.circle(img, (p.centroid[0].astype(np.int), p.centroid[1].astype(np.int)), 1, p.color, -1)
                    cv2.circle(img, (p.ground_point[0], p.ground_point[1]), 3, p.color, -1)
                    persons_old_tmp.append(p)

            persons_old = persons_old_tmp

        z_img = draw_points_in_birdeye(z_img, persons_old)

        cv2.imshow('output', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        writer.write(img)


    print("\n\nTime taken:", datetime.now() - startTime, "\n")

    if isinstance(loader, VideoDataLoader):
        loader.close()
    writer.release()


'''
def main_matteo_old_CORRETTA():
    # frame_extraction_from_video('./Videos/video1.mp4', 'Frames1')
    # mog()
    # col = ColorGenerator()

    CUDA = torch.cuda.is_available()

    # CUDA = False

    if CUDA is True:
        Tensor = torch.cuda.FloatTensor
        device = torch.device(P.CUDA.DEVICE)
        IMG_SIZE = P.CUDA.IMG_SIZE
    else:
        Tensor = torch.FloatTensor
        device = torch.device(P.CPU.DEVICE)
        IMG_SIZE = P.CPU.IMG_SIZE

    net = create_darknet_instance(IMG_SIZE, device, P.DARKNET.CONF_THS, P.DARKNET.NMS_THS)

    loader = VideoDataLoader('./Videos/video1.mp4', IMG_SIZE)

    colors = P.COLORS
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('./output_OLD/output_video6.avi', fourcc, 20.0, (640, 480))

    # old_persons_exists = False
    old_persons = []
    startTime = datetime.now()


    for idx, (img, torch_img) in enumerate(loader):
        if img is None or torch_img is None:
            continue
        print('Frame ', idx)
        torch_img = torch_img.type(Tensor).to(device)

        detections = net.detect(torch_img)[0]
        if detections is not None:
            detections = detections[detections[:, -1] == 0.]
            detections = rescale_boxes(detections, IMG_SIZE, img.shape[:2])

            tmp_persons = []
            id_per_frame = 0
            for i, detection in enumerate(detections):
                person = PersonOLD(detection[:4].cpu().numpy(), colors[i])
                person.id = id_per_frame
                sift_img = np.copy(img)
                person = set_sift_keypoints(sift_img, person)

                person.centroid_past.append(person.centroid)
                id_per_frame += 1

                # ---- NUOVA FUNZIONE per trackare old persons
                # a = predict_centroid(old_persons)

                # person, old_persons, tmp_persons = follow_old_person(person, old_persons, tmp_persons)
                im = np.copy(img)
                person, old_persons, tmp_persons = follow_SIFT(person, old_persons, tmp_persons, im)

                person.draw_bounding_box_on_img(img)
                print(person.id)
                cv2.circle(img, (person.centroid[0].astype(np.int), person.centroid[1].astype(np.int)), 1, person.color, -1)

                # -- DRAW FUTURE CENTROID
                for i in old_persons:
                    if i.centroid_future != (0, 0):
                        cv2.circle(img, (i.centroid_future[0].astype(np.int), i.centroid_future[1].astype(np.int)), 1, [0,255,255], -1)

            if tmp_persons:
                old_persons.extend(tmp_persons)
        else:
            # no detection
            print("NO DETECTION")

        cv2.imshow('output', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print("\n\nTime taken:", datetime.now() - startTime, "\n")

        writer.write(img)


    print('a')
    tracking_centroid(old_persons, img)
    print('a')
    print("Time taken:", datetime.now() - startTime, "\n\n\n")
    if isinstance(loader, VideoDataLoader):
        loader.close()
    writer.release()

if __name__ == '__main__':
    #main_matteo_old_CORRETTA()
    main_matteo()

'''
