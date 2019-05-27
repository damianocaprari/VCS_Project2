import torch
import cv2
from data import VideoDataLoader
from yolo_v3 import create_darknet_instance
from utils import *
from person import *
import numpy as np
from parameters import Parameters as P


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

    loader = VideoDataLoader('./Videos/video3.mp4', IMG_SIZE)

    colors = P.COLORS
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    # old_persons_exists = False
    persons_old = []

    max_used_id = 0

    for idx, (img, torch_img) in enumerate(loader):
        if img is None or torch_img is None:
            continue
        print('Frame ', idx)
        torch_img = torch_img.type(Tensor).to(device)

        detections = net.detect(torch_img)[0]
        if detections is not None:
            detections = detections[detections[:, -1] == 0.]
            detections = rescale_boxes(detections, IMG_SIZE, img.shape[:2])

            # tmp_persons = []
            persons_new = []

            for i, detection in enumerate(detections):
                person = Person(detection[:4].cpu().numpy(), colors[i])
                # max_used_id  += 1
                persons_new.append(person)
                '''
                # -----
                person.centroid_past.append(person.centroid)
                # -----
                # ---- NUOVA FUNZIONE per trackare old persons
                a = predict_centroid(old_persons)
                person, old_persons, tmp_persons = follow_old_person(person, old_persons, tmp_persons)
                person.draw_bounding_box_on_img(img)
                print(person.id)
                cv2.circle(img, (person.centroid[0].astype(np.int), person.centroid[1].astype(np.int)), 1, person.color, -1)
                # -------------------------
                for i in old_persons:
                    if i.centroid_future != (0, 0):
                        cv2.circle(img, (i.centroid_future[0].astype(np.int), i.centroid_future[1].astype(np.int)), 1, [0,255,255], -1)
            if tmp_persons:
                old_persons.extend(tmp_persons)
                # old_persons_exists = True
            '''

            print('Persons in the frame:', len(persons_new))
            print("Old person", len(persons_old))
            persons_old, max_used_id = update_persons(persons_new, persons_old, max_used_id)

            for p in persons_old:
                p.draw_bounding_box_on_img(img)
                print(p.id)
                cv2.circle(img, (p.centroid[0].astype(np.int), p.centroid[1].astype(np.int)), 1, p.color, -1)

        cv2.imshow('output', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #cv2.waitKey(10)

        writer.write(img)

    else:
        # no detection
        print("NO DETECTION")

    print('a')
    #tracking_centroid(persons_old, img)
    print('a')

    if isinstance(loader, VideoDataLoader):
        loader.close()
    writer.release()



def main_matteo_old_CORRETTA():
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

    loader = VideoDataLoader('./Videos/video3.mp4', IMG_SIZE)

    colors = P.COLORS
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    # old_persons_exists = False
    old_persons = []

    for idx, (img, torch_img) in enumerate(loader):
        if img is None or torch_img is None:
            continue
        print(img.shape)
        print('Frame ', idx)
        torch_img = torch_img.type(Tensor).to(device)

        detections = net.detect(torch_img)[0]
        if detections is not None:
            detections = detections[detections[:, -1] == 0.]
            detections = rescale_boxes(detections, IMG_SIZE, img.shape[:2])

            tmp_persons = []
            id_per_frame = 0
            for i, detection in enumerate(detections):
                person = Person(detection[:4].cpu().numpy(), colors[i])
                person.id = id_per_frame
                # -----
                person.centroid_past.append(person.centroid)
                # -----
                id_per_frame += 1

                # ---- NUOVA FUNZIONE per trackare old persons
                a = predict_centroid(old_persons)

                person, old_persons, tmp_persons = follow_old_person(person, old_persons, tmp_persons)

                person.draw_bounding_box_on_img(img)
                print(person.id)
                cv2.circle(img, (person.centroid[0].astype(np.int), person.centroid[1].astype(np.int)), 1, person.color, -1)

                # -------------------------
                for i in old_persons:
                    if i.centroid_future != (0, 0):
                        cv2.circle(img, (i.centroid_future[0].astype(np.int), i.centroid_future[1].astype(np.int)), 1, [0,255,255], -1)

            if tmp_persons:
                old_persons.extend(tmp_persons)
                # old_persons_exists = True

        cv2.imshow('output', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #cv2.waitKey(10)

        writer.write(img)

    else:
        # no detection
        print("NO DETECTION")

    print('a')
    tracking_centroid(old_persons, img)
    print('a')

    if isinstance(loader, VideoDataLoader):
        loader.close()
    writer.release()


if __name__ == '__main__':
    main_matteo()

