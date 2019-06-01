import torch
import cv2

from data import VideoDataLoader, ImageFolderDataLoader
from yolo_v3 import create_darknet_instance
from utils import rescale_boxes
from person import Person

from sort import SORT

from datetime import datetime


def analyse_detections(detections, tracker, img, img_size):
    if detections is not None:
        detections = detections[detections[:, -1] == 0.]
        detections = rescale_boxes(detections, img_size, img.shape[:2])
        d1 = detections[:, :4]
        d2 = detections[:, 5].view(-1, 1)
        detections = torch.cat((d1, d2), dim=1)
        sure_trks = tracker.update(detections.numpy())
        for sure_trks_idx in sure_trks:
            person = tracker.trackers[sure_trks_idx]
            person.draw_bounding_box_on_img(img)
    return img


def main_marco():
    CUDA = torch.cuda.is_available()
    if CUDA is True:
        Tensor = torch.cuda.FloatTensor
        device = torch.device('cuda:0')
        IMG_SIZE = 416
    else:
        Tensor = torch.FloatTensor
        device = torch.device('cpu')
        IMG_SIZE = 160

    startTime = datetime.now()

    net = create_darknet_instance(IMG_SIZE, device, 0.8, 0.4)

    loader = VideoDataLoader('./Videos/vid2.avi', IMG_SIZE)
    tracker = SORT(Person, max_age=10, min_hits=3)

    out_folder = 'output/'

    for idx, (img, torch_img) in enumerate(loader):
        print(idx)
        img = cv2.resize(img, (640, 480))
        if img is None or torch_img is None:
            continue
        torch_img = torch_img.type(Tensor).to(device)
        detections = net.detect(torch_img)[0]
        img = analyse_detections(detections, tracker, img, IMG_SIZE)
        out_file = out_folder + str(idx) + '.jpg'
        cv2.imwrite(out_file, img)

    if isinstance(loader, VideoDataLoader):
        loader.close()
    writer.release()
    print("\n\nTime taken:", datetime.now() - startTime, "\n")
