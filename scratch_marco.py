import torch
import cv2

from data import VideoDataLoader
from yolo_v3 import create_darknet_instance
from utils import rescale_boxes
from person import Person


# todo ONLY main function, others in utils


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

    net = create_darknet_instance(IMG_SIZE, device, 0.8, 0.4)

    loader = VideoDataLoader('./Videos/video1.mp4', IMG_SIZE)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    for idx, (img, torch_img) in enumerate(loader):
        if img is None or torch_img is None:
            continue
        print('Frame ', idx)
        torch_img = torch_img.type(Tensor).to(device)

        detections = net.detect(torch_img)[0]
        if detections is not None:
            detections = detections[detections[:, -1] == 0.]
            detections = rescale_boxes(detections, IMG_SIZE, img.shape[:2])
            for i, detection in enumerate(detections):
                person = Person(detection[:4].cpu().numpy(), colors[i])
                person.draw_bounding_box_on_img(img)
        writer.write(img)

    if isinstance(loader, VideoDataLoader):
        loader.close()
    writer.release()
