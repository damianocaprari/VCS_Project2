import torch
import cv2

from data import VideoDataLoader
from yolo_v3 import create_darknet_instance
from utils import rescale_boxes
from person import Person

from parameters import Parameters as P


# todo ONLY main function, others in utils


def main_marco():
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

    loader = VideoDataLoader('./Videos/video1.mp4', IMG_SIZE)
    fourcc = cv2.VideoWriter_fourcc(*P.VIDEOWRITER.FORMAT)
    writer = cv2.VideoWriter('output.avi', fourcc, P.VIDEOWRITER.FPS, P.VIDEOWRITER.SIZE)

    colors = P.COLORS
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


if __name__ == '__main__':
    main_marco()