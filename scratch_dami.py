import torch
import cv2
import numpy as np

from data import VideoDataLoader
from yolo_v3 import create_darknet_instance
from utils import rescale_boxes
#from person import Person
from person_old import PersonOLD

from parameters import Parameters as P


# todo ONLY main function, others in utils

def alignImages(im1, im2):

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    minimap_grid_pts = [
        [250, 125],
        [660, 236],
        [771, 588],
        [359, 476]
    ]

    minimap_pts = [
        [383, 209],
        [955, 364],
        [1109, 855],
        [536, 699]
    ]

    perspective_pts = [
        [243, 60],
        [531, 168],
        [551, 381],
        [132, 167]
    ]

    # Extract location of good matches
    minimap_pts = np.array(minimap_grid_pts, dtype=np.float32)
    perspective_pts = np.array(perspective_pts, dtype=np.float32)

    # Find homography
    h, mask = cv2.findHomography(perspective_pts, minimap_pts)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


def main_dami_minimap():
    # Read reference image
    refFilename = "cvcs02_minimap_grid.jpeg"
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = "Frames1/frame0.jpg"
    print("Reading image to align : ", imFilename)
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im, imReference)

    # Write aligned image to disk.
    outFilename = "aligned.jpg"
    print("Saving aligned image : ", outFilename)
    cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography : \n", h)


def main_dami():
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
                person = PersonOLD(detection[:4].cpu().numpy(), colors[i])
                person.draw_bounding_box_on_img(img)
        writer.write(img)

    if isinstance(loader, VideoDataLoader):
        loader.close()
    writer.release()


if __name__ == '__main__':
    main_dami_minimap()
    # main_dami()
