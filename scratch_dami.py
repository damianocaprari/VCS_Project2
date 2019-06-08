import torch
import cv2
import numpy as np

#from data import VideoDataLoader
#from yolo_v3 import create_darknet_instance
#from utils import rescale_boxes
#from person import Person
from person_old import PersonOLD

#from utils import from_camera_to_birdeye
#from parameters import Parameters as P


# todo ONLY main function, others in utils

# USED TO CREATE THE MINIMAP

def get_birdeye(im1):
    minimap_pts = [
        [0, 0], #TR
        [1045, 0], #TL
        [1045, 930], #BR
        [222, 705] #BL grass
    ]

    target_size = (1800, 1800)
    displacement = (1500, 1250)
    delta = (target_size[0] - displacement[0], target_size[1] - displacement[1])

    minimap_grid_pts = [
        [minimap_pts[0][0] + delta[0], minimap_pts[0][1] + delta[1]],
        [minimap_pts[1][0] + delta[0], minimap_pts[1][1] + delta[1]],
        [minimap_pts[2][0] + delta[0], minimap_pts[2][1] + delta[1]],
        [minimap_pts[3][0] + delta[0], minimap_pts[3][1] + delta[1]]
    ]
    perspective_pts = [
        [390, 345],
        [1305, 225],
        [1225, 700],
        [270, 615]
    ]

    # Extract location of good matches
    minimap_pts = np.array(minimap_grid_pts, dtype=np.float32)
    perspective_pts = np.array(perspective_pts, dtype=np.float32)

    # Find homography
    h, mask = cv2.findHomography(perspective_pts, minimap_pts)

    # Use homography
    im1Reg = cv2.warpPerspective(im1, h, target_size)

    return im1Reg, h



def main_dami_minimap():
    # Read image to be aligned
    imFilename = "img0.png"
    print("Reading image to align : ", imFilename)
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = get_birdeye(im)

    # Write aligned image to disk.
    outFilename = "aligned.jpg"
    print("Saving aligned image : ", outFilename)
    cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography : \n", h)


"""
main_prove_con_perspectiveTransform():
target_size = (1000, 1000)
    dst = np.array([
        [510 + (target_size[0] - 1100), 50 + (target_size[1] - 750)],
        [900 + (target_size[0] - 1100), 150 + (target_size[1] - 750)],
        [1020 + (target_size[0] - 1100), 500 + (target_size[1] - 750)],
        [610 + (target_size[0] - 1100), 390 + (target_size[1] - 750)]
    ], dtype=np.float32)

    src = np.array([
        [243, 60],
        [531, 168],
        [551, 381],
        [132, 167]
    ], dtype=np.float32)


    M = cv2.getPerspectiveTransform(src, dst)
    ret = cv2.perspectiveTransform(np.array([src]), P.HOMOGRAPHY.MAT)

    print(dst == map_points_onto_minimap(src))
"""

"""
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
"""


if __name__ == '__main__':
    main_dami_minimap()
    #main_dami()


