class Parameters:

    class CUDA:
        DEVICE = 'cuda:0'
        IMG_SIZE = 416

    class CPU:
        DEVICE = 'cpu'
        IMG_SIZE = 160

    class DARKNET:
        CONF_THS = 0.8
        NMS_THS = 0.4

    class VIDEOWRITER:
        FORMAT = 'XVID'
        FPS = 20.0
        SIZE = (640, 480)
        COLOR = True

    COLORS = [[255, 0, 0],
              [0, 255, 0],
              [0, 0, 255],
              [0, 255, 255],
              [255, 0, 255],
              [255, 255, 0],
              [255, 255, 255],
              [0, 0, 0]]

    # TODO altri



