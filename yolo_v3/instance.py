from .models import Darknet


def create_darknet_instance(img_size, device, conf_ths, nms_ths):
    # Load the net and configure it with predefined model and weights
    net = Darknet('./yolo_v3/config/yolov3.cfg', img_size).to(device)
    net.load_darknet_weights('./yolo_v3/weights/yolov3.weights')
    net.load_classes('./yolo_v3/data/coco.names')
    net.conf_ths = conf_ths
    net.nms_ths = nms_ths

    # We don't need to train the network so we set it in evaluation mode
    net.eval()

    return net
