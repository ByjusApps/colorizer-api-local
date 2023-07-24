import cv2
import imutils

from consts import *


def _create_net(style):
    path = f"{DIRPATH_STYLES}/{style}.t7"
    net = cv2.dnn.readNetFromTorch(path)

    return net


def transfer_style(input_img, style):
    input_img = imutils.resize(input_img, width=RESIZE_WIDTH)
    net = _create_net(style)

    blob = cv2.dnn.blobFromImage(
        input_img,
        1.0,
        input_img.shape[:2][::-1],
        (103.939, 116.779, 123.680),
        swapRB=False,
        crop=False,
    )

    net.setInput(blob)
    output_img = net.forward()

    output_img = output_img.reshape((3, output_img.shape[2], output_img.shape[3]))
    output_img[0] += 103.939
    output_img[1] += 116.779
    output_img[2] += 123.680
    output_img = output_img.transpose(1, 2, 0)

    return output_img
