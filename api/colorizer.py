import cv2
import imutils
import numpy as np

from consts import *


def _create_net():
    net = cv2.dnn.readNetFromCaffe(PATH_PROTOTXT, PATH_CAFFE_MODEL)
    pts = np.load(PATH_PTS_NPY)

    layer1 = net.getLayerId("class8_ab")
    layer2 = net.getLayerId("conv8_313_rh")

    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(layer1).blobs = [pts.astype("float32")]
    net.getLayer(layer2).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    return net


def colorize(input_img):
    input_img = imutils.resize(input_img, width=RESIZE_WIDTH)
    net = _create_net()

    normalized = input_img.astype("float32") / 255.0
    lab_image = cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB)
    resized = cv2.resize(lab_image, (224, 224))
    l = cv2.split(resized)[0] - 50

    net.setInput(cv2.dnn.blobFromImage(l))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (input_img.shape[1], input_img.shape[0]))

    l = cv2.split(lab_image)[0]
    LAB_colored = np.concatenate((l[:, :, np.newaxis], ab), axis=2)

    output_img = cv2.cvtColor(LAB_colored, cv2.COLOR_LAB2BGR)
    output_img = np.clip(output_img, 0, 1)
    output_img = (255 * output_img).astype("uint8")

    return output_img
