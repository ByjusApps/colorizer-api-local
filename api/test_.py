import unittest

import cv2
import matplotlib.pyplot as plt
from colorizer import colorize
from style_transferer import transfer_style

DIRPATH = f".tests_data"


def _test_colorize(file_number):
    path = f"{DIRPATH}/colorize/{file_number}.jpg"
    img = colorize(cv2.imread(path))
    plt.title(f"{path}")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def _test_transfer_style(file_number, style):
    path = f"{DIRPATH}/style_transfer/{file_number}.jpg"
    img = transfer_style(cv2.imread(path), style)
    plt.title(f"{path} - {style}")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0)
    plt.show()


class TestColorize(unittest.TestCase):
    def test_img1(self):
        _test_colorize(1)

    def test_img2(self):
        _test_colorize(2)

    def test_img3(self):
        _test_colorize(3)

    def test_img4(self):
        _test_colorize(4)


class TestStyleTransferer(unittest.TestCase):
    def test_img1_style1(self):
        _test_transfer_style(1, "composition_vii")

    def test_img1_style2(self):
        _test_transfer_style(1, "la_muse")

    def test_img1_style3(self):
        _test_transfer_style(1, "the_wave")

    def test_img2_style1(self):
        _test_transfer_style(2, "composition_vii")

    def test_img2_style2(self):
        _test_transfer_style(2, "la_muse")

    def test_img2_style3(self):
        _test_transfer_style(2, "the_wave")


if __name__ == "__main__":
    unittest.main()
