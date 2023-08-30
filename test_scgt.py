from scgt import *
import random

def test_get_pixel():
    gt = GeoTiff.from_file("test_files/test_Psilopogon_rafflesii_300.tif")
    for _ in range(1000):
        x = random.randint(0, gt.width)
        y = random.randint(0, gt.height)
        p1 = gt.get_pixel_value(x, y)
        p2 = gt.get_slow_pixel_value(x, y)
        assert p1 == p2
