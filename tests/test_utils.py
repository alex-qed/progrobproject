"""
Unit-tests for key helper functions of Gesture-Spot
"""

import cv2
import math
import pytest

from src.gesture.ptd import Med, dead, clipmap


# ------------------ 1. camera availability ------------------
@pytest.mark.skipif(not cv2.VideoCapture(0).isOpened(),
                    reason="Web-camera not detected on this system")
def test_camera_open():
    """Проверяем, что первая камера действительно открывается."""
    cap = cv2.VideoCapture(0)
    opened = cap.isOpened()
    cap.release()
    assert opened


# ------------------ 2. median filter ------------------------
def test_median_filter_window5():
    """
    На последовательности из пяти чисел фильтр должен
    вернуть медиану последних 5 элементов.
    """
    mf = Med(5)
    seq = [1, 100, 2, 3, 4]    # медиана = 3
    for x in seq:
        med = mf(x)
    assert med == 3


# ------------------ 3. dead-zone logic ----------------------
@pytest.mark.parametrize("prev,new,th,expected", [
    (10.0, 11.5, 2.0, 10.0),   # внутри dead-zone → не меняется
    (10.0, 13.1, 2.0, 13.1),   # вне зоны → обновляется
])
def test_dead_zone(prev, new, th, expected):
    assert math.isclose(dead(prev, new, th), expected, rel_tol=1e-6)


# ------------------ 4. clip_map linearity -------------------
def test_clip_map_linear():
    """
    Проверяем, что clip_map расставляет
    значения 0.25 / 0.5 / 0.75 ровно в середины диапазона.
    """
    a, b = 0.0, 100.0
    c, d = -50.0, 50.0
    quarter = clipmap(25, a, b, c, d)
    mid     = clipmap(50, a, b, c, d)
    three_q = clipmap(75, a, b, c, d)
    assert math.isclose(quarter, -25.0, rel_tol=1e-6)
    assert math.isclose(mid,      0.0, rel_tol=1e-6)
    assert math.isclose(three_q, 25.0, rel_tol=1e-6)