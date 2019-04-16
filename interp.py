import numpy as np
import PIL.Image
import scipy.misc


def get_item(arr, *args):
    indexes = []
    for i, entry in enumerate(args):
        index = entry
        if index < 0:
            index = abs(index) - 1
        if index >= arr.shape[i]:
            index = arr.shape[i] - index % arr.shape[i] - 1
        indexes.append(index)
    r = arr
    for index in indexes:
        r = r[index]
    return r


def get_w(x):
    a = -0.5
    absx = abs(x)
    if absx <= 1:
        return (a + 2) * absx**3 - (a + 3) * absx ** 2 + 1
    elif 1 < absx < 2:
        return a * absx**3 - 5 * a * absx**2 + 8 * a * absx - 4 * a
    else:
        return 0