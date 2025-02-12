from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def get_default_cmaps():
    cmaps = [CustomCmap([1, 1, 1], x) for x in
             [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [255 / 255, 165 / 255, 0 / 255],
              [74 / 255, 4 / 255, 4 / 255],[72 / 255, 60 / 255, 50 / 255] , [255 / 255, 165 / 255, 0 / 255],[205 / 255, 127 / 255, 50 / 255]]]
    colormaps = [convert_cmap_to_cv(x) for x in cmaps]

    return colormaps


def CustomCmap(from_rgb, to_rgb):
    # from color r,g,b
    r1, g1, b1 = from_rgb

    # to color r,g,b
    r2, g2, b2 = to_rgb

    cdict = {'red': ((0, r1, r1),
                     (1, r2, r2)),
             'green': ((0, g1, g1),
                       (1, g2, g2)),
             'blue': ((0, b1, b1),
                      (1, b2, b2))}

    cmap = LinearSegmentedColormap('custom_cmap', cdict)
    return cmap

def convert_cmap_to_cv(cmap, log=True, gamma=1.5):
    sm = plt.cm.ScalarMappable(cmap=cmap)
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
    if log:
        color_range = (color_range ** gamma) / (255 ** gamma)
        color_range = np.uint8(color_range * 255)

    return color_range.reshape(256, 1, 3)
