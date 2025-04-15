# Reference implementation:
# https://towardsdatascience.com/simple-steps-to-create-custom-colormaps-in-python-f21482778aa2 (Accessed 16.05.2022)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors


def inter_from_256(x):
    return np.interp(x=x, xp=[0, 255], fp=[0, 1])

class MyColormap():
    cdict = {
        'red': ((0.0, inter_from_256(64), inter_from_256(64)),
                (1 / 5 * 1, inter_from_256(112), inter_from_256(112)),
                (1 / 5 * 2, inter_from_256(230), inter_from_256(230)),
                (1 / 5 * 3, inter_from_256(253), inter_from_256(253)),
                (1 / 5 * 4, inter_from_256(244), inter_from_256(244)),
                (1.0, inter_from_256(169), inter_from_256(169))),
        'green': ((0.0, inter_from_256(57), inter_from_256(57)),
                  (1 / 5 * 1, inter_from_256(198), inter_from_256(198)),
                  (1 / 5 * 2, inter_from_256(241), inter_from_256(241)),
                  (1 / 5 * 3, inter_from_256(219), inter_from_256(219)),
                  (1 / 5 * 4, inter_from_256(109), inter_from_256(109)),
                  (1.0, inter_from_256(23), inter_from_256(23))),
        'blue': ((0.0, inter_from_256(144), inter_from_256(144)),
                 (1 / 5 * 1, inter_from_256(162), inter_from_256(162)),
                 (1 / 5 * 2, inter_from_256(246), inter_from_256(146)),
                 (1 / 5 * 3, inter_from_256(127), inter_from_256(127)),
                 (1 / 5 * 4, inter_from_256(69), inter_from_256(69)),
                 (1.0, inter_from_256(69), inter_from_256(69))),
    }
    cmap = colors.LinearSegmentedColormap('new_cmap', segmentdata=cdict)

    @staticmethod
    def get_color_from_value(value):
        return MyColormap.cmap(value)