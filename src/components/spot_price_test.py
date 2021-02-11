import numpy as np


def get_spot_price_test():
    prices = np.asarray(
        [
            0.447,
            0.424,
            0.406,
            0.412,
            0.428,
            0.472,
            0.548,
            0.703,
            0.876,
            0.870,
            0.855,
            0.892,
            0.878,
            0.853,
            0.815,
            0.846,
            0.930,
            1.036,
            0.878,
            0.807,
            0.671,
            0.605,
            0.6,
            0.525,
        ]
    )
    return np.repeat(prices, 6)
