import numpy as np
from nordpool import elspot, elbas
from pprint import pprint


def get_spot_price():
    prices_spot = elspot.Prices(currency="NOK")
    prices_area = prices_spot.hourly(areas=["Oslo"])

    prices = []
    for item in prices_area["areas"]["Oslo"]["values"]:
        # print(item["end"].strftime("%d.%m.%Y-%H"))
        prices.append(item["value"])

    return prices


if __name__ == "__main__":
    get_spot_price()