import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from nordpool import elspot
from pprint import pprint


def get_spot_price(end_date=None):
    """
    Uses the python-package nordpool to get spotprices for Oslo area in NOK.
    Returns a df with date and price
    """
    prices_spot = elspot.Prices(currency="NOK")
    prices_area = prices_spot.hourly(end_date=end_date, areas=["Oslo"])
    dates = []
    prices = []
    for item in prices_area["areas"]["Oslo"]["values"]:
        dates.append(item["end"])
        prices.append(item["value"])

    return pd.DataFrame(data={"time": dates, "price": prices})


def get_historic_spotprices(start, stop=None):
    """
    Gets hourly starting at the datetime object - start.
    """
    spot_prices = pd.DataFrame(columns=["time", "price"])
    current_time = start.date()
    while current_time:
        spot = get_spot_price(current_time)
        spot_prices = spot_prices.append(spot)
        current_time += timedelta(days=1)
        if current_time >= datetime.now().date():
            current_time = None
        elif stop and (current_time >= datetime.now().date()):
            current_time = None

    return spot_prices


if __name__ == "__main__":
    spot_prices = get_historic_spotprices(datetime(2021, 3, 17))

    spot_prices.to_csv("./spot_prices.csv")
