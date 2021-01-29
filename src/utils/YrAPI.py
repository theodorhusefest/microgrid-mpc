import requests
import json

# Hourly data for the first two days.
# Two days into the future the predicitons are for each 6h intervall.
def get_yr_data():
    response = requests.get(
        "https://api.met.no/weatherapi/locationforecast/2.0/compact?lat=59.233033&lon=9.263709&altitude=14",
        headers={
            "User-Agent": "NicoAndTheo",
            "content-type": "text",
        },
    ).text

    data = json.loads(response)
    temperature = []
    wind_speed = []
    wind_from_direction = []

    for x in data["properties"]["timeseries"]:
        temperature.append(x["data"]["instant"]["details"]["air_temperature"])
        wind_speed.append(x["data"]["instant"]["details"]["wind_speed"])
        wind_from_direction.append(
            x["data"]["instant"]["details"]["wind_from_direction"]
        )

    return temperature, wind_speed, wind_from_direction
