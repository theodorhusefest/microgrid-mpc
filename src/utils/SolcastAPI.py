import requests
import json
from dateutil import parser

# Need to insert API-key
# 30min prediction intervalls for one week ahead.
def get_solcast_data(api_key):
    response = requests.get(
        "https://api.solcast.com.au/weather_sites/60ea-e64d-04f9-41db/forecasts?format=json&api_key={}".format(
            api_key
        ),
        headers={
            "User-Agent": "theodoth@stud.ntnu.no",
            "content-type": "text",
        },
    ).text

    data = json.loads(response)
    GHI = []
    end_time = []
    temperature = []

    print(response)
    for x in data["forecasts"]:
        print(x.keys())
        GHI.append(x["ghi"])
        end_time.append(parser.parse(x["period_end"], ignoretz=True))
        temperature.append(x["air_temp"])

    return GHI, temperature, end_time


get_solcast_data("35E054Xn3h8OhxAYDvPRDWzB3_0t2KVw")