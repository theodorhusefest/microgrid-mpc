import requests
import json

# Need to insert API-key
# 30min prediction intervalls for one week ahead.
def get_solcast_data():
    response = requests.get(
        "https://api.solcast.com.au/weather_sites/0ae3-fb0a-96fc-c673/forecasts?format=json&api_key=API-NØKKEL-HER",
        headers={
            "User-Agent": "nicolho@stud.ntnu.no",
            "content-type": "text",
        },
    ).text

    data = json.loads(response)
    GHI = []
    end_time = []
    temperature = []

    for x in data["forecasts"]:
        GHI.append(x["ghi"])
        end_time.append(x["period_end"])
        temperature.append(x["air_temp"])

    return GHI, temperature, end_time
