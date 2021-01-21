import requests
import pandas as pd


class YRApi:
    def __init__(self):
        self.lat = 59.2114
        self.lon = 9.5900

        self.headers = {"User-Agent": "microgrid-mpc"}

    def request_forecast(self):
        """
        Gets complete forecast
        Raises error if request fails
        """
        request_url = "https://api.met.no/weatherapi/locationforecast/2.0/complete.json?lat={}&lon={}".format(
            self.lat, self.lon
        )
        r = requests.get(request_url)
        if r.status_code != 200:
            print("Status Code: {}".format(r.status_code))
            raise ValueError

        return r

    def parse_forcast(self, r):
        """
        Cleans and parses the forecasts
        """
        df = pd.read_json(r.text)
        ts = (df[df.index == "timeseries"]).drop(["type", "geometry"], axis=1)
        ts = pd.json_normalize(ts.explode("properties")["properties"])
        print(ts.shape)
        print(ts.columns)
        print(ts["data.instant.details.wind_speed"])


if __name__ == "__main__":
    yr = YRApi()
    r = yr.request_forecast()
    yr.parse_forcast(r)
