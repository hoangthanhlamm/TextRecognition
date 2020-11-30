import requests
import numpy as np
import pandas as pd
import random
import json
import logging
import time

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def test_predict(n_tests):
    url = "http://localhost:8096/prediction"

    data = pd.read_csv('Data/test_final.csv')
    data = data.to_numpy()
    data_len = data.shape[0]
    times = []
    for i in range(n_tests):
        time.sleep(1)
        idx = random.randrange(0, data_len)
        path = data[idx, 0]

        body = {
            "img": path
        }
        try:
            response = requests.post(url, json=body)
            response = json.loads(response.content.decode())
            print("Label: {label} - Time: {time}".format(label=response['predicted'], time=response['time']))
            times.append(response.get('time'))
        except Exception as err:
            LOGGER.exception(err)

    times = np.array(times)
    mean = times.mean()
    std = times.std()

    print("Mean of Prediction's time: ", mean)
    print("Standard Deviation of Prediction's time: ", std)

    return mean, std


if __name__ == '__main__':
    test_predict(n_tests=100)
