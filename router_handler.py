from aiohttp.web import json_response
from json.decoder import JSONDecodeError
import pandas as pd

import logging
from datetime import datetime

from lib.utils.errors import ApiBadRequest
from lib.models.CRNNModel import CRNNModel
from config import *

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


class RouterHandler(object):
    def __init__(self, loop):
        self._loop = loop
        self.model = CRNNModel(model_path=model_path, initial_state=False)

    async def train(self, request):
        start = datetime.now()
        self.model.build_model('train')
        self.model.fit(epochs=epochs)
        end = datetime.now()
        return json_response({
            "status": "Success",
            "time": str(end - start)
        })

    async def evaluation(self, request):
        start = datetime.now()
        body = await decode_request(request)
        able_fields = ['filename', 'paths', 'labels']
        body = filter_fields(able_fields, body)

        filename = test_path
        if body.get('filename') is not None:
            filename = body.get('filename')

        try:
            data = pd.read_csv(filename)
        except FileNotFoundError as err:
            logging.exception(err)
            return json_response({
                "status": "Fail",
                "detail": "File not found"
            })

        paths = body.get('paths')
        labels = body.get('labels')

        if paths is None or labels is None:
            paths = data['Image'].values.tolist()
            labels = data['Label'].values.tolist()

        accuracy, letter_accuracy = self.model.evaluate(paths, labels)
        end = datetime.now()
        time = end - start

        return json_response({
            "status": "Success",
            "accuracy": accuracy,
            "letter_accuracy": letter_accuracy,
            "time": time.total_seconds()
        })

    async def prediction(self, request):
        start = datetime.now()
        body = await decode_request(request)
        required_fields = ['img']
        validate_fields(required_fields, body)

        img = body.get('img')
        predicted = self.model.predict(img)
        end = datetime.now()
        time = end - start

        if predicted is None:
            return json_response({
                "status": "Fail",
                "detail": "Image not found"
            })

        return json_response({
            "status": "Success",
            "predicted": predicted,
            "time": time.total_seconds()
        })


async def decode_request(request):
    try:
        return await request.json()
    except JSONDecodeError:
        raise ApiBadRequest('Improper JSON format')


def validate_fields(required_fields, body):
    for field in required_fields:
        if body.get(field) is None:
            raise ApiBadRequest("'{}' parameter is required".format(field))


def filter_fields(able_fields, body):
    result = {}
    for field in able_fields:
        if body.get(field):
            result[field] = body.get(field)
    return result
