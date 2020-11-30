from aiohttp.web import json_response
from json.decoder import JSONDecodeError

import logging
from datetime import datetime

from train import train
from evaluation import evaluation
from predict import predict
from utils.errors import ApiBadRequest
from net.CRNN import text_recognition_model

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


class RouterHandler(object):
    def __init__(self, loop):
        self._loop = loop
        self.model = text_recognition_model('predict')
        self.model.load_weights('checkpoint/model.weights.hdf5')

    async def train(self, request):
        start = datetime.now()
        train(self.model)
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

        accuracy, letter_accuracy = evaluation(self.model, **body)
        end = datetime.now()
        time = end - start

        if accuracy is None:
            return json_response({
                "status": "Fail",
                "detail": "File not found"
            })

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
        predicted = predict(self.model, img)
        end = datetime.now()
        time = end - start

        if predicted is None:
            return json_response({
                "status": "Fail",
                "time": time.total_seconds()
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
