from aiohttp.web import json_response
from json.decoder import JSONDecodeError

import logging

from train import train
from evaluation import evaluation
from predict import predict
from utils.errors import ApiBadRequest

LOGGER = logging.getLogger(__name__)


class RouterHandler(object):
    def __init__(self, loop):
        self._loop = loop

    async def train(self, request):
        train()
        return json_response({
            "status": "Success"
        })

    async def evaluation(self, request):
        body = await decode_request(request)
        able_fields = ['filename', 'paths', 'labels']
        body = filter_fields(able_fields, body)

        accuracy, letter_accuracy = evaluation(**body)
        return json_response({
            "status": "Success",
            "accuracy": accuracy,
            "letter_accuracy": letter_accuracy
        })

    async def prediction(self, request):
        body = await decode_request(request)
        required_fields = ['img']
        validate_fields(required_fields, body)

        img = body.get('img')
        predicted = predict(img)
        return json_response({
            "status": "Success",
            "predicted": predicted
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
