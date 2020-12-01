import gdown
import logging
import os

from config import download_data_path


def download_data():
    url = download_data_path
    dest = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data.tar.gz')

    try:
        gdown.download(url, dest)
    except Exception as err:
        logging.exception(err)


if __name__ == '__main__':
    download_data()
