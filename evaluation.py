import pandas as pd
from datetime import datetime
import logging

from lib.utils.utils import predict_label


def predict_data_output(model, images, labels, n=None):
    if n is None or n > len(images):
        n = len(images)
    start = datetime.now()
    acc = 0
    letter_acc = 0
    letter_cnt = 0
    cnt = 0
    letter_mis = []  # count miss match letter for each images
    predicteds = []
    for i in range(n):
        predicted = predict_label(model, images[i])
        if predicted is None:
            continue
        predicteds.append(predicted)
        actual = labels[i]

        cnt += 1
        mis = 0
        for j in range(min(len(predicted), len(actual))):
            if predicted[j] == actual[j]:
                letter_acc += 1
            else:
                mis += 1
        letter_cnt += max(len(predicted), len(actual))
        letter_mis.append(mis)

        if predicted == actual:
            acc += 1
        if cnt % 100 == 0:
            print("Processed {} images".format(cnt))
    end = datetime.now()
    print("Total Times: ", end - start)
    return acc, letter_acc, letter_cnt, letter_mis, cnt


def evaluation(model, filename=None, paths=None, labels=None):
    if filename is None:
        filename = 'Data/csv/test_final.csv'

    # Load Data for evaluation
    try:
        data = pd.read_csv(filename)
    except FileNotFoundError as err:
        logging.exception(err)
        return None, None

    if paths is None or labels is None:
        paths = data['Image'].values.tolist()
        labels = data['Label'].values.tolist()

    # Model for Evaluation
    # model = text_recognition_model('predict')
    # model.load_weights('checkpoint/model.h5')

    # Calculate accuracy
    acc, letter_acc, letter_cnt, mis_match, n_predicteds = predict_data_output(model, paths, labels)

    accuracy = round((acc / n_predicteds) * 100, 2)
    letter_accuracy = round((letter_acc / letter_cnt) * 100, 2)

    print("Validation Accuracy: ", accuracy, " %")
    print("Validation Letter Accuracy: ", letter_accuracy, " %")

    return accuracy, letter_accuracy
