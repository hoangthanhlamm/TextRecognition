import pandas as pd
from datetime import datetime

from net.CRNN import text_recognition_model
from utils.functions import predict_label
from config.config import *


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
    return acc, letter_acc, letter_cnt, letter_mis, predicteds


def evaluation(filename=None, paths=None, labels=None):
    if filename is None:
        filename = 'Data/test_final.csv'

    # Load Data for evaluation
    data = pd.read_csv(filename)
    if paths is None or labels is None:
        paths = data['Image'].values.tolist()
        labels = data['Label'].values.tolist()

    print("Paths: ")
    print(paths[:5])
    print("Labels: ")
    print(labels[:5])

    # Model for Evaluation
    model = text_recognition_model('predict')
    model.load_weights('checkpoint/final_model.h5')

    # Calculate accuracy
    acc, letter_acc, letter_cnt, mis_match, predicteds = predict_data_output(model, paths, labels, test_size)

    accuracy = round((acc / len(labels)) * 100, 2)
    letter_accuracy = round((letter_acc / letter_cnt) * 100, 2)

    print("Validation Accuracy: ", accuracy, " %")
    print("Validation Letter Accuracy: ", letter_accuracy, " %")

    return accuracy, letter_accuracy
