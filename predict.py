import cv2
import matplotlib.pyplot as plt

from utils.functions import predict_label


def predict(model, img):
    # Model for Predict
    # model = text_recognition_model('predict')
    # model.load_weights('checkpoint/final_model.h5')

    # Predict
    predicted = predict_label(model, img)

    # # Visualize
    # image = cv2.imread(img)
    #
    # plt.title(predicted)
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()

    return predicted
