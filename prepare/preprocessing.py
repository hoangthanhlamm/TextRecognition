import pandas as pd
import cv2
from datetime import datetime


def store_grayscale(destination_folder, files, labels, n=None):
    images = []
    count_ = 0
    start = datetime.now()
    print('Total Images: ', len(files))
    for i, f in enumerate(files):
        dest = destination_folder + str(count_ + 1) + '_' + labels[i] + '.jpg'

        img = cv2.imread(f)
        if img is None:
            continue
        img_grayscale = img[:, :, 1]
        cv2.imwrite(dest, img_grayscale)
        images.append([dest, labels[i]])

        count_ += 1
        if count_ % 100 == 0:
            print('Processed Images: ', count_)
        if n is not None and count_ >= n:
            break
    images_df = pd.DataFrame(images, columns=['Image', 'Label'])
    end = datetime.now()
    print('Time taken for preprocessing: ', end - start)
    return images_df


datasets = ['train', 'val', 'test']
for dataset in datasets:
    filename = 'Data/' + dataset + '_data.csv'
    data = pd.read_csv(filename)
    files_ = data['Image'].values
    labels_ = data['Label'].values
    data_df = store_grayscale('Data/Train/', files_, labels_)

    dest = 'Data/' + dataset + '_final.csv'
    data_df.to_csv(dest, index=False)
