train_size = 3000
val_size = 1400
test_size = 500

img_width = 170
img_height = 32
img_channel = 1

letters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Number of classes for softmax (+1 for space)
n_classes = len(letters) + 1
epochs = 20
batch_size = 16
max_length = 15

model_path = 'Data/models/model.weights.hdf5'
test_path = 'Data/csv/test_final.csv'

download_data_path = "https://drive.google.com/uc?id=1trrO0sUwNf6mXWvV45SMfLrPVJ9BLwLE"
