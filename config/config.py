train_size = 3000
val_size = 1400
test_size = 500

img_width = 170
img_height = 32
img_channel = 1

letters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Number of classes for softmax (+1 for space)
n_classes = len(letters) + 1
batch_size = 16
max_length = 15
