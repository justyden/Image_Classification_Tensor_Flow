#!/mnt/c/Program Files/Coding Applications/Visual Studio Code Programs/Python Programs/CMPSC 497 Special Topics CV Lab 13/Image_Classification_Tensor_Flow/linux_venv/bin/python
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import sys

print(sys.path)

PATH = os.path.dirname('src/TrainingData')
train_dir = PATH
BATCH_SIZE = 8
IMG_SIZE = (160, 160)
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, shuffle=True, batch_size=BATCH_SIZE,
    image_size=IMG_SIZE)