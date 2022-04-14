import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os.path
import numpy as np
import pandas as pd
import h5py

from pathlib import Path
from fastai.vision import *
from fastai.metrics import error_rate

import requests
from PIL import Image
from io import BytesIO
import cv2

# dataset = h5py.File('C:/Users/Drew/OneDrive/Pictures/ml-food/food_c101_n1000_r384x384x3.h5', 'r')
#
# print(dataset.keys())
#
# grid = dataset['category']
#
# # bs = 128   # batch size
# # arch = models.resnet50
#
# path = Path('C:/Users/Drew/OneDrive/Pictures/ml-food')
# path_h5 = path
# path_img = path/'images'
# path_meta = path/'meta/meta'
#
#
# def label_from_folder_map(class_to_label_map):
#     return  lambda o: class_to_label_map[(o.parts if isinstance(o, Path) else o.split(os.path.sep))[-2]]
#
#
# # Develop dictionary mapping from classes to labels
# classes = pd.read_csv(path_meta/'classes.txt', header=None, index_col=0,)
# labels = pd.read_csv(path_meta/'labels.txt', header=None)
# classes['map'] = labels[0].values
# classes_to_labels_map = classes['map'].to_dict()
# label_from_folder_food_func = label_from_folder_map(classes_to_labels_map)
#
#
# # Setup the training ImageList for the DataBunch
# train_df = pd.read_csv(path_meta/'train.txt', header=None).apply(lambda x : x + '.jpg')
# train_image_list = ImageList.from_df(train_df, path_img)
#
# # Setup the validation ImageList for the DataBunch
# valid_df = pd.read_csv(path_meta/'test.txt', header=None).apply(lambda x : x + '.jpg')
# valid_image_list = ImageList.from_df(valid_df, path_img)

# image_dir = Path('C:/Users/Drew/OneDrive/Pictures/ml-food/images')
image_dir = Path('/Users/collinstratton/Documents/archive/images')
images_paths = list(image_dir.glob(r'**/*.jpg'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], images_paths))

images_paths = pd.Series(images_paths, name='Image_Path').astype(str)
labels = pd.Series(labels, name='Label')

images = pd.concat([images_paths, labels], axis=1)

category_samples = []
for category in images['Label'].unique():
    category_slice = images.query('Label == @category')
    category_samples.append(category_slice.sample(300, random_state=1))

# We group in table and check the number of images by category
images_samples = pd.concat(category_samples, axis=0)
images_samples['Label'].value_counts()

# Testing and training
train_data, test_data = train_test_split(images_samples, train_size=0.7, shuffle=True, random_state=1)

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_images = train_generator.flow_from_dataframe(
    dataframe=train_data,
    x_col='Image_Path',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_data,
    x_col='Image_Path',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_data,
    x_col='Image_Path',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

pretrained_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,  # quitamos la ultima capa del modelo original
    weights='imagenet',
    pooling='avg',
)

# Freeze the weights of the network to maintain transfer learning
pretrained_model.trainable = False

inputs = pretrained_model.input

# We create two layers of our own
x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)

# We create our output layer, with the 101 classes we have
output_layer = tf.keras.layers.Dense(101, activation='softmax')(x)

# We unite the original model with our new layers
model = tf.keras.Model(inputs, output_layer)

print(model.summary())

# Training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=50,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]
)

# Training Results - Prediction Accuracy
results = model.evaluate(test_images, verbose=1)
print("Precision: {:.2f}%".format(results[1] * 100))

# Making a prediction with an internet image
url = 'https://www.hola.com/imagenes/cocina/tecnicas-de-cocina/20210820194795/como-hacer-patatas-fritas-perfectas/0-986-565/portada-patatas-age-t.jpg'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img = np.array(img).astype(float) / 255

img = cv2.resize(img, (224, 224))
predict = model.predict(img.reshape(-1, 224, 224, 3))

print(np.argmax(predict))