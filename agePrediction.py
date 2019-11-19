#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:04:25 2019

@author: cis
"""
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE

with open("datasets/train.csv", "r") as train_csv:
    train_data = train_csv.readlines()
    
with open("datasets/val.csv", "r") as val_csv:
    test_data = val_csv.readlines()

train_data = train_data[1:]
test_data = test_data[1:]

label_data = {}

e = 0
for i in train_data[:10000]:
    filepath, age, gender = i.split(",")
    age, gender = int(age), int(gender)
#    age = tf.cast(age, tf.int64)
#    gender = tf.cast(gender, tf.int64)
#    
    img = tf.io.read_file("crop_dir/"+filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
#    img.set_shape([128, 128, 3])
    
    label_data[e] =  [img, age, gender]
    e += 1

model = models.Sequential()
model.add(layers.Conv2D(128, (3,3), activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(100, activation='softmax'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model = model.fit()

model.summary()


#for i in train_data[10000:20000]:
#    filepath, age, gender = i.split(",")
#    age, gender = int(age), int(gender)
##    age = tf.cast(age, tf.int64)
##    gender = tf.cast(gender, tf.int64)
##    
#    img = tf.io.read_file("crop_dir/"+filepath)
#    img = tf.image.decode_jpeg(img, channels=3)
#    img = tf.image.convert_image_dtype(img, tf.float32)
##    img.set_shape([128, 128, 3])
#    
#    label_data[e] =  [img, age, gender]
#    e += 1    
##    
##img_path = []
##
##for i in range(len(train_data)):
##    train_data[i] = train_data[i].split(",")
##    img_path.append(train_data[i][0])
#
##test_img_path = []
##for i in range(len(test_label)):
##    test_label[i] = test_label[i].split(",")
##    test_img_path.append(test_label[i][0])
#
##crop_dir = "crop_dir"
##t_label = []
##for i in train_label:
##    t_label.append(i[1])
#
#list_ds = tf.data.Dataset.list_files('crop_dir/*/*')
#
#for f in list_ds.take(5):
#    print(f.numpy())
#
#def decode_image(img):
#    img = tf.image.decode_jpeg(img, channels=3)
#    img = tf.image.convert_image_dtype(img, tf.float32)
#    return img
#
##img = []
#def get_label(train_label):
#    labels = [i[1] for i in train_label]
#    return labels
#
#def process_path(crop_dir):
#    label = get_label(train_label)
#    img = tf.io.read_file(crop_dir)
#    img = decode_image(img)
#    return img, label
#
#labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

i=0
for image, label in labeled_ds.take(15):
#    plt.imshow(image)
    plt.subplot(3,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    i+=1
    plt.xlabel(label)
#    print("shape: ", image.numpy().shape)
#    print("Label: ", label.numpy())
plt.show()


model = models.Sequential()
model.add()