#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:04:25 2019

@author: cis
"""
import csv
from datetime import datetime
import random
import os

import cv2
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


db = loadmat("imdb_crop/imdb.mat")['imdb'][0, 0]
num_records = len(db["face_score"][0])

indices = list(range(num_records))
random.shuffle(indices)

train_indices = indices[:int(len(indices) * 0.8)]
test_indices = indices[int(len(indices) * 0.8):]

headers = ['filename', 'age', 'gender']

def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1
    
full_path = db["full_path"][0]
dob = db["dob"][0]
gender = db["gender"][0]
photo_taken = db["photo_taken"][0]  # year
face_score = db["face_score"][0]
second_face_score = db["second_face_score"][0]
age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

crop_dir = 'crop_dir'
output_dir = 'datasets'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(crop_dir):
    os.makedirs(crop_dir)

train_csv = open(os.path.join(output_dir, 'train.csv'), 'w')
train_writer = csv.writer(train_csv, delimiter=',', )
train_writer.writerow(headers)

val_csv = open(os.path.join(output_dir, 'val.csv'), 'w')
val_writer = csv.writer(val_csv, delimiter=',')
val_writer.writerow(headers)

# TQDM is used to show % wise interation progress (JUST FOR BETTER PROGRESS DISPLAY)
def clean_and_resize(writer):
    for i in tqdm(indices):
        filename = str(full_path[i][0])
        if not os.path.exists(os.path.join(crop_dir, os.path.dirname(filename))):
            os.makedirs(os.path.join(crop_dir, os.path.dirname(filename)))
            
        img_path = os.path.join("imdb_crop", filename)
        
        if float(face_score[i]) < 1.0:
            continue
        
        if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue
    
        if ~(0 <= age[i] <= 100):
            continue
    
        if np.isnan(gender[i]):
            continue
        
        img_gender = int(gender[i])
        img_age = int(age[i])
        
        img = cv2.imread(img_path)
        crop = cv2.resize(img, (165, 165))
        crop_filepath = os.path.join(crop_dir, filename)
        cv2.imwrite(crop_filepath, crop)
        
        writer.writerow([filename, img_age, img_gender])

clean_and_resize(train_writer)
clean_and_resize(val_writer)

