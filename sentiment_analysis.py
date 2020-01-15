#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 12:31:58 2020

@author: cis
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
import random
from collections import Counter
import tensorflow as tf

lemma = WordNetLemmatizer()
no_of_lines=10000000

import pandas as pd

data = pd.read_csv("/home/cis/sentiment_train.csv", index_col='id')

convert_dict = {'sentiment': int, 
                'text': str
               }
data = data.astype(convert_dict)
#data_new = data[data["text"] == str]

pos_data = data[data["sentiment"] == 1]
neg_data = data[data["sentiment"] == 0]


pos = ""
neg = ""

for row in pos_data.itertuples(index = True, name ='Pandas'): 
    pos +=  getattr(row, "text")
    pos += "\n"
    
for row in neg_data.itertuples(index = True, name ='Pandas'): 
    neg +=  getattr(row, "text")
    neg += "\n"

def create_lexicon(pos,neg):
    lexicon = []
    for fi in [pos,neg]:
        contents = fi.split("\n")
        for l in contents[:no_of_lines]:
            all_words = word_tokenize(l.lower())
            lexicon += list(all_words)
    lexicon = [lemma.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    l2=[]
    for w in w_counts:
        if 1000>w_counts[w]> 50:
            l2.append(w)
    print(len(l2))
    return l2

def sample_handling(sample,lexicon,classification):
    featureset= []
    contents = sample.split("\n")
    for l in contents[:no_of_lines]:
        current_words = word_tokenize(l.lower())
        current_words = [lemma.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))
        for word in current_words:
            if word.lower() in lexicon:
                index_value = lexicon.index(word.lower())
                features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])
    return featureset

def create_featuresets_and_labels(pos,neg,test_size=0.1):
    lexicon = create_lexicon(pos,neg)
    features = []
    features += sample_handling(pos, lexicon,[1,0])
    features += sample_handling(neg, lexicon,[0,1])
    random.shuffle(features)
    
    features = np.array(features)
    
    testing_size = int(test_size*len(features))
    
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])
    
    return train_x,train_y,test_x,test_y

train_x,train_y,test_x,test_y = create_featuresets_and_labels(pos, neg)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
batch_size = 100

x = tf.placeholder('float',shape=[None, len(train_x[0])])
y = tf.placeholder('float')

def neural_network_model(data):
    #output = input*weight + bias
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}
    
    #as (input_data * weights) + bias
    
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    output = tf.matmul(l3, output_layer['weights'])+ output_layer['biases']
    
    return output

def train_nn(x):
    pred = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    no_of_epochs = 10
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch in range(no_of_epochs):
            epoch_loss = 0
            
            i = 0
            while i<len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                
                _, c = sess.run([optimizer, cost], feed_dict={x:batch_x, y:batch_y})
                epoch_loss += c
                i += batch_size
            print("Epoch",epoch,"Completed out of",no_of_epochs, "loss", epoch_loss)
            
        correct = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,"float"))
        print("Accuracy :",accuracy.eval({x:test_x, y:test_y}))
        
train_nn_1 = train_nn(x)

## Predictions

pred_data = pd.read_csv("/home/cis/Downloads/sentiment_test_x.csv")
del(pred_data)