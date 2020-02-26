# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 12:52:13 2019

@author: TsungYuan
"""

import numpy as np
from numpy import genfromtxt
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, Dropout, MaxPooling1D, GlobalAveragePooling1D
import os

def one_hot(data):
    Y_train=np.zeros((1890,8))
    for j in range(1890):
        Y_train[j,int(data[j,167])] = 1
    return Y_train
 
def evaluate_model(X_train, X_test, y_train, y_test):
    while True:
        model_m = Sequential()
        model_m.add(Conv1D(10, 10, activation='relu', input_shape=(167, 1)))
        model_m.add(Conv1D(10, 10, activation='relu'))
        model_m.add(MaxPooling1D(5))
        model_m.add(Conv1D(60, 10, activation='relu'))
        model_m.add(Conv1D(60, 10, activation='relu'))
        model_m.add(GlobalAveragePooling1D())
        model_m.add(Dropout(0.3))
        model_m.add(Dense(8, activation='softmax'))
        print(model_m.summary())
        model_m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model_m.fit(X_train, y_train, epochs=50, batch_size=100, verbose=1)
        _, accuracy = model_m.evaluate(X_test, y_test, batch_size=10, verbose=0)
        print(accuracy,history.history['acc'][-1])
        if accuracy >= 0.95 and float(history.history['acc'][-1]) >= 0.95:
            model_m.save('my_model.h5')
            break

file = ['G11','G15','G17','G19','G32','G34','G48','G49']

for i in range(8):
    filename = os.getcwd() + '/train/' + file[i] + '.csv'
    my_data = genfromtxt(filename, delimiter=',')
    label = np.full((my_data.shape[1]), i)
    newdata = np.insert(my_data.T, 167, values=label, axis=1)
    if i == 0:
        data = newdata
    data = np.append(data,newdata ,axis=0)
    
data = data.reshape(1890,168,1)
random.shuffle(data)

Y_train = one_hot(data)

X_train, X_test, y_train, y_test = train_test_split(data[:,0:167,:],Y_train, test_size=0.33)

evaluate_model(X_train, X_test, y_train, y_test)
