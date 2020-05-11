# References https://github.com/seth814/Audio-Classification
'''
Adapted and modified by Esther Rodriguez
April 2020
'''

import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
import pickle
from keras.callbacks import ModelCheckpoint
from cfg import Config

#checks if the matrix of features is already in the pickle folder
def check_data():
    if os.path.isfile(config.p_path):
        print('Loading existing data for {} model'.format(config.mode))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None

#builds the matrix of features
def build_rand_feat():
    tmp = check_data()
    if tmp:
        return tmp.data[0], tmp.data[1]
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')

    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.genre == rand_class].index)
        #reads the clean wave files for the training set
        rate, wav = wavfile.read('clean16k/'+file)
        genre = df.at[file, 'genre']
        rand_index = np.random.randint(0, wav.shape[0] - config.step)
        sample = wav[rand_index:rand_index+config.step]
        X_sample = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample)
        y.append(classes.index(genre))
    config.min = _min
    config.max = _max
    X, y = np.array(X), np.array(y)
    X = (X - _min)/(_max - _min)
    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif config.mode == 'time':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y = to_categorical(y, num_classes = 6)
    config.data = (X,y)

    with open(config.p_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=2)
    return X, y

### CNN Model ###
def get_conv_model():
    model = Sequential()
    # activation layers
    model.add(Conv2D(8, (7,7), activation='tanh', strides=(1,1),
                     padding='same', input_shape=input_shape))

    model.add(Conv2D(16, (5,5), activation='relu', strides=(1,1),
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (5,5), activation='relu', strides=(1,1),
                     padding='same'))
    model.add(Conv2D(64, (5,5), activation='relu', strides=(1,1),
                     padding='same'))
    model.add(Conv2D(128, (5,5), activation='relu', strides=(1,1),
                      padding='same'))
    model.add(MaxPool2D((2,2)))
    #dropout=0.1
    model.add(Dropout(0.1))
    model.add(Flatten())
    #dense layers
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(6,activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['acc'])
    return model


#creates a dataframe using the id of the songs in the training dataset
df = pd. read_csv('training_set.csv')
df.set_index('fname', inplace=True)

#adds the signal shape to the dataframe
for f in df.index:
    rate, signal = wavfile.read('clean16k/'+f)
    df.at[f,'length'] = signal.shape[0]/rate

#find the unique classes
classes = list(np.unique(df.genre))
#find the class distribution by genre
class_dist = df.groupby(['genre'])['length'].mean()

#finds the probability distribution for each class
n_samples = 2 * int(df['length'].sum()/0.1)
prob_dist = class_dist/class_dist.sum()
choices = np.random.choice(class_dist.index, p = prob_dist)


config = Config(mode='conv')

#convolutional model
if config.mode == 'conv':
    X, y = build_rand_feat()
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2], 1)
    model =  get_conv_model()
#recurrent model
elif config.mode == 'time':
    X, y = build_rand_feat()
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2])
    model = get_recurrent_model()


class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)
checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc',
                             verbose=1, mode='max', save_best_only=True,
                             save_weights_only=False, period=1)

#bulds the CNN model with specified epochs, batch size, and validation split
# Trained the Network with 20 epochs but changed it to 5 so that it runs faster
model.fit(X, y, epochs = 5, batch_size=128,
          shuffle=True, validation_split=0.1, callbacks=[checkpoint])

model.save(config.model_path)
