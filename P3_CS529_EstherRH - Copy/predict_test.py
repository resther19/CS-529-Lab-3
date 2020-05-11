# References https://github.com/seth814/Audio-Classification
'''
Adapted and modified by Esther Rodriguez
April 2020
'''

import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score

#using the CNN model gives probabilities to each class for each instance and chooses the class
#with the highest probability as the prediction for that instance
def build_predictions(audio_dir):
    y_pred = []
    fn_prob = {}
    print('Extracting features from audio')
    for fn in tqdm(os.listdir(audio_dir)):
        rate, wav = wavfile.read(os.path.join(audio_dir, fn))
        y_prob = []

        for i in range(0, wav.shape[0]-config.step, config.step):
            sample = wav[i:i+config.step]
            x = mfcc(sample, rate, numcep=config.nfeat,
                     nfilt=config.nfilt, nfft = config.nfft)
            x = (x - config.min)/(config.max - config.min)

            if config.mode=='conv':
                x = x.reshape(1, x.shape[0], x.shape[1], 1)
            elif config.mode=='time':
                x = np.expand_dims(x, axis=0)
            y_hat = model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))


        fn_prob[fn] = np.mean(y_prob, axis=0).flatten()

    return y_pred, fn_prob

#creates a dataframe using the id of the songs in the testing dataset
df = pd.read_csv('test_set.csv')
classes = list(range(6))
p_path = os.path.join('pickles', 'conv.p')

#if the data already exists, it opens it from the pickle folder
with open(p_path,'rb') as handle:
    config = pickle.load(handle)

#loads the CNN model
model = load_model(config.model_path)

#builds predictions for the clean wave files for the training set
y_pred, fn_prob = build_predictions('clean_test')

#adds the predicted class to the dataframe
y_probs = []
for i, row in df.iterrows():
    y_prob = fn_prob[row.fname]
    y_probs.append(y_prob)
    for c, p in zip(classes, y_prob):
        df.at[i,c] = p

y_pred = [classes[np.argmax(y)] for y in y_probs]
df['y_pred'] = y_pred

#creates a csv file with the dataframe
df.to_csv('predict_test.csv', index=False)
