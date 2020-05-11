# References https://github.com/seth814/Audio-Classification
'''
Adapted and modified by Esther Rodriguez
April 2020
'''

from __future__ import print_function
import librosa
from scipy.io import wavfile
import numpy as np
import scipy as sp
from tqdm import tqdm
import pandas as pd
from sklearn.decomposition import FastICA #, PCA
from python_speech_features import mfcc, logfbank
import matplotlib.pyplot as plt
import matplotlib as mtpt
import os
import datetime

begin_time = datetime.datetime.now()

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask



df = pd.read_csv('training_set.csv')
df.set_index('fname', inplace=True)
df2.reset_index(inplace=True)

if len(os.listdir('clean16k')) == 0:
    for f in tqdm(df.fname):
        signal, rate = librosa.load('wav_files/'+f, sr=16000)
        mask = envelope(signal, rate, 0.0005)
        wavfile.write(filename='clean16k/'+f, rate=rate, data=signal[mask])



end_time = datetime.datetime.now()
print(end_time - begin_time)
