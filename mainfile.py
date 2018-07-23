# -*- coding: utf-8 -*-
"""
** deeplean-ai.com **
** dl-lab **
created by :: adityac8
"""

import numpy as np
import os
import librosa
import cPickle
import csv
from scipy import signal
from scikits.audiolab import wavread
import scipy

from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input
from keras.utils import to_categorical

from util import *
from model import *

def feature_extraction(wav_fd, fe_fd):
    names = [na for na in os.listdir(wav_fd) if na.endswith('.wav')]
    names = sorted(names)
    for na in names:
        print na
        path = wav_fd + '/' + na
        wav, fs, enc = wavread( path )        
        if wav.ndim == 2:
            wav=np.mean(wav, axis=-1)
        assert fs == fsx
        ham_win = np.hamming(n_fft)
        [f, t, x] = signal.spectral.spectrogram(x=wav, 
                                                window=ham_win, 
                                                nperseg=n_fft, 
                                                noverlap=0, 
                                                detrend=False, 
                                                return_onesided=True, 
                                                mode='magnitude') 
        x = x.T
        if globals().get('melW') is None:
            global melW
            melW = librosa.filters.mel(sr=fs, 
                                       n_fft=n_fft, 
                                       n_mels=64, 
                                       fmin=0., 
                                       fmax=22100)
        x = np.dot(x, melW.T)
        x = np.log(x + 1e-8)
        out_path = fe_fd + '/' + na[0:-4] + '.f'
        cPickle.dump(x, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

def mat_2d_to_3d(X, agg_num, hop):
    # pad to at least one block
    len_X, n_in = X.shape
    if (len_X < agg_num):
        X = np.concatenate((X, np.zeros((agg_num-len_X, n_in))))
        
    # agg 2d to 3d
    len_X = len(X)
    i1 = 0
    X3d = []
    while (i1+agg_num <= len_X):
        X3d.append(X[i1:i1+agg_num])
        i1 += hop
    return np.array(X3d)

def train_data():
    with open( label_csv, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    
    X3d_all = []
    y_all = []
    i=0
    for li in lis:
        # load data
        [na, lb] = li[0].split('\t')
        na = na.split('/')[1][0:-4]
        path = dev_fd + '/' + na + '.f'
        #i+=1
        #print i
        try:
            X = cPickle.load( open( path, 'rb' ) )
        except Exception as e:
            print 'Error while parsing',path
            continue
        i+=1
        X3d = mat_2d_to_3d( X, agg_num, hop )
        X3d_all.append( X3d )
        y_all += [ lb_to_id[lb] ] * len( X3d )
    
    print 'Files loaded',i
    X3d_all = np.concatenate( X3d_all )
    y_all = np.array( y_all )
    
    return X3d_all, y_all
