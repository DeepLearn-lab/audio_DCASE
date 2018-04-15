# -*- coding: utf-8 -*-
"""
** deeplean-ai.com **
** dl-lab **
created by :: adityac8
"""
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input
from keras.utils import to_categorical
from util import *
inpx = Input(shape=(1,10,64),name='inpx')
    
x = Conv2D(filters=nb_filter,
           kernel_size=filter_length,
           data_format='channels_first',
           padding='same',
           activation=act1)(inpx)

hx = MaxPooling2D(pool_size=pool_size)(x)
h = Flatten()(hx)
wrap = Dense(input_neurons, activation=act2,name='wrap')(h)
score = Dense(num_classes,activation=act3,name='score')(wrap)

model = Model([inpx],score)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

