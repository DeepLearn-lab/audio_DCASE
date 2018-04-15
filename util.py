# -*- coding: utf-8 -*-
"""
** deeplean-ai.com **
** dl-lab **
created by :: akshitac8
"""

#Tell your wavs folder
wav_dev_fd = "D:/workspace/aditya_akshita/datax/dcase/audios/development"
wav_eva_fd = "D:/workspace/aditya_akshita/datax/dcase/audios/evaluation"
dev_fd = "D:/workspace/aditya_akshita/datax/dcase/features/development"
eva_fd = "D:/workspace/aditya_akshita/datax/dcase/features/evaluation"
label_csv = "D:/workspace/aditya_akshita/datax/dcase/texts/development/meta.txt"
txt_eva_path = 'D:/workspace/aditya_akshita/datax/dcase/texts/evaluation/test.txt'
new_p = 'D:/workspace/aditya_akshita/datax/dcase/texts/evaluation/evaluate.txt'
labels = ['bus', 'cafe/restaurant', 'car', 'city_center', 'forest_path', 
           'grocery_store', 'home', 'beach', 'library', 'metro_station', 
           'office', 'residential_area', 'train', 'tram', 'park']
            
lb_to_id = {lb:id for id, lb in enumerate(labels)}
id_to_lb = {id:lb for id, lb in enumerate(labels)}

fsx = 44100.
n_fft = 1024.
agg_num=10
hop=10
num_classes=len(labels)

input_neurons=200
act1='relu'
act2='relu'
act3='softmax'
epochs=100
batchsize=100
nb_filter = 100
filter_length =3
pool_size=(2,2)
