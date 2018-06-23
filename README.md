# audio_DCASE

Paper Implementation for :

[1] Deep Neural Network Baseline For Dcase Challenge 2016 [[Paper](http://www.cs.tut.fi/sgn/arg/dcase2016/documents/challenge_technical_reports/DCASE2016_Kong_3008.pdf)]

This code runs on the DCASE 2016 Audio Dataset.

## You need to define:

`wav_dev_fd` development audio folder

`wav_eva_fd` evaluation audio folder

`dev_fd` development features folder

`eva_fd` evaluation features folder

`label_csv` development meta file

`txt_eva_path` evaluation test file

`new_p` evaluation evaluate file

## Cloning the repo
Go ahead and clone this repository using
```
$ git clone https://github.com/DeepLearn-lab/audio_CHIME.git
``` 

## Quick Run
If you are looking for a quick running version go inside `single_file` folder and run
```
$ python mainfile.py
```

## Detailed Task
The process involves three steps:
1. Feature Extraction
2. Training on Development Dataset
3. Testing on Evaluation Dataset

### Feature Extraction

We are going to extract **mel frequencies** on raw audio waveforms. Go ahead and uncomment  
```feature_extraction``` function which would extract these features and save it in the `.f` pickle.

### Training

We train our model on these extracted featuers. We use a convolution neural network for training and testing purpose. Alteration in model can be done in `model.py` file.
All `hyper-parameters` can be set in `util.py`. Once you have made all the required changes or want to run on the pre-set ones, run 
```
$ python mainfile.py 
```

This will run the model which we test and use `EER` for rating our model.
