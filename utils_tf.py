import matplotlib.pyplot as plt
import numpy as np
import argparse, os
import pandas as pd
import glob
import cv2
import pickle

classes = ('Client', 'Imposter')

def readDb():
    lstFilesTrainValid = 'data/NormalizedFace/client_train_normalized.txt'
    lstFilesTrainImposter = 'data/NormalizedFace/imposter_train_normalized.txt'
    lstFilesTestValid = 'data/NormalizedFace/client_test_normalized.txt'
    lstFilesTestImposter = 'data/NormalizedFace/imposter_test_normalized.txt'

    dfTrainValid = pd.read_csv(lstFilesTrainValid, header=None, names=['Path'])
    dfTrainValid['Path'] = 'data/NormalizedFace/ClientNormalized/' + dfTrainValid['Path']
    dfTrainValid['Label'] = 1
    dfTrainImposter = pd.read_csv(lstFilesTrainImposter, header=None, names=['Path'])
    dfTrainImposter['Path'] = 'data/NormalizedFace/ImposterNormalized/' + dfTrainImposter['Path']
    dfTrainImposter['Label'] = 0

    dfTestValid = pd.read_csv(lstFilesTestValid, header=None, names=['Path'])
    dfTestValid['Path'] = 'data/NormalizedFace/ClientNormalized/' + dfTestValid['Path']
    dfTestValid['Label'] = 1
    dfTestImposter = pd.read_csv(lstFilesTestImposter, header=None, names=['Path'])
    dfTestImposter['Path'] = 'data/NormalizedFace/ImposterNormalized/' + dfTestImposter['Path']
    dfTestImposter['Label'] = 0

    dfTrain = pd.concat([dfTrainValid, dfTrainImposter])
    dfTest = pd.concat([dfTestValid, dfTestImposter])

    return dfTrain, dfTest

def process_train_data():
    print("process_train_data")
    cache_dir = './cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache_file, cache_data = "train.pkl", None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass

    if not cache_data:
        train, _ = readDb()
        trainx  = np.array([cv2.imread(t.replace('\\', '/'), 0) for t in train['Path'].ravel()])
        trainy = train['Label'].ravel()
        if cache_file is not None:
            cache_data = {'trainx': trainx, 'trainy': trainy}
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        trainx, trainy = cache_data['trainx'], cache_data['trainy']
    return trainx, trainy

def process_test_data():
    print("process_test_data")
    cache_dir = './cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache_file, cache_data = 'test.pkl', None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass  
    if not cache_data:
        _, test = readDb()
        testx  = np.array([cv2.imread(t.replace('\\', '/'), 0) for t in test['Path'].ravel()])
        testy = test['Label'].ravel()
        if cache_file is not None:
            cache_data = {'testx': testx, 'testy': testy}
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        testx, testy = cache_data['testx'], cache_data['testy']
    return testx, testy

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--epochs', type=int, default=1)
    # parser.add_argument('--learning-rate', type=float, default=0.01)
    # parser.add_argument('--batch-size', type=int, default=128)
    # parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    # parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    # parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    # parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    args, _ = parser.parse_known_args()
    trainx, trainy  = process_train_data()
    X_test, y_test = process_test_data()

    # shuffle training data and split them into training and validation
    indices = np.random.permutation(trainx.shape[0])
    # 20% to val
    split_idx = int(trainx.shape[0]*0.8)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    X_train, X_validation = trainx[train_idx,:], trainx[val_idx,:]
    y_train, y_validation = trainy[train_idx], trainy[val_idx]

    # get overall stat of the whole dataset
    n_train = X_train.shape[0]
    n_validation = X_validation.shape[0]
    n_test = X_test.shape[0]
    image_shape = X_train[0].shape
    n_classes = len(np.unique(y_train))
    print("There are {} training examples ".format(n_train))
    print("There are {} validation examples".format(n_validation))
    print("There are {} testing examples".format(n_test))
    print("Image data shape is {}".format(image_shape))
    print("There are {} classes".format(n_classes))

    upload_dir = './upload'
    if not os.path.exists(upload_dir): # Make sure that the folder exists
        os.makedirs(upload_dir)


    np.savez(os.path.join(upload_dir,'training'), image=X_train, label=y_train)
    np.savez(os.path.join(upload_dir,'validation'), image=X_validation, label=y_validation)