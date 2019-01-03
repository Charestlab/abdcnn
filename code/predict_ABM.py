# PERMUTATION TEST DISTRIBUTION
import os
import numpy as np
from sklearn import linear_model
from random import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn import preprocessing
from multiprocessing import Pool
from sklearn.feature_selection import VarianceThreshold


def predict_save(layer):
    """
    Function that predicts the ABM from layer unit activation
    """
    print(f'Starting predictions for layer {layer}!')
    DCNN_PATH = '../DCNN_features'
    DATA_PATH = '../data'
    AB = np.loadtxt(os.path.join(DATA_PATH, 'ABmag_allsubs.txt'))

    ns = AB.shape[0] # n subjects
    ni = AB.shape[1] # n images

    # set up pipeline with scaler, feature selection and linear regression
    lr = linear_model.LinearRegression()
    clf = make_pipeline(preprocessing.StandardScaler(), VarianceThreshold(threshold=(.8 * (1 - .8))), lr)


    # Load layer data
    X = np.load(os.path.join(DCNN_PATH, 'features_{0}.npy'.format(layer)))

    preds = np.zeros((ns, X.shape[0]))
    for sub in range(ns):
        #print_progress(sub, range(ns))
        preds[sub, :] = cross_val_predict(clf, X, AB[sub, :], cv=ni)

    np.save(os.path.join(DATA_PATH, f'{layer}_prediction.npy'), preds)
    print(f'Layer {layer} done!')



if __name__ == '__main__':
    layers = ('conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8', 'GloVe')
    for layer in layers:
        predict_save(layer)
