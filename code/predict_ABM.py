# PERMUTATION TEST DISTRIBUTION
import os
import numpy as np
import pandas as pd
from sklearn import linear_model
from random import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
import warnings
from tqdm import tqdm
warnings.simplefilter(action="ignore", category=FutureWarning)


def predict_save(layer):
    """
    Function that predicts the ABM from layer unit activation
    """
    layerdims = {
    'conv1': (48, 96, 55*55),
    'conv2': (48, 256, 27*27),
    'conv3': (48, 384, 13*13),
    'conv4': (48, 384, 13*13),
    'conv5': (48, 256, 13*13)}

    print(f'Starting predictions for layer {layer}!')
    DCNN_PATH = '../DCNN_features'
    DATA_PATH = '../data'
    AB = np.loadtxt(os.path.join(DATA_PATH, 'ABmag_allsubs.txt'))
    AB_df = pd.read_csv('../data/ABM_subs.csv')
    AB = np.zeros((len(np.unique(AB_df['subject'])), 48))

    for x, s in enumerate(np.unique(AB_df['subject'])):
        AB[x, :] =  AB_df[AB_df['subject'] == s]['ABM'].values

    ns = AB.shape[0] # n subjects
    ni = AB.shape[1] # n images

    # set up pipeline with scaler, feature selection and linear regression
    lr = linear_model.LinearRegression()
    clf = make_pipeline(VarianceThreshold(threshold=0.15), preprocessing.StandardScaler(), lr)

    # Load layer data
    X = np.load(os.path.join(DCNN_PATH, 'features_{0}.npy'.format(layer)))
    if 'conv' in layer:
        laydims = layerdims[layer]
        X = X.reshape(laydims).mean(2)

    preds = np.zeros((ns, X.shape[0]))
    for sub in tqdm(range(ns)):
        preds[sub, :] = cross_val_predict(clf, X, AB[sub, :], cv=ni)

    np.save(os.path.join(DATA_PATH, f'test_{layer}_prediction.npy'), preds)
    print(f'Layer {layer} done!')



if __name__ == '__main__':
    layers = ('conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8')
    for layer in layers:
        predict_save(layer)
