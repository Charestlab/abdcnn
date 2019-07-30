# PERMUTATION TEST DISTRIBUTION
import os
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn import preprocessing
import sys
import random
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
import warnings
warnings.filterwarnings("ignore")
"""
Shuffles the labels and fits the regression for ABM for each layer and subject

Run this by:
python predict_ABM_PERMUTATIONS.py conv5

"""

def print_progress(i, n):
    state = (i+1)/n*100
    n_bars = int(state/10)
    print('-'*n_bars, '{0:.2f}% done!'.format(state), end='\r')

# These are the layer dimensions of the convolutional layers in AlexNet
# (n images x n features x n rows x n columns)
layerdims = {
    'conv1': (48, 96, 55*55),
    'conv2': (48, 256, 27*27),
    'conv3': (48, 384, 13*13),
    'conv4': (48, 384, 13*13),
    'conv5': (48, 256, 13*13)}

# Set paths and load AB data
DCNN_PATH = '../DCNN_features'
DATA_PATH = '../data'
# Sub x ABM per image
AB = np.loadtxt(os.path.join(DATA_PATH, 'ABmag_allsubs.txt'))
AB_df = pd.read_csv('../data/ABM_subs.csv')
AB = np.zeros((len(np.unique(AB_df['subject'])), 48))

for x, s in enumerate(np.unique(AB_df['subject'])):
    AB[x, :] =  AB_df[AB_df['subject'] == s]['ABM'].values

ns = AB.shape[0] # n subjects
ni = AB.shape[1] # n images

# lets do 5000 permutations
nperm = 3000

# Make sure that every layer get the same permutation by
# predefining the permutations
imi = np.array([random.sample(range(0, ni), ni) for x in range(nperm)])

layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
#layers = [ 'fc6', 'fc7', 'fc8']
# User input for now
#layers = [sys.argv[1]]

# define our pipeline (standardscaler, feature selection, and linear regression)
lr = linear_model.LinearRegression()
clf = make_pipeline(VarianceThreshold(), preprocessing.StandardScaler(), lr)

for layer in layers:
    print(f'Running layer: {layer}')
    predicted = np.zeros([nperm, ns, ni])
    X = np.load(os.path.join(DCNN_PATH, f'features_{layer}.npy'))

    # we want to average over the feature dimension for convolutional layers
    if 'conv' in layer:
        laydims = layerdims[layer]
        X = X.reshape(laydims).mean(2)
    for perm in range(nperm):
        # get the permuted
        X_ = X[imi[perm,:], :]
        print_progress(perm, nperm)
        for sub in range(ns):
            predicted[perm, sub, :] = cross_val_predict(clf, X_, AB[sub, :], cv=48, n_jobs=2)

    np.save(os.path.join(DATA_PATH, f'{layer}_prediction_permutations.npy'), predicted)
