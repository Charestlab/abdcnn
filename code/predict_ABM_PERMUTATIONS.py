# PERMUTATION TEST DISTRIBUTION
import os
import numpy as np
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn import preprocessing
import sys
import random
from sklearn.feature_selection import VarianceThreshold

"""
Shuffles the labels and fits the regression for ABM for each layer and subject

Run this by:
python predict_ABM_PERMUTATIONS.py conv5

"""

def print_progress(i, n):
    state = (i+1)/n*100
    n_bars = int(state/10)
    print('-'*n_bars, '{0:.2f}% done!'.format(state), end='\r')



# Set paths and load AB data
DCNN_PATH = '../DCNN_features'
DATA_PATH = '../data'
# Sub x ABM per image
AB = np.loadtxt(os.path.join(DATA_PATH, 'ABmag_allsubs.txt'))

ns = AB.shape[0] # n subjects
ni = AB.shape[1] # n images

# lets do 3000 permutations
nperm = 3000

# Make sure that every layer get the same permutation by
# predefining the permutations
imi = np.array([random.sample(range(0, ni), ni) for x in range(nperm)])

layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
# User input for now
layers = [sys.argv[1]]

# define our pipeline (standardscaler, feature selection, and linear regression)
lr = linear_model.LinearRegression()
clf = make_pipeline(preprocessing.StandardScaler(), VarianceThreshold(threshold=(0.15)), lr)

for layer in layers:
    print(f'Running layer: {layer}')
    predicted = np.zeros([nperm, ns, ni])
    X = np.load(os.path.join(DCNN_PATH, f'features_{layer}.npy'))

    for perm in range(nperm):
        # get the permuted
        X_ = X[imi[perm,:], :]
        print_progress(perm, nperm)
        for sub in range(ns):
            predicted[perm, sub, :] = cross_val_predict(clf, X_, AB[sub, :], cv=48)

    np.save(os.path.join(DATA_PATH, f'{layer}_prediction_permutations.npy'), predicted)
