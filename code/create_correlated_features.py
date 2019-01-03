# PERMUTATION TEST DISTRIBUTION
import os
import numpy as np

"""
just a small script to create precomputed correlations between images across
layers
"""

layers = ('conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8')

DCNN_PATH = '../DCNN_features'

for layer in layers:
    print('Layer', layer)
    features = np.load(os.path.join(DCNN_PATH, f'features_{layer}.npy'))

    saveto = os.path.join(DCNN_PATH, f'correlated_features_{layer}.npy')
    np.save(saveto, np.corrcoef(features))
