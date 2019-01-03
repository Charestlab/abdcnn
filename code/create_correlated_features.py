# PERMUTATION TEST DISTRIBUTION
import os
import numpy as np

"""
just a small script to create precomputed correlations between images across
layers
"""

layers = ('conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8')

DCNN_PATH = '../DCNN_features'

layerdims = {
    'conv1': (48, 96, 55*55),
    'conv2': (48, 256, 27*27),
    'conv3': (48, 384, 13*13),
    'conv4': (48, 384, 13*13),
    'conv5': (48, 256, 13*13)}
for layer in layers:
    print('Layer', layer)
    features = np.load(os.path.join(DCNN_PATH, f'features_{layer}.npy'))
    if 'conv' in layer:
        laydims = layerdims[layer]
        features = features.reshape(laydims).mean(2)

    saveto = os.path.join(DCNN_PATH, f'correlated_features_{layer}.npy')
    np.save(saveto, np.corrcoef(features))
