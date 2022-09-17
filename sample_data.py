import os

import numpy as np
import pandas as pd

path = 'fashion550k'
duplicateids_path = os.path.join(path, 'duplicateids.npy')
trainids_path = os.path.join(path, 'trainids.npy')
validids_path = os.path.join(path, 'validids.npy')
testids_path = os.path.join(path, 'testids.npy')
noisy_path = os.path.join(path, 'annotation/noisy.npy')
cleaned_path = os.path.join(path, 'annotation/verified.npy')

def read_npy(path):
    data = np.load(path)
    print(data.shape)
    print(data)
    
read_npy(duplicateids_path)
read_npy(trainids_path)
read_npy(validids_path)
read_npy(testids_path)
read_npy(noisy_path)
read_npy(cleaned_path)