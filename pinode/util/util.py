import random
import numpy as np
import torch
from copy import deepcopy


def _fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def add_noise(data, std_scale=0.3):
    data_noisy = data.copy()
    for i in range(data.shape[1]):
        data_noisy[:,i] = data[:,i] + np.random.normal(0, std_scale*np.std(data[:,i], axis=0), data[:,i].shape)
    return data_noisy

def format_elapsed_time(tic, toc):
    """Small function to print the time elapsed between tic and toc in a nice manner
    """

    diff = toc - tic
    hours = int(diff // 3600)
    minutes = int((diff - 3600 * hours) // 60)
    seconds = str(int(diff - 3600 * hours - 60 * minutes))

    # Put everything in strings
    hours = str(hours)
    minutes = str(minutes)

    # Add a zero for one digit numbers for consistency
    if len(hours) == 1:
        hours = '0' + hours
    if len(minutes) == 1:
        minutes = '0' + minutes
    if len(seconds) == 1:
        seconds = '0' + seconds

    # Final nice looking print
    return f"{hours}:{minutes}:{seconds}"

def normalize(U, X, U_val, X_val):
    max_ = [X[:,:,i].max() for i in range(4)]
    min_ = [X[:,:,i].min() for i in range(4)]

    norm_X = deepcopy(X)
    norm_X_val = deepcopy(X_val)
    for i in range(4):
        norm_X[:,:,i] = (X[:,:,i] - min_[i]) / (max_[i] - min_[i]) * 0.8 + 0.1 
        norm_X_val[:,:,i] = (X_val[:,:,i] - min_[i]) / (max_[i] - min_[i]) * 0.8 + 0.1 

    norm_U = deepcopy(U)
    norm_U = U / (max_[-1] - min_[-1]) * 0.8
    norm_U_val = deepcopy(norm_U)
    
    return norm_U, norm_X, norm_U_val, norm_X_val, (min_, max_)

def inverse_normalize(X, min_, max_):

    norm_X = X
    for i in range(4):
        norm_X[:,:,i] = (X[:,:,i] - 0.1) * (max_[i] - min_[i]) / 0.8 + min_[i] 
    
    return norm_X
    