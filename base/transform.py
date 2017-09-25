import numpy as np

def RollingWindow(x, window, dilated_rate=0, stride=1, b=0):
    window = np.concatenate([np.zeros((window.shape[0], dilated_rate)), np.expand_dims(window, axis=1)], axis=1).reshape(-1)

    
