import numpy as np

def RollingWindow(x, window, dilated_rate=0, stride=1, b=0, padding=0):
    x = np.array([0]*padding + list(x))
    if dilated_rate > 0:
        window = np.concatenate([np.zeros((window.shape[0], dilated_rate)), np.expand_dims(window, axis=1)], axis=1).reshape(-1)
        window = window[dilated_rate:]
    res = []
    for i in range(len(x)):
        if i * stride - b + 1 - len(window) >= 0:
            res.append(np.sum(x[i * stride - b + 1 - len(window): i * stride - b + 1] * window))
        elif i * stride - b + 1 > 0:
            res.append(np.sum(x[0: i * stride - b + 1] * window[len(window) + b - 1 - i * stride:]))
    return np.array(res)
    
    
