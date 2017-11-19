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
    
def getKalmanFilter(sig_ksi, sig_eta, z0): 
    def kalman(z_k1, u_k):
        kalman.Ee = (kalman.sig_eta**2) * (kalman.Ee + kalman.sig_ksi**2) / (kalman.Ee + kalman.sig_eta**2 + kalman.sig_ksi**2)
        kalman.K = kalman.Ee / kalman.sig_eta ** 2
        kalman.x = kalman.K * z_k1 + (1 - kalman.K) * (kalman.x + u_k)

        return kalman.x
    kalman.Ee = sig_eta ** 2
    kalman.x = z0
    kalman.K = 0
    return kalman

def getExample_seq2seq(arr, window_size, idx, count=1):
    return arr[idx:idx+window_size], arr[idx+window_size+1:idx+window_size+count+1]

