import cv2
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_normalize_map(atten_map, w, h, is_norm=True):
    atten_map = cv2.resize(atten_map, dsize=(w, h))
    if is_norm:
        min_val = np.min(atten_map)
        max_val = np.max(atten_map)
        atten_norm = (atten_map - min_val) / (max_val - min_val + 1e-10)
    else:
        threshold = 38.4
        scale = 8.372
        atten_norm = sigmoid((atten_map - threshold) / scale)
    return atten_norm