import numpy as np
def add_white_noise(img, sigma):
    return np.clip(img + np.random.normal(0, sigma, img.shape), 0, 1)
