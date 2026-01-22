
from data_loader import DataLoader
from supplementary import load_mnist
from pathlib import Path
import numpy as np
import random 

def add_white_noise(image, sigma):
    noise = np.random.normal(loc=0.0, scale=sigma, size=image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0.0, 1.0)


def add_noise_to_mnist(images, labels,percentage, sigma):
    # Load MNIST dataset
    images = images.reshape(len(images), -1)  # normalize to [0,1]
    noisy_images = images.copy()

    n = len(images)
    k = round(percentage / 1000 * n)

    # Randomly select k indices to add noise
    noisy_indices = random.sample(range(n), k)
    for idx in noisy_indices:
        noisy_images[idx] = add_white_noise(noisy_images[idx], sigma)

    return noisy_images, labels, noisy_indices