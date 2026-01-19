
from data_loader import DataLoader
from supplementary import load_mnist
from pathlib import Path
import numpy as np
import random 

def add_white_noise(img, sigma):
    return np.clip(img + np.random.normal(0, sigma, img.shape), 0, 1)


def add_noise_to_mnist(percentage=0.75, sigma=0.2, kind='train', data_dir=None):
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "data"

    # Load MNIST dataset
    images, labels = load_mnist(data_dir, kind=kind)
    images = images.reshape(len(images), -1) / 255  # normalize to [0,1]
    noisy_images = images.copy()

    n = len(images)
    k = round(percentage * n)

    # Randomly select k indices to add noise
    noisy_indices = random.sample(range(n), k)
    for idx in noisy_indices:
        noisy_images[idx] = add_white_noise(noisy_images[idx], sigma)

    return noisy_images, labels, noisy_indices