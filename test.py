from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataLoader
from supplementary import Value, load_mnist

from noise import add_white_noise

# Set printing precision for NumPy so that we don't get needlessly many digits in our answers.
np.set_printoptions(precision=2)

# Get images and corresponding labels from the (fashion-)mnist dataset
data_dir = Path(__file__).resolve().parent / "data"
print(data_dir)
train_images, train_y = load_mnist(data_dir, kind='train')

# Reshape each of the 60 000 images from a 28x28 image into a 784 vector.
# Rescale the values in the 784 to be in [0,1] instead of [0, 255].
#train_images = train_images.reshape(60_000, 784) / 255

n = 16
fig, axis = plt.subplots(4, 4)
labels = np.array([])
for i in range(n):
    j = np.random.random_integers(0,60000)
    labels = np.append(labels, train_y[j])
    img = (train_images[j]/255).reshape((28,28))
    axis[i%4, i//4].imshow(add_white_noise(img, 0.2), cmap='gray', vmin=0, vmax=1)
    axis[i%4, i//4].axis('off')
print(labels)
plt.show()