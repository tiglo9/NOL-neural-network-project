import numpy as np
import random

class Value:
    def __init__(self, data, expr: str = "", children=()):
        self.expr = expr
        if isinstance(data, Value):
            self.data = data.data
        else:
            self.data = data
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self.children = set(children)

    def _backward_gradient_step(self):
        pass

    def backward(self, primary=True, direction=None):
        if primary:
            if direction is not None:
                self.grad = direction
            else:
                self.grad = np.ones_like(self.grad)

        self._backward_gradient_step()

        for child in self.children:
            child.backward(primary=False)

    def __add__(self, other):
        """This function is called when you add two Value instances with each other. The arguments `self` and
        `other` will be the left Value instance and the right Value instance respectfully.
        """
        result = Value(self.data + other.data, f"({self.expr}+{other.expr})", (self, other))

        def _backward_gradient_step():
            if self.data.ndim == 2 and other.data.ndim == 1:  # matrix + vector
                self.grad += result.grad
                other.grad += np.sum(result.grad, axis=0)
            elif self.data.ndim == 1 and other.data.ndim == 2:  # vector + matrix
                self.grad += np.sum(result.grad, axis=0)
                other.grad += result.grad
            else:  # 2 matrices or 2 vectors
                self.grad += result.grad
                other.grad += result.grad

        result._backward_gradient_step = _backward_gradient_step
        return result

    def __mul__(self, other):
        """This function is called when you multiply two Value instances with each other. The arguments `self` and
        `other` will be the left Value instance and the right Value instance respectfully.
        """
        result = Value(self.data * other.data, f"{self.expr}*{other.expr}", (self, other))

        def _backward_gradient_step():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad

        result._backward_gradient_step = _backward_gradient_step
        return result

    def __matmul__(self, other):
        """This function is called when you matrix multiply two Value instances with each other. The arguments `self`
        and `other` will be the left Value instance and the right Value instance respectfully.
        """
        result = Value(self.data @ other.data, f"{self.expr}@{other.expr}", (self, other))

        def _backward_gradient_step():
            if result.data.ndim == 0:  # 2 vectors ==> scalar
                self.grad += result.grad * other.data
                other.grad += self.data * result.grad
            elif self.data.ndim == 2 and other.data.ndim == 1:  # matrix @ vector ==> vector
                self.grad += np.outer(result.grad, other.data)
                other.grad += self.data.T @ result.grad
            elif self.data.ndim == 1 and other.data.ndim == 2:  # vector @ matrix ==> vector
                self.grad += result.grad @ other.data.T
                other.grad += np.outer(self.data, result.grad)
            else:  # 2 matrices ==> matrix
                self.grad += result.grad @ other.data.T
                other.grad += self.data.T @ result.grad

        result._backward_gradient_step = _backward_gradient_step
        return result

    def reset_grad(self):
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        for child in self.children:
            child.reset_grad()

    def __repr__(self):
        return f"Value({self.data})"

    def __getitem__(self, key):
        return self.data[key]


def allclose(x1: Value | np.ndarray, x2: Value | np.ndarray, *args, **kwargs) -> bool:
    return np.allclose(
        x1.data if isinstance(x1, Value) else x1,
        x2.data if isinstance(x2, Value) else x2,
        *args,
        **kwargs
    )


def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def compare_networks(p_vals, s_vals, loss_vals, accuracy_vals):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    # Create a grid for interpolation
    p_grid = np.linspace(min(p_vals), max(p_vals), 200)
    s_grid = np.linspace(min(s_vals), max(s_vals), 200)
    P, S = np.meshgrid(p_grid, s_grid)

    # Interpolate values onto the grid
    grid_accuracy = griddata((p_vals, s_vals), accuracy_vals, (P, S), method='cubic')
    grid_loss = griddata((p_vals, s_vals), loss_vals, (P, S), method='cubic')

    # Set fixed color limits
    acc_min, acc_max = 0, 1           # Accuracy between 0 and 1
    loss_min, loss_max = np.min(loss_vals), np.max(loss_vals)  # or set fixed range if desired

    # Accuracy heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(grid_accuracy, extent=(min(p_vals), max(p_vals), min(s_vals), max(s_vals)), 
               origin='lower', aspect='auto', cmap='viridis', vmin=acc_min, vmax=acc_max)
    plt.colorbar(label='Test Accuracy')
    plt.xlabel('p')
    plt.ylabel('s')
    plt.title('Test Accuracy Heatmap')
    plt.show()

    # Loss heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(grid_loss, extent=(min(p_vals), max(p_vals), min(s_vals), max(s_vals)), 
               origin='lower', aspect='auto', cmap='magma', vmin=loss_min, vmax=loss_max)
    plt.colorbar(label='Test Loss')
    plt.xlabel('p')
    plt.ylabel('s')
    plt.title('Test Loss Heatmap')
    plt.show()


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