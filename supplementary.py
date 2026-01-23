import numpy as np


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

def test_on_mult_data(neural_network, p=None, s=None, epoch=1):
    from data_loader import DataLoader
    import numpy as np
    from tqdm import tqdm
    from loss_functions import mse_loss
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    
    checked_clean_check = False
    clean_accuracy = None
    points = []

    for p_val in p:
        for s_val in s:
            # Load appropriate test set
            if p_val != 0 and s_val != 0:
                test_images = np.load(
                    f"data/custom_test_sets/noisy_mnist_t10k_p{p_val}_s{s_val}_images.npz"
                )["images"]
                test_y = np.load(
                    f"data/custom_test_sets/noisy_mnist_t10k_p{p_val}_s{s_val}_labels.npz"
                )["labels"][:, 0]
            else:
                # Standard clean test set
                if not checked_clean_check:
                    test_images, test_y = load_mnist('data', kind='t10k')
                    test_images = test_images.reshape(10_000, 784) / 255
                    checked_clean_check = True
                # Use cached clean_accuracy if already computed
                test_images_clean = test_images
                test_y_clean = test_y

            # Prepare one-hot labels
            test_labels = np.zeros((len(test_y), 10))
            test_labels[np.arange(len(test_y)), test_y] = 1

            # Wrap in DataLoader
            test_dataset = list(zip(test_images, test_labels))
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False)
            test_dataset_size = len(test_dataset)

            test_loss = 0.0
            correctly_classified = 0

            for batch in tqdm(test_loader, desc=f"Testing epoch {epoch}, p={p_val}, s={s_val}"):
                images = np.vstack([image for (image, _) in batch])
                labels = np.vstack([label for (_, label) in batch])

                images = Value(images, expr="X")
                labels = Value(labels, expr="Y")

                output = neural_network(images)
                loss = mse_loss(output, labels)
                test_loss += loss.data

                true_classification = np.argmax(labels.data, axis=1)
                predicted_classification = np.argmax(output.data, axis=1)
                correctly_classified += np.sum(true_classification == predicted_classification)

            test_accuracy = correctly_classified / test_dataset_size

            # Cache the clean accuracy for reuse
            if (p_val == 0 or s_val == 0) and clean_accuracy is None:
                clean_accuracy = test_accuracy

            # Fill in points
            if p_val == 0:
                # Accuracy is the same along the entire p=0 line
                for s_fill in s:
                    points.append((0, s_fill, clean_accuracy))
            elif s_val == 0:
                # Accuracy is the same along the entire s=0 line
                for p_fill in p:
                    points.append((p_fill, 0, clean_accuracy))
            else:
                points.append((p_val, s_val, test_accuracy))

            print(f"Test loss and accuracy for p={p_val}, s={s_val}")
            print(f"test loss: {test_loss:.4f}")
            print(f"test accuracy: {test_accuracy:.4f}\n")

    # Convert to numpy array for plotting
    points_array = np.array(points)
    p_vals = points_array[:, 0]
    s_vals = points_array[:, 1]
    test_accs = points_array[:, 2]

    mean_test_acc = np.mean(test_accs)

    print(f"Average test loss and accuracy over all p and s")
    print(f"average test accuracy: {mean_test_acc:.4f}\n")

    # Create grid for interpolation
    grid_p = np.linspace(0, 100, 100)
    grid_s = np.linspace(0, 0.4, 40)
    P, S = np.meshgrid(grid_p, grid_s)

    grid_accuracy = griddata((p_vals, s_vals), test_accs, (P, S), method='cubic')

    plt.figure(figsize=(8, 6))
    plt.imshow(grid_accuracy, extent=(0, 100, 0, 0.4), origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Test Accuracy')
    plt.xlabel('p')
    plt.ylabel('s')
    plt.title('Test Accuracy Heatmap')
    plt.show()

    return points_array