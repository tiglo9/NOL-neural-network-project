
import numpy as np
import random
from noise import add_white_noise, add_noise_to_mnist
from pathlib import Path
import matplotlib.pyplot as plt

# Set printing precision for NumPy
np.set_printoptions(precision=2)


def save_noisy_dataset(save_path, noisy_images, extended_labels):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)  # create folder if it doesn't exist
    np.savez_compressed(save_path, images=noisy_images, labels=extended_labels)
    print(f"Saved noisy dataset to {save_path}")


# --- Main loop ---
percentages = [0.25, 0.5, 0.75, 1]
sigmas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

results = {}
save_sets = False  # <-- Keep this True to save datasets

for p in percentages:
    for s in sigmas:
        noisy_images, labels, noisy_indices = add_noise_to_mnist(percentage=p, sigma=s)
        
        # Create noise flag array
        noise_flags = np.zeros(len(labels), dtype=int)
        noise_flags[noisy_indices] = 1
        
        # Combine labels with noise flag
        extended_labels = np.column_stack((labels, noise_flags))
        
        results[(p, s)] = (noisy_images, extended_labels)
        print(f"Added noise with percentage={p}, sigma={s}")
        
        if save_sets:
            base_path = Path(f"data/custom/noisy_mnist_t10k_p{int(p*100)}_s{s}")
            base_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save images separately
            images_path = base_path.with_name(base_path.name + "_images.npz")
            np.savez_compressed(images_path, images=noisy_images)
            
            # Save labels separately
            labels_path = base_path.with_name(base_path.name + "_labels.npz")
            np.savez_compressed(labels_path, labels=extended_labels)
            
            print(f"Saved images to {images_path}")
            print(f"Saved labels to {labels_path}")


# --- Plot a few samples ---
plot_fig = False

if plot_fig:
    num_plot = 16  # number of images to show per figure
    for (p, s), (images, labels) in results.items():
        plt.figure(figsize=(12, 12))
        
        for i in range(num_plot):
            j = np.random.randint(0, len(images))
            img = images[j].reshape(28, 28)
            mnist_label = labels[j, 0]
            noise_flag = labels[j, 1]
            
            plt.subplot(4, 4, i + 1)
            plt.imshow(img, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            
            plt.text(
                9, 33,  # x, y in pixel coordinates
                f"Label: {mnist_label}\nNoise: {noise_flag}",
                color='black', fontsize=9
            )
        
        plt.suptitle(f"Noise: {int(p*100)}% images, sigma={s}", fontsize=14)
        plt.subplots_adjust(hspace=0.5)
        plt.show()
