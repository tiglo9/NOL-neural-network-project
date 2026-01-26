from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import copy

from activation_functions import logi, softmax, relu
from data_loader import DataLoader
from models import NeuralNetwork
from supplementary import Value, load_mnist, add_noise_to_mnist

from train_and_test import train_network, test_network, train_and_test_p_s, proces_worker  # unified functions

np.set_printoptions(precision=2)

# ============================================================
#                    CONFIGURATION
# ============================================================

train_on_noise = True
test_on_noise = False

train_noise_settings = (50, 0.1)
test_noise_settings = (100, 0.2)

test_on_multiple = True
test_on_mult_p = [0, 25, 50, 75, 100]
test_on_mult_s = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

data_dir = Path(__file__).resolve().parent / "data"

batch_size = 32
learning_rate = 0.0002 
epochs = 2

noise_grid_search = True
grid_search = False  # toggle grid search

p_list = np.arange(0, 100, 5)
s_list = np.arange(0, 0.6, 0.03)
# p_list = [0, 1]
# s_list = [0]

# ============================================================
#                LOAD AND PREPROCESS TRAIN DATA
# ============================================================

train_images, train_y = load_mnist(data_dir, kind="train")
train_images = train_images.reshape(60_000, 784) / 255

if train_on_noise:
    p, s = train_noise_settings
    train_images, train_y, _ = add_noise_to_mnist(train_images, train_y, p, s)

train_labels = np.zeros((60_000, 10))
train_labels[np.arange(60_000), train_y] = 1

# Validation split
validation_subset = 5000
validation_images = train_images[:validation_subset]
validation_labels = train_labels[:validation_subset]
train_images = train_images[validation_subset:]
train_labels = train_labels[validation_subset:]

train_dataset = list(zip(train_images, train_labels))
validation_dataset = list(zip(validation_images, validation_labels))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

train_dataset_size = len(train_dataset)
validation_dataset_size = len(validation_dataset)

# ============================================================
#                LOAD AND PREPROCESS TEST DATA (SINGLE MODE)
# ============================================================

if not test_on_multiple:
    if not test_on_noise:
        test_images, test_y = load_mnist(data_dir, kind="t10k")
        test_images = test_images.reshape(10_000, 784) / 255
    else:
        p, s = test_noise_settings
        test_images = np.load(
            f"{data_dir}/custom_test_sets/noisy_mnist_t10k_p{p}_s{s}_images.npz"
        )["images"]
        test_y = np.load(
            f"{data_dir}/custom_test_sets/noisy_mnist_t10k_p{p}_s{s}_labels.npz"
        )["labels"][:, 0]

    test_labels = np.zeros((len(test_y), 10))
    test_labels[np.arange(len(test_y)), test_y] = 1

    test_dataset = list(zip(test_images, test_labels))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_dataset_size = len(test_dataset)

# ============================================================
#                    MODEL
# ============================================================

neural_network = NeuralNetwork(
    layers=[784, 128, 64, 10],
    activation_functions=[relu, relu, softmax]
)

# ============================================================
#                    MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    if grid_search:
        print("=== Running GRID SEARCH sequentially ===")
        learning_rates = [9e-5]
        epochs = [15]
        results = []

        for lr, ep in zip(learning_rates, epochs):
            print(f"\n--- Training network with learning rate {lr} ---")
            net_copy = copy.deepcopy(neural_network)

            # Train
            train_acc, train_loss, val_acc, val_loss = train_network(
                net_copy,
                train_loader,
                train_dataset_size,
                validation_loader,
                validation_dataset_size,
                learning_rate=lr,
                epochs=ep  # shorter for grid search
            )

            with open("grid_search_results.txt", "a") as f:
                f.write(f"\nLR: {lr}, Epochs: {ep}\n")
                f.write(f"=======================================================\n")
                for epoch in range(1, ep + 1):
                    f.write(
                        f"Epoch {epoch}: "
                        f"Train Acc: {train_acc[epoch-1]:.4f}, "
                        f"Train Loss: {train_loss[epoch-1]:.4f}, "
                        f"Val Acc: {val_acc[epoch-1]:.4f}, "
                        f"Val Loss: {val_loss[epoch-1]:.4f}\n"
                    )
                f.write("=======================================================\n")


            # ---------- PLOTTING ----------

            # Loss: single-axis
            plt.figure()
            plt.title(f"Loss: Train vs Validation (LR={lr})")
            plt.semilogy(np.arange(1, ep+1), train_loss, label="Train")
            plt.semilogy(np.arange(1, ep+1), val_loss, label="Validation")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"loss_single_axis_lr_{lr}.png", dpi=300, bbox_inches="tight")
            plt.close()


            # Loss: dual-axis
            fig_loss, ax1 = plt.subplots()
            ax1.set_title(f"Loss: Train vs Validation (dual-axis) LR={lr}")
            color = "tab:blue"
            ax1.semilogy(np.arange(1, ep+1), train_loss, color=color, label="Train")
            ax1.set_ylabel("Train Loss", color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()
            color = "tab:orange"
            ax2.semilogy(np.arange(1, ep+1), val_loss, color=color, label="Validation")
            ax2.set_ylabel("Validation Loss", color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            fig_loss.tight_layout()
            fig_loss.savefig(f"loss_dual_axis_lr_{lr}.png", dpi=300, bbox_inches="tight")
            plt.close(fig_loss)


            # Accuracy: single-axis
            plt.figure()
            plt.title(f"Accuracy: Train vs Validation (LR={lr})")
            plt.plot(np.arange(1, ep+1), train_acc, label="Train")
            plt.plot(np.arange(1, ep+1), val_acc, label="Validation")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"accuracy_single_axis_lr_{lr}.png", dpi=300, bbox_inches="tight")
            plt.close()


            # Accuracy: dual-axis
            fig_acc, ax1 = plt.subplots()
            ax1.set_title(f"Accuracy: Train vs Validation (dual-axis) LR={lr}")
            color = "tab:blue"
            ax1.plot(np.arange(1, ep+1), train_acc, color=color, label="Train")
            ax1.set_ylabel("Train Accuracy", color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()
            color = "tab:orange"
            ax2.plot(np.arange(1, ep+1), val_acc, color=color, label="Validation")
            ax2.set_ylabel("Validation Accuracy", color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            fig_acc.tight_layout()
            fig_acc.savefig(f"accuracy_dual_axis_lr_{lr}.png", dpi=300, bbox_inches="tight")
            plt.close(fig_acc)


            # Test
            if not test_on_multiple:
                test_results = test_network(
                    net_copy,
                    test_loader=test_loader,
                    test_dataset_size=test_dataset_size,
                    test_on_multiple=False
                )
                test_acc = test_results["accuracy"]
            else:
                test_results = test_network(
                    net_copy,
                    test_on_multiple=True,
                    p_list=test_on_mult_p,
                    s_list=test_on_mult_s,
                    data_dir=str(data_dir),
                    epoch=1
                )
                test_acc = test_results["mean_accuracy"]
            
            with open("grid_search_results.txt", "a") as f:
                f.write("Test Results\n=======================================================\n")
                for p, s, acc, loss in test_results["points"]:
                    f.write(
                        f"[TEST] p={int(p)}, s={s} | acc={acc:.4f}, loss={loss:.4f}\n"
                    )
                f.write(f"=======================================================\n\n")

            # Store results
            results.append({
                "lr": lr,
                "train_acc": train_acc[-1],
                "val_acc": val_acc[-1],
                "test_acc": test_acc
            })

        with open("grid_search_results.txt", "a") as f:
            f.write("\n=== GRID SEARCH RESULTS ===")
            for r in results:
                f.write(
                    f"LR={r['lr']:.4f} | "
                    f"Train Acc={r['train_acc']:.4f} | "
                    f"Val Acc={r['val_acc']:.4f} | "
                    f"Test Acc={r['test_acc']:.4f}\n"
                )

        # Print summary
        print("\n=== GRID SEARCH RESULTS ===")
        for r in results:
            print(
                f"LR={r['lr']:.4f} | "
                f"Train Acc={r['train_acc']:.4f} | "
                f"Val Acc={r['val_acc']:.4f} | "
                f"Test Acc={r['test_acc']:.4f}"
            )

    if noise_grid_search:
        print("=== Running NOISE GRID SEARCH (p, s) ===")

        noise_summary = []

        for p in p_list:
            for s in s_list:
                base_path = Path("noise_grid_search_results")
                (base_path / f"p={p}" / "plots").mkdir(parents=True, exist_ok=True)

                results_dict = train_and_test_p_s(
                    neural_network=neural_network,
                    data_dir="data",
                    path=base_path,
                    p=p,
                    s=s,
                    learning_rate=learning_rate,
                    epochs=epochs
                )

                # Only keep the summary info you care about
                noise_summary.append({
                    "p": p,
                    "s": s,
                    "test_mean_accuracy": results_dict["test_mean_accuracy"],
                    "test_mean_loss": results_dict["test_mean_loss"]
                })

        # Sort by test_mean_accuracy if desired
        noise_summary.sort(key=lambda x: x["test_mean_accuracy"], reverse=True)

        print("\n=== NOISE GRID SEARCH SUMMARY ===")
        for r in noise_summary:
            print(
                f"p={r['p']}, s={r['s']} | "
                f"Mean Test Acc={r['test_mean_accuracy']:.4f}, "
                f"Mean Test Loss={r['test_mean_loss']:.4f}"
            )

        with open("noise_grid_search_results/noise_grid_search_results_summary.txt", "a") as f:
            f.write("=== NOISE GRID SEARCH SUMMARY ===\n")
            for r in noise_summary:
                f.write(
                    f"p={r['p']}, s={r['s']} | "
                    f"Mean Test Acc={r['test_mean_accuracy']:.4f}, "
                    f"Mean Test Loss={r['test_mean_loss']:.4f}\n"
                )   
    else:
        print("=== Running NORMAL training ===")
        train_acc, train_loss, val_acc, val_loss = train_network(
            neural_network,
            train_loader,
            train_dataset_size,
            validation_loader,
            validation_dataset_size,
            learning_rate,
            epochs
        )

        # ============================================================
        #                    PLOTTING
        # ============================================================

        epochs_range = np.arange(1, len(train_loss) + 1)

        plt.figure()
        plt.title("Loss: Train vs Validation")
        plt.semilogy(epochs_range, train_loss, label="Train")
        plt.semilogy(epochs_range, val_loss, label="Validation")
        plt.legend()
        plt.show()

        plt.figure()
        plt.title("Accuracy: Train vs Validation")
        plt.plot(epochs_range, train_acc, label="Train")
        plt.plot(epochs_range, val_acc, label="Validation")
        plt.legend()
        plt.show()

        # ============================================================
        #                    TESTING (NEW UNIFIED WAY)
        # ============================================================

        test_results = test_network(
            neural_network,
            test_loader=test_loader if not test_on_multiple else None,
            test_dataset_size=test_dataset_size if not test_on_multiple else None,
            test_on_multiple=test_on_multiple,
            p_list=test_on_mult_p if test_on_multiple else None,
            s_list=test_on_mult_s if test_on_multiple else None,
            data_dir=str(data_dir),
            epoch=epochs
        )

        if not test_on_multiple:
            print("\n=== FINAL TEST RESULTS ===")
            print(f"Test accuracy: {test_results['accuracy']:.4f}")
            print(f"Test loss:     {test_results['loss']:.4f}")
        else:
            print("Mean accuracy:", test_results["mean_accuracy"])
            print("Mean loss:", test_results["mean_loss"])

        # ============================================================
        #           VISUALIZE RANDOM TEST PREDICTIONS
        # ============================================================

        if not test_on_multiple:
            r = np.random.randint(0, len(test_images) - 10)
            plt.figure(figsize=(15, 10))

            for i in range(9):
                plt.subplot(3, 3, i + 1)
                image = Value(np.array(test_images[r + i]), "x")
                output = neural_network(image)

                plt.imshow(image.data.reshape(28, 28), cmap="gray")
                plt.title(
                    f"True: {test_y[r+i]}, Pred: {np.argmax(output.data)}",
                    fontsize=10
                )
                plt.axis("off")

            plt.tight_layout()
            plt.show()