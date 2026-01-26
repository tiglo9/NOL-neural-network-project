def train_network(neural_network, train_loader, train_dataset_size,
                  validation_loader, validation_dataset_size,
                  learning_rate, epochs):

    import numpy as np
    from tqdm import tqdm
    from loss_functions import mse_loss
    from supplementary import Value

    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []

    # Reset Adam ONCE before training
    neural_network.reset_adam_params()

    for epoch in range(1, epochs + 1):
        # ===== TRAINING =====
        train_loss = 0.0
        correctly_classified = 0

        for batch in tqdm(train_loader, desc=f"Training epoch {epoch}"):
            neural_network.reset_gradients()

            images = np.vstack([x for (x, _) in batch])
            labels = np.vstack([y for (_, y) in batch])

            images = Value(images, expr="X")
            labels = Value(labels, expr="Y")

            output = neural_network(images)
            loss = mse_loss(output, labels)
            loss.backward()
            neural_network.adam_descent(learning_rate, 1e-10)

            train_loss += loss.data

            true = np.argmax(labels.data, axis=1)
            pred = np.argmax(output.data, axis=1)
            correctly_classified += np.sum(true == pred)

        train_losses.append(train_loss)
        train_accuracies.append(correctly_classified / train_dataset_size)

        print(f"[TRAIN] Acc: {train_accuracies[-1]:.4f}, Loss: {train_loss:.4f}")

        # ===== VALIDATION =====
        validation_loss = 0.0
        correctly_classified = 0

        for batch in tqdm(validation_loader, desc=f"Validation epoch {epoch}"):
            images = np.vstack([x for (x, _) in batch])
            labels = np.vstack([y for (_, y) in batch])

            images = Value(images, expr="X")
            labels = Value(labels, expr="Y")

            output = neural_network(images)
            loss = mse_loss(output, labels)

            validation_loss += loss.data

            true = np.argmax(labels.data, axis=1)
            pred = np.argmax(output.data, axis=1)
            correctly_classified += np.sum(true == pred)

        validation_losses.append(validation_loss)
        validation_accuracies.append(correctly_classified / validation_dataset_size)

        print(f"[VAL]   Acc: {validation_accuracies[-1]:.4f}, Loss: {validation_loss:.4f}\n")

    return train_accuracies, train_losses, validation_accuracies, validation_losses

def test_network(
    neural_network,
    test_loader=None,
    test_dataset_size=None,
    *,
    test_on_multiple=False,
    p_list=None,
    s_list=None,
    data_dir="data",
    batch_size=32,
    epoch=1,
    plot_heatmap=True
):
    import numpy as np
    from tqdm import tqdm
    from loss_functions import mse_loss
    from supplementary import Value, load_mnist
    from data_loader import DataLoader
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    # ============================================================
    #               SINGLE STANDARD TEST SET MODE
    # ============================================================
    if not test_on_multiple:
        if test_loader is None or test_dataset_size is None:
            raise ValueError("test_loader and test_dataset_size must be provided when test_on_multiple=False")

        test_loss = 0.0
        correctly_classified = 0

        for batch in tqdm(test_loader, desc="Testing"):
            images = np.vstack([x for (x, _) in batch])
            labels = np.vstack([y for (_, y) in batch])

            images = Value(images, expr="X")
            labels = Value(labels, expr="Y")

            output = neural_network(images)
            loss = mse_loss(output, labels)

            test_loss += loss.data

            true = np.argmax(labels.data, axis=1)
            pred = np.argmax(output.data, axis=1)
            correctly_classified += np.sum(true == pred)

        accuracy = correctly_classified / test_dataset_size

        print(f"[TEST] Loss: {test_loss:.4f}")
        print(f"[TEST] Accuracy: {accuracy:.4f}")

        return {
            "mode": "single",
            "loss": test_loss,
            "accuracy": accuracy
        }

    # ============================================================
    #               MULTI (p, s) ROBUSTNESS MODE
    # ============================================================
    else:
        if p_list is None or s_list is None:
            raise ValueError("p_list and s_list must be provided when test_on_multiple=True")

        points = []

        for p_val in p_list:
            for s_val in s_list:
                # Load test data
                if p_val != 0 and s_val != 0:
                    test_images = np.load(
                        f"{data_dir}/custom_test_sets/noisy_mnist_t10k_p{p_val}_s{s_val}_images.npz"
                    )["images"]
                    test_y = np.load(
                        f"{data_dir}/custom_test_sets/noisy_mnist_t10k_p{p_val}_s{s_val}_labels.npz"
                    )["labels"][:, 0]
                else:
                    test_images, test_y = load_mnist(data_dir, kind="t10k")
                    test_images = test_images.reshape(10_000, 784) / 255

                # One-hot encode
                test_labels = np.zeros((len(test_y), 10))
                test_labels[np.arange(len(test_y)), test_y] = 1

                test_dataset = list(zip(test_images, test_labels))
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                test_dataset_size = len(test_dataset)

                test_loss = 0.0
                correctly_classified = 0

                for batch in tqdm(test_loader, desc=f"Testing for epoch {epoch}, p={p_val}, s={s_val}"):
                    images = np.vstack([x for (x, _) in batch])
                    labels = np.vstack([y for (_, y) in batch])

                    images = Value(images, expr="X")
                    labels = Value(labels, expr="Y")

                    output = neural_network(images)
                    loss = mse_loss(output, labels)

                    test_loss += loss.data

                    true = np.argmax(labels.data, axis=1)
                    pred = np.argmax(output.data, axis=1)
                    correctly_classified += np.sum(true == pred)

                accuracy = correctly_classified / test_dataset_size
                points.append((p_val, s_val, accuracy, test_loss))

                print(
                    f"[TEST] p={p_val}, s={s_val} | "
                    f"acc={accuracy:.4f}, loss={test_loss:.4f}"
                )

        points_array = np.array(points)

        # Optional heatmap
        if plot_heatmap:
            p_vals = points_array[:, 0]
            s_vals = points_array[:, 1]
            accs   = points_array[:, 2]

            grid_p = np.linspace(min(p_vals), max(p_vals), 100)
            grid_s = np.linspace(min(s_vals), max(s_vals), 40)
            P, S = np.meshgrid(grid_p, grid_s)

            grid_acc = griddata((p_vals, s_vals), accs, (P, S), method="cubic")

            plt.figure(figsize=(8, 6))
            plt.imshow(grid_acc, extent=(min(p_vals), max(p_vals), min(s_vals), max(s_vals)),
                       origin="lower", aspect="auto", cmap="viridis")
            plt.colorbar(label="Test Accuracy")
            plt.xlabel("p")
            plt.ylabel("s")
            plt.title("Test Accuracy Heatmap")
            plt.show()

        return {
            "mode": "multiple",
            "points": points_array,
            "mean_accuracy": np.mean(points_array[:, 2]),
            "mean_loss": np.mean(points_array[:, 3])
        }


def grid_worker(args):
    net, lr, train_loader, train_size, val_loader, val_size, test_loader, test_size = args

    train_acc, train_loss, val_acc, val_loss = train_network(
        net,
        train_loader,
        train_size,
        val_loader,
        val_size,
        lr,
        epochs=1,
        plot_heatmap=False
    )

    test_results = test_network(
        net,
        test_loader=test_loader,
        test_dataset_size=test_size,
        test_on_multiple=False
    )

    return {
        "lr": lr,
        "train_acc": train_acc[-1],
        "val_acc": val_acc[-1],
        "test_acc": test_results["accuracy"],
    }

def train_and_test_p_s(neural_network, data_dir, path, p, s, learning_rate = 0.0002, epochs = 20, batch_size = 32):
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import copy
    from scipy.interpolate import griddata
    from activation_functions import logi, softmax
    from data_loader import DataLoader
    from models import NeuralNetwork
    from supplementary import Value, load_mnist, add_noise_to_mnist

    print(f"\n--- Training with noise: p={p}, s={s} ---")

    lr = learning_rate
    ep = epochs

    noise_results = {}

    # Reload clean MNIST every run
    train_images, train_y = load_mnist(data_dir, kind="train")
    train_images = train_images.reshape(60_000, 784) / 255

    # Apply noise
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

    # Fresh network each time
    net = copy.deepcopy(neural_network)

    # Train
    train_acc, train_loss, val_acc, val_loss = train_network(
        net,
        train_loader,
        train_dataset_size,
        validation_loader,
        validation_dataset_size,
        learning_rate=learning_rate,
        epochs=epochs
    )


    # Loss: single-axis
    plt.figure()
    plt.title(f"Loss: Train vs Validation (LR={lr})")
    plt.semilogy(np.arange(1, ep+1), train_loss, label="Train")
    plt.semilogy(np.arange(1, ep+1), val_loss, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{path}/p={p}/plots/loss_single_axis_p={p}_s={s}.png", dpi=300, bbox_inches="tight")
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
    fig_loss.savefig(f"{path}/p={p}/plots/loss_dual_axis_p={p}_s={s}.png", dpi=300, bbox_inches="tight")
    plt.close(fig_loss)


    # Accuracy: single-axis
    plt.figure()
    plt.title(f"Accuracy: Train vs Validation (p={p} and s={s})")
    plt.plot(np.arange(1, ep+1), train_acc, label="Train")
    plt.plot(np.arange(1, ep+1), val_acc, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{path}/p={p}/plots/accuracy_single_axis_p_{p}_s={s}.png", dpi=300, bbox_inches="tight")
    plt.close()


    # Accuracy: dual-axis
    fig_acc, ax1 = plt.subplots()
    ax1.set_title(f"Accuracy: Train vs Validation (dual-axis) p={p} and s={s}")
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
    fig_acc.savefig(f"{path}/p={p}/plots/accuracy_dual_axis_lr_p={p}_s={s}.png", dpi=300, bbox_inches="tight")
    plt.close(fig_acc)


    # Test on your multiple-noise benchmark
    test_results = test_network(
        net,
        test_on_multiple=True,
        p_list=[0, 25, 50, 75, 100],
        s_list=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        data_dir=str(data_dir),
        epoch=epochs,
        plot_heatmap=False
    )

    test_points_array = test_results["points"]
    test_mean_accuracy = test_results["mean_accuracy"]
    test_mean_loss = test_results["mean_loss"]

    p_vals = test_points_array[:, 0]
    s_vals = test_points_array[:, 1]
    accs   = test_points_array[:, 2]

    grid_p = np.linspace(min(p_vals), max(p_vals), 100)
    grid_s = np.linspace(min(s_vals), max(s_vals), 40)
    P, S = np.meshgrid(grid_p, grid_s)

    grid_acc = griddata((p_vals, s_vals), accs, (P, S), method="cubic")

    plt.figure(figsize=(8, 6))
    plt.imshow(grid_acc, extent=(min(p_vals), max(p_vals), min(s_vals), max(s_vals)),
                origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(label="Test Accuracy")
    plt.plot(p, s, "ro", markersize=6)
    plt.xlabel("p")
    plt.ylabel("s")
    plt.title("Test Accuracy Heatmap")
    plt.savefig(f"{path}/p={p}/plots/Test_accuracy_heatmap_p={p}_s={s}.png")
    plt.close()

    with open(f"{path}/p={p}/test_results_s={s}.txt", "a") as f:
        f.write("Test Results\n=======================================================\n")
        for p1, s1, acc, loss in test_results["points"]:
            f.write(
                f"[TEST] p={int(p1)}, s={s1} | acc={acc:.4f}, loss={loss:.4f}\n"
            )
        f.write(f"=======================================================\n\n")

    from pathlib import Path
    save_path = Path(path) / "networks" / f"p={p}_s={s}"
    save_path.mkdir(parents=True, exist_ok=True)  # make sure folder exists
    net.save(save_path)


    return {
    "p": p,
    "s": s,
    "points": test_points_array,
    "test_mean_accuracy": test_mean_accuracy,
    "test_mean_loss": test_mean_loss
    } 

def make_network():
    from models import NeuralNetwork
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import copy

    from activation_functions import logi, softmax
    from data_loader import DataLoader
    from models import NeuralNetwork
    from supplementary import Value, load_mnist, add_noise_to_mnist
    return NeuralNetwork(
        layers=[784, 256, 128, 64, 10],
        activation_functions=[logi, logi, logi, softmax]
    )


def proces_worker(p, s, learning_rate=0.0002, epochs=1):
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import copy
    from scipy.interpolate import griddata
    from activation_functions import logi, softmax
    from data_loader import DataLoader
    from models import NeuralNetwork
    from supplementary import Value, load_mnist, add_noise_to_mnist

    base_path = Path("noise_grid_search_results")
    (base_path / f"p={p}" / "plots").mkdir(parents=True, exist_ok=True)

    train_and_test_p_s(
        neural_network=make_network(),
        data_dir="data",
        path=base_path,
        p=p,
        s=s,
        learning_rate=learning_rate,
        epochs=epochs
    )


    # Optional: global summary file
    with open(base_path / "noise_grid_search_summary.txt", "w") as f:
        f.write("p\ts\tMean Accuracy\tMean Loss\n")
        for r in noise_results:
            f.write(
                f"{r['p']}\t{r['s']}\t"
                f"{r['test_mean_accuracy']:.4f}\t"
                f"{r['test_mean_loss']:.4f}\n"
            )

    print("\n=== NOISE GRID SEARCH COMPLETE ===")
    print(f"All results saved in: {base_path.resolve()}")
