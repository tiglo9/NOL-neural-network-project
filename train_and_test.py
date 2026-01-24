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
        epochs=1
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


