from multiprocessing import Process
from pathlib import Path
from train_and_test import train_and_test_p_s


if __name__ == "__main__":
    ps = np.arange(0, 100, 5)
    ss = np.arange(0, 0.6, 0.03)
    ps = [20]
    ss = [0,2]
    processes = []
    for p in ps:
        for s in ps:
            processes.append(Process(target = train_and_test_p_s, args=(
                path,
                p,
                s
                ))
