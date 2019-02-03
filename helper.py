from math import *
import numpy as np
import os


# Visualizing CompCar, takes indicides and shows in a grid
def cifar_grid(X, Y, inds, n_col, clabels, predictions=None):
    import matplotlib.pyplot as plt
    if predictions is not None:
        if Y.shape != predictions.shape:
            print("Predictions must equal Y in length!\n")
            return (None)
    N = len(inds)
    n_row = int(ceil(1.0 * N / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(10, 10))

    for j in range(n_row):
        for k in range(n_col):
            i_inds = j * n_col + k
            i_data = inds[i_inds]

            axes[j][k].set_axis_off()
            if i_inds < N:
                axes[j][k].imshow(X[i_data, ...], interpolation="nearest")
                encoded_label = np.argmax(Y[i_data, ...])
                label = clabels[encoded_label]
                axes[j][k].set_title(label + "\nEncodedLabel: "+ str(encoded_label))
                if predictions is not None:
                    pred = clabels[np.argmax(predictions[i_data, ...])]
                    if label != pred:
                        label += " n"
                        axes[j][k].set_title(pred, color="red")

    fig.set_tight_layout(True)
    return fig



# Function to find latest checkpoint file
def last_ckpt(dir):
    fl = os.listdir(dir)
    fl = [x for x in fl if x.endswith(".hdf5")]
    cf = ""
    if len(fl) > 0:
        accs = [float(x.split("-")[3][0:-5]) for x in fl]
        m = max(accs)
        iaccs = [i for i, j in enumerate(accs) if j == m]
        fl = [fl[x] for x in iaccs]
        epochs = [int(x.split("-")[2]) for x in fl]
        cf = fl[epochs.index(max(epochs))]
        cf = os.path.join(dir, cf)

    return cf