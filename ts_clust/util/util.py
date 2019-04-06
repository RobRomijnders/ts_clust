from collections import Counter
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD


def open_data(direc, ratio_train=0.8, dataset="ECG5000"):
    """Input:
    direc: location of the UCR archive
    ratio_train: ratio to split training and testset
    dataset: name of the dataset in the UCR archive"""
    datadir = direc + '/' + dataset + '/' + dataset
    data_train = np.loadtxt(datadir + '_TRAIN', delimiter=',')
    data_test_val = np.loadtxt(datadir + '_TEST', delimiter=',')[:-1]
    data = np.concatenate((data_train, data_test_val), axis=0)

    N, D = data.shape

    ind_cut = int(ratio_train * N)

    np.random.seed(5627)
    ind = np.random.permutation(N)
    X_train, X_val, y_train, y_val = data[ind[:ind_cut], 1:], \
                                     data[ind[ind_cut:], 1:], \
                                     data[ind[:ind_cut], 0], \
                                     data[ind[ind_cut:], 0]

    base = np.min(y_train)  # Check if data is 0-based
    if base != 0:
        y_train -= base
        y_val -= base

    return X_train, X_val, y_train, y_val


def plot_data(X_train, y_train, plot_row=5):
    counts = dict(Counter(y_train))
    num_classes = len(np.unique(y_train))
    f, axarr = plt.subplots(plot_row, num_classes)
    for c in np.unique(y_train):  # Loops over classes, plot as columns
        c = int(c)
        ind = np.where(y_train == c)
        ind_plot = np.random.choice(ind[0], size=plot_row)
        for n in range(plot_row):  # Loops over rows
            axarr[n, c].plot(X_train[ind_plot[n], :])
            # Only shops axes for bottom row and left column
            if n == 0:
                axarr[n, c].set_title('Class %.0f (%.0f)' % (c, counts[float(c)]))
            if not n == plot_row - 1:
                plt.setp([axarr[n, c].get_xticklabels()], visible=False)
            if not c == 0:
                plt.setp([axarr[n, c].get_yticklabels()], visible=False)
    f.subplots_adjust(hspace=0)  # No horizontal space between subplots
    f.subplots_adjust(wspace=0)  # No vertical space between subplots
    plt.show()
    return


def plot_z_run(z_run, label=None):
    f1, ax1 = plt.subplots(2, 1)

    # First fit a PCA
    PCA_model = TruncatedSVD(n_components=3).fit(z_run)
    z_run_reduced = PCA_model.transform(z_run)
    ax1[0].scatter(z_run_reduced[:, 0], z_run_reduced[:, 1], c=label, marker='*', linewidths=0)
    ax1[0].set_title('PCA on z_run')

    # THen fit a tSNE
    tSNE_model = TSNE(verbose=2, perplexity=80, min_grad_norm=1E-12, n_iter=500)
    z_run_tsne = tSNE_model.fit_transform(z_run)
    ax1[1].scatter(z_run_tsne[:, 0], z_run_tsne[:, 1], c=label, marker='*', linewidths=0)
    ax1[1].set_title('tSNE on z_run')

    plt.show()
    return


def find_best_accuracy(y_pred, y_target):
    num_labels = int(np.max(y_target)) + 1

    y_pred_copy = np.copy(y_pred)

    def iterate_acc():
        for mapping in product(range(num_labels), repeat=num_labels):
            y_pred = np.copy(y_pred_copy)
            np.put(y_pred, np.arange(num_labels), mapping)
            yield np.mean(np.equal(y_pred, y_target))
    best_acc = max(iterate_acc())
    print(best_acc)
