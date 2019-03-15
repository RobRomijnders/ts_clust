import numpy as np
from ts_clust.util.util import plot_z_run


def plot_reps():
    representations = np.load('output_rep/latent_reps.npy')
    print(representations.shape)

    plot_z_run(representations)


if __name__ == '__main__':
    plot_reps()
