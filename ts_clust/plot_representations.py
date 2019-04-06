import numpy as np
from ts_clust.util.util import plot_z_run


def plot_reps():
    dataset_name = 'ECG5000'
    representations = np.load(f'output_rep/latent_reps_{dataset_name}.npy')
    labels = np.load(f'output_rep/latent_reps_{dataset_name}_label.npy')

    print(representations.shape)

    plot_z_run(representations, label=labels)


if __name__ == '__main__':
    plot_reps()
