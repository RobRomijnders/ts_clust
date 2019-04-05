import tensorflow as tf
import json
from ts_clust.util.util import open_data, plot_z_run
from os.path import join
import numpy as np


def extract_reps(data_dir, log_dir, conf):
    dataset_name = "ChlorineConcentration"
    log_dir = join(log_dir, dataset_name)

    # Load the data
    # TODO (rob) use same random seed at training, otherwise we have leakage!
    X_train, X_val, y_train, y_val = open_data(data_dir, dataset=dataset_name)
    Nval = X_val.shape[0]

    with tf.Session() as sess:
        # Restore the model and the placeholders
        # TODO (rob) simply get the latest checkpoint here
        saver = tf.train.import_meta_graph(join(log_dir, "model.ckpt-10.meta"))
        saver.restore(sess, join(log_dir, "model.ckpt-10"))

        input_ph = tf.get_collection('input')[0]
        latent_representation_ph = tf.get_collection('latent_rep')[0]
        keep_prob_ph = tf.get_collection('keep_prob')[0]

        # Extract the latent space coordinates of the validation set
        start = 0
        latent_representations = []

        while start + conf["batch_size"] < Nval:
            run_ind = range(start, start + conf["batch_size"])
            z_mu_fetch = sess.run(latent_representation_ph, feed_dict={input_ph: X_val[run_ind], keep_prob_ph: 1.0})
            latent_representations.append(z_mu_fetch)
            start += conf["batch_size"]

        latent_representations = np.concatenate(latent_representations, axis=0)
        np.save(f'output_rep/latent_reps_{dataset_name}.npy', latent_representations)

        label = y_val[:start]
        np.save(f'output_rep/latent_reps_{dataset_name}_label.npy', label)

        # plot_z_run(latent_representations, label)


if __name__ == '__main__':
    # Please download data here: https://www.cs.ucr.edu/~eamonn/time_series_data/
    # and set direc to the location of the data directory
    data_direc = 'data/UCR_datasets'
    LOG_DIR = "log_tb"  # Directory for the logging

    with open('model_settings.json') as f:
        config = json.load(f)

    extract_reps(data_direc, LOG_DIR, config)
