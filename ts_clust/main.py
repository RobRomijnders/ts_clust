# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:43:29 2016

@author: Rob Romijnders
"""

import numpy as np
import tensorflow as tf
import os
from ts_clust.model import Model
from ts_clust.util.util import open_data, plot_data, plot_z_run, find_best_accuracy
from sklearn.cluster import SpectralClustering
tf.logging.set_verbosity(tf.logging.ERROR)
import json


"""Hyperparameters"""
# Please download data here: https://www.cs.ucr.edu/~eamonn/time_series_data/
# and set direc to the location of the data directory
data_direc = 'data/UCR_datasets'
LOG_DIR = "log_tb"  # Directory for the logging

with open('model_settings.json') as f:
    config = json.load(f)

plot_every = 100  # after _plot_every_ GD steps, there's console output
max_iterations = 1000  # maximum number of iterations
dropout = 0.8  # Dropout rate


# Load the data
X_train, X_val, y_train, y_val = open_data(data_direc)

N = X_train.shape[0]
Nval = X_val.shape[0]
D = X_train.shape[1]
config['sl'] = sl = D  # sequence length
print('We have %s observations with %s dimensions' % (N, D))

# Organize the classes
num_classes = int(np.max(y_train)) + 1
for i in range(num_classes):
    print(f'Label {i:2.0f} has {np.mean(np.equal(y_train, i)):8.3f} percentage of labels')


# Plot data   # and save high quality plt.savefig('data_examples.eps', format='eps', dpi=1000)
do_plot_example = False
if do_plot_example:
    plot_data(X_train, y_train)

"""Training time!"""
with tf.Session(graph=tf.Graph()) as sess:
    model = Model(config)
    perf_collect = np.zeros((2, int(np.floor(max_iterations / plot_every))))
    # Proclaim the epochs
    epochs = np.floor(config["batch_size"] * max_iterations / N)
    print('Train with approximately %d epochs' % epochs)

    sess.run(model.init_op)
    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)  # writer for Tensorboard

    step = 0  # Step is a counter for filling the numpy array perf_collect
    for i in range(max_iterations):
        batch_ind = np.random.choice(N, config["batch_size"], replace=False)
        result = sess.run([model.loss, model.loss_seq, model.loss_lat_batch, model.train_step],
                          feed_dict={model.input_placeholder: X_train[batch_ind], model.keep_prob: dropout})

        if i % plot_every == 0:
            # Save train performances
            perf_collect[0, step] = loss_train = result[0]
            loss_train_seq, lost_train_lat = result[1], result[2]

            # Calculate and save validation performance
            batch_ind_val = np.random.choice(Nval, config["batch_size"], replace=False)

            result = sess.run([model.loss, model.loss_seq, model.loss_lat_batch, model.merged],
                              feed_dict={model.input_placeholder: X_val[batch_ind_val], model.keep_prob: 1.0})
            perf_collect[1, step] = loss_val = result[0]
            loss_val_seq, lost_val_lat = result[1], result[2]
            # and save to Tensorboard
            summary_str = result[3]
            writer.add_summary(summary_str, i)
            writer.flush()

            print(f"At {i:6} / {max_iterations:6} "
                  f"train ({loss_train:5.3f}, {loss_train_seq:5.3f}, {lost_train_lat:5.3f}), "
                  f"val ({loss_val:5.3f}, {loss_val_seq:5.3f}, {lost_val_lat:5.3f}) "
                  f"in order (total, seq, lat)")
            step += 1


    # Save the model
    model.saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), step)
