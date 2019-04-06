# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:43:29 2016

@author: Rob Romijnders

"""
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell


class Model:
    def __init__(self, config):
        # Hyperparameters
        num_layers = config['num_layers']
        hidden_size = config['hidden_size']
        max_grad_norm = config['max_grad_norm']
        batch_size = config['batch_size']
        sl = config['sl']
        crd = config['crd']
        num_l = config['num_l']
        learning_rate = config['learning_rate']
        self.sl = sl
        self.batch_size = batch_size

        # Nodes for the input variables
        self.input_placeholder = tf.placeholder("float", shape=[None, sl], name='Input_data')
        self.x_exp = tf.expand_dims(self.input_placeholder, 1)
        self.keep_prob = tf.placeholder("float")

        with tf.variable_scope("Encoder"):
            # Th encoder cell, multi-layered with dropout
            cell_enc = tf.contrib.rnn.MultiRNNCell([LSTMCell(hidden_size) for _ in range(num_layers)])
            cell_enc = tf.contrib.rnn.DropoutWrapper(cell_enc, output_keep_prob=self.keep_prob)

            outputs_enc, _ = tf.nn.static_rnn(cell_enc,
                                              inputs=tf.unstack(self.x_exp, axis=2),
                                              dtype=tf.float32)
            cell_output = outputs_enc[-1]  # Use the hidden state at the final time step. Therefore, index -1

            # layer for mean of z
            W_mu = tf.get_variable('W_mu', [hidden_size, num_l])
            b_mu = tf.get_variable('b_mu', [num_l])

            # For all intents and purposes, self.z_mu is the Tensor containing the hidden representations
            # I got many questions over email about this. If you want to do visualization, clustering or subsequent
            #   classification, then use this z_mu
            self.z_mu = tf.nn.xw_plus_b(cell_output, W_mu, b_mu, name='z_mu')  # mu, mean, of latent space

            # Train the point in latent space to have zero-mean and unit-variance on batch basis
            lat_mean, lat_var = tf.nn.moments(self.z_mu, axes=[1])
            self.loss_lat_batch = tf.reduce_mean(tf.square(lat_mean) + lat_var - tf.log(lat_var) - 1)

        with tf.name_scope("Lat_2_dec"):
            # layer to generate initial state
            W_state = tf.get_variable('W_state', [num_l, hidden_size])
            b_state = tf.get_variable('b_state', [hidden_size])
            z_state = tf.nn.xw_plus_b(self.z_mu, W_state, b_state, name='z_state')  # mu, mean, of latent space

        with tf.variable_scope("Decoder"):
            # The decoder, also multi-layered
            cell_dec = tf.contrib.rnn.MultiRNNCell([LSTMCell(hidden_size) for _ in range(num_layers)])

            # Initial state
            initial_state_dec = tuple([(z_state, z_state)] +
                                      [(tf.zeros_like(z_state), tf.zeros_like(z_state))]* (num_layers - 1))
            dec_inputs = tf.unstack(tf.zeros_like(self.x_exp), axis=2)
            outputs_dec, _ = tf.nn.static_rnn(cell_dec,
                                              inputs=dec_inputs,
                                              initial_state=initial_state_dec)
        with tf.name_scope("Out_layer"):
            params_o = 2 * crd  # Number of coordinates + variances
            W_o = tf.get_variable('W_o', [hidden_size, params_o])
            b_o = tf.get_variable('b_o', [params_o, 1])
            outputs = tf.stack(outputs_dec, axis=-1)  # tensor in [batch_size, hidden_size, seq_len]

            # Get the parameters for the Gaussian distro's
            h_out = tf.einsum('ijk,jp->ipk', outputs, W_o) + b_o  # Multiply over the second dimension
            h_mu, h_sigma_log = tf.unstack(h_out, axis=1)
            h_sigma = tf.exp(h_sigma_log)

            # Calculate the log probability of the input sequence under the parametrized Gaussians
            dist = tf.contrib.distributions.Normal(h_mu, h_sigma)
            px = dist.log_prob(self.input_placeholder)
            self.loss_seq = tf.reduce_mean(-px)

        with tf.name_scope("train"):
            # Use learning rte decay
            global_step = tf.Variable(0, trainable=False)
            lr = tf.train.exponential_decay(learning_rate, global_step, 1000, 0.1, staircase=False)

            self.loss = self.loss_seq + 0.01 * self.loss_lat_batch

            # Route the gradients so that we can plot them on Tensorboard
            tvars = tf.trainable_variables()
            # We clip the gradients to prevent explosion
            grads = tf.gradients(self.loss, tvars)
            grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
            self.numel = tf.constant([[0]])

            # And apply the gradients
            optimizer = tf.train.AdamOptimizer(lr)
            gradients = zip(grads, tvars)
            self.train_step = optimizer.apply_gradients(gradients, global_step=global_step)

            self.numel = tf.constant([[0]])
        tf.summary.tensor_summary('lat_state', self.z_mu)

        tf.add_to_collection('hello', [self.input_placeholder, self.z_mu])

        tf.add_to_collection('input', self.input_placeholder)
        tf.add_to_collection('latent_rep', self.z_mu)
        tf.add_to_collection('keep_prob', self.keep_prob)

        self.saver = tf.train.Saver()

        # Define one op to call all summaries
        self.merged = tf.summary.merge_all()
        # and one op to initialize the variables
        self.init_op = tf.global_variables_initializer()
