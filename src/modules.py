import os
import time
import json

import numpy as np
import tensorflow as tf
lstm = tf.contrib.rnn.LSTMCell
fc = tf.contrib.layers.fully_connected

from IPython import embed

from config import TrainConfig
from data_util import read_sequential_target


def encoder(x, length, n_units=10, n_layers=1, scope="encoder"):
    with tf.variable_scope(scope, reuse=False):
        enc_cells = []
        enc_states = []
        for j in range(n_layers):
            with tf.variable_scope("layer_{}".format(j), reuse=False):
                enc_cell = lstm(num_units=n_units,
                                use_peepholes=True,
                                forget_bias=0.8)
                enc_cells.append(enc_cell)
        enc_cell = tf.contrib.rnn.MultiRNNCell(enc_cells)
        _, enc_states = tf.nn.dynamic_rnn(enc_cell,
                                          x,
                                          sequence_length=length,
                                          dtype=tf.float32,
                                          time_major=True)
        enc_states = tf.transpose(enc_states, (2,0,1,3)) # batchsize, n_layers, 2, n_units
        enc_states = tf.reshape(enc_states, (int(enc_states.shape[0]), -1))
    return enc_states

def L_decoder(init_in, init_state, length, out_dim,
              n_units=10, n_layers=1, scope="L_decoder"):
    y = []
    init_state = tf.reshape(init_state,
                            (int(init_state.shape[0]), n_layers, 2, n_units))
    init_state = tf.transpose(init_state, (1,2,0,3))

    with tf.variable_scope(scope, reuse=False):
        dec_cells = []
        for i in range(length-1):
            layer_in = init_in
            enc_states = []
            if i == 0:
                fc_reuse = False
                for j in range(n_layers):
                    with tf.variable_scope("layer_{}".format(j), reuse=False):
                        enc_state = (init_state[j][0], init_state[j][1])
                        dec_cell = lstm(num_units=n_units,
                                        use_peepholes=True,
                                        forget_bias=0.8)
                        dec_cells.append(dec_cell)
                        h, enc_state = dec_cell(layer_in, enc_state)
                        enc_states.append(enc_state)
                        layer_in = h
            else:
                layer_in = out
                tf.get_variable_scope().reuse_variables()
                fc_reuse = True
                for j in range(n_layers):
                    with tf.variable_scope("layer_{}".format(j), reuse=False):
                        dec_cell = dec_cells[j]
                        enc_state = prev_enc_states[j]
                        h, enc_state = dec_cell(layer_in, enc_state)
                        enc_states.append(enc_state)
                        layer_in = h
            prev_enc_states = enc_states
            out = fc(layer_in, out_dim,
                     activation_fn=tf.nn.softmax,
                     reuse=fc_reuse,
                     scope="L_decoder_fc")
            y.append(out)
    y = tf.stack(y)
    return y

def VB_decoder(V_in, init_B_in, init_state, length, out_dim,
               n_units=100, n_layers=1, scope="VB_decoder"):
    y = []
    init_state = tf.reshape(init_state,
                            (int(init_state.shape[0]), n_layers, 2, n_units))
    init_state = tf.transpose(init_state, (1,2,0,3))
    
    with tf.variable_scope(scope, reuse=False):
        dec_cells = []
        for i in range(length-1):
            current_V_in = V_in[i]
            enc_states = []
            if i == 0:
                current_B_in = init_B_in
                layer_in = tf.concat([current_V_in, current_B_in], axis=1)
                fc_reuse = False
                for j in range(n_layers):
                    with tf.variable_scope("layer_{}".format(j), reuse=False):
                        enc_state = (init_state[j][0], init_state[j][1])
                        dec_cell = lstm(num_units=n_units,
                                        use_peepholes=True,
                                        forget_bias=0.8)
                        dec_cells.append(dec_cell)
                        h, enc_state = dec_cell(layer_in, enc_state)
                        enc_states.append(enc_state)
                        layer_in = h
            else:
                current_B_in = out
                layer_in = tf.concat([current_V_in, current_B_in], axis=1)
                tf.get_variable_scope().reuse_variables()
                fc_reuse = True
                for j in range(n_layers):
                    with tf.variable_scope("layer_{}".format(j), reuse=False):
                        dec_cell = dec_cells[j]
                        enc_state = prev_enc_states[j]
                        h, enc_state = dec_cell(layer_in, enc_state)
                        enc_states.append(enc_state)
                        layer_in = h
            prev_enc_states = enc_states

            out = fc(layer_in, out_dim,
                     activation_fn=tf.tanh,
                     reuse=fc_reuse,
                     scope="VB_decoder_fc")
            y.append(out)
    y = tf.stack(y)
    return y

def make_placeholders(L_shape, B_shape, V_shape, batchsize):
    place_holders = {}
    place_holders["L_fw"] = tf.placeholder(tf.float32, [L_shape[0], batchsize, L_shape[2]], name="L_fw") 
    place_holders["B_fw"] = tf.placeholder(tf.float32, [B_shape[0], batchsize, B_shape[2]], name="B_fw")
    place_holders["V_fw"] = tf.placeholder(tf.float32, [V_shape[0], batchsize, V_shape[2]], name="V_fw")
    place_holders["L_bw"] = tf.placeholder(tf.float32, [L_shape[0], batchsize, L_shape[2]], name="L_bw") 
    place_holders["B_bw"] = tf.placeholder(tf.float32, [B_shape[0], batchsize, B_shape[2]], name="B_bw")
    place_holders["V_bw"] = tf.placeholder(tf.float32, [V_shape[0], batchsize, V_shape[2]], name="V_bw")
    #place_holders["L_bin"] = tf.placeholder(tf.float32, [L_shape[0], batchsize, L_shape[2]], name="L_bin") 
    place_holders["B_bin"] = tf.placeholder(tf.float32, [B_shape[0], batchsize, B_shape[2]], name="B_bin")
    #place_holders["V_bin"] = tf.placeholder(tf.float32, [V_shape[0], batchsize, V_shape[2]], name="V_bin")
    place_holders["L_len"] = tf.placeholder(tf.float32, [batchsize], name="L_len")
    place_holders["B_len"] = tf.placeholder(tf.float32, [batchsize], name="B_len")
    place_holders["V_len"] = tf.placeholder(tf.float32, [batchsize], name="V_len")
    return place_holders

def aligned_discriminative_loss(X, Y, margin=1.0):
    batchsize = int(X.shape[0])
    X_tile = tf.tile(X, (batchsize, 1))
    Y_tile = tf.reshape(tf.tile(Y, (1, batchsize)),
                        (batchsize**2, -1))

    pair_loss = tf.sqrt(tf.reduce_sum(tf.square(X-Y), axis=1))
    all_pairs = tf.square(X_tile-Y_tile)
    loss_array = tf.reshape(tf.sqrt(tf.reduce_sum(all_pairs, axis=1)),
                            (batchsize, batchsize))

    x_diff = tf.expand_dims(pair_loss, axis=0) - loss_array + margin
    y_diff = tf.expand_dims(pair_loss, axis=1) - loss_array + margin
    x_diff = tf.maximum(x_diff, 0)
    y_diff = tf.maximum(y_diff, 0)
    mask = 1.0 - tf.eye(batchsize)
    x_diff = x_diff * mask
    y_diff = y_diff * mask
    
    return tf.reduce_mean(x_diff) + tf.reduce_mean(y_diff) + tf.reduce_mean(pair_loss)
