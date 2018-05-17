import os
import time
import json

import numpy as np
import tensorflow as tf
fc = tf.contrib.layers.fully_connected
from IPython import embed

from config import NetConfig, TrainConfig
from data_util import read_sequential_target
from task_util import *
from modules import *

def main():
    net_conf = NetConfig()
    net_conf.set_conf("./net_conf.txt")

    L_num_units = net_conf.L_num_units
    L_num_layers = net_conf.L_num_layers
    VB_num_units = net_conf.VB_num_units
    VB_num_layers = net_conf.VB_num_layers
    SHARE_DIM = net_conf.S_dim
    
    train_conf = TrainConfig()
    train_conf.set_conf("./train_conf.txt")
    
    seed = train_conf.seed
    batchsize = train_conf.batchsize
    epoch = train_conf.epoch
    save_dir = train_conf.save_dir
    if not os.path.exists(os.path.dirname(save_dir)):
        os.mkdir(os.path.dirname(save_dir))
    
    L_data_dir = train_conf.L_dir
    B_data_dir = train_conf.B_dir
    V_data_dir = train_conf.V_dir
    L_fw, L_bw, L_bin, L_len, filenames = read_sequential_target(L_data_dir, True)
    print len(filenames)
    B_fw, B_bw, B_bin, B_len = read_sequential_target(B_data_dir)
    V_fw, V_bw, V_bin, V_len = read_sequential_target(V_data_dir)
    L_shape = L_fw.shape
    B_shape = B_fw.shape
    V_shape = V_fw.shape

    if train_conf.test:
        L_data_dir_test = train_conf.L_dir_test
        B_data_dir_test = train_conf.B_dir_test
        V_data_dir_test = train_conf.V_dir_test
        L_fw_u, L_bw_u, L_bin_u, L_len_u, filenames_u = read_sequential_target(L_data_dir_test, True)
        print len(filenames_u)
        B_fw_u, B_bw_u, B_bin_u, B_len_u = read_sequential_target(B_data_dir_test)
        V_fw_u, V_bw_u, V_bin_u, V_len_u = read_sequential_target(V_data_dir_test)
        L_shape_u = L_fw_u.shape
        B_shape_u = B_fw_u.shape
        V_shape_u = V_fw_u.shape
        
    np.random.seed(seed)

    tf.reset_default_graph()
    tf.set_random_seed(seed)

    placeholders = make_placeholders(L_shape, B_shape, V_shape, batchsize)

    ##### encoding #####
    L_enc_final_state = encoder(placeholders["L_bw"],
                                placeholders["L_len"],
                                n_units=L_num_units,
                                n_layers=L_num_layers,
                                scope="L_encoder")

    VB_input = tf.concat([placeholders["V_bw"],
                          placeholders["B_bw"]],
                         axis=2)
    VB_enc_final_state = encoder(VB_input,
                                 placeholders["V_len"], 
                                 n_units=VB_num_units,
                                 n_layers=VB_num_layers,
                                 scope="VB_encoder")

    ##### sharing #####
    L_shared = fc(L_enc_final_state, SHARE_DIM,
                  activation_fn=None, scope="L_share")
    VB_shared = fc(VB_enc_final_state, SHARE_DIM,
                   activation_fn=None, scope="VB_share")  
    L_dec_init_state = fc(L_shared, L_num_units*L_num_layers*2,
                          activation_fn=None, scope="L_postshare")
    VB_dec_init_state = fc(VB_shared, VB_num_units*VB_num_layers*2,
                           activation_fn=None, scope="VB_postshare")

    ##### decoding #####
    L_output = L_decoder(placeholders["L_fw"][0],
                         L_dec_init_state,
                         length=L_shape[0],
                         out_dim=L_shape[2],
                         n_units=L_num_units,
                         n_layers=L_num_layers,
                         scope="L_decoder")
    B_output = VB_decoder(placeholders["V_fw"],
                          placeholders["B_fw"][0],
                          VB_dec_init_state,
                          length=B_shape[0],
                          out_dim=B_shape[2],
                          n_units=VB_num_units,
                          n_layers=VB_num_layers,
                          scope="VB_decoder")
    
    with tf.name_scope('loss'):
        L_output = L_output # no need to multiply binary array
        B_output = B_output * placeholders["B_bin"][1:]
        L_loss = tf.reduce_mean(-tf.reduce_sum(placeholders["L_fw"][1:]*tf.log(L_output),
                                               reduction_indices=[2])) 
        B_loss = tf.reduce_mean(tf.square(B_output-placeholders["B_fw"][1:]))
        share_loss = aligned_discriminative_loss(L_shared, VB_shared)
        loss = net_conf.L_weight*L_loss + net_conf.B_weight*B_loss + net_conf.S_weight*share_loss
        
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    gpuConfig = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True,
                                  per_process_gpu_memory_fraction=train_conf.gpu_use_rate),
        device_count={'GPU': 1})
        
    sess = tf.Session(config=gpuConfig)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())

    previous = time.time()
    for step in range(epoch):
        batch_idx = np.random.permutation(B_shape[1])[:batchsize]
        feed_dict = {placeholders["L_fw"]: L_fw[:, batch_idx, :],
                     placeholders["B_fw"]: B_fw[:, batch_idx, :],
                     placeholders["V_fw"]: V_fw[:, batch_idx, :],
                     placeholders["L_bw"]: L_bw[:, batch_idx, :],
                     placeholders["B_bw"]: B_bw[:, batch_idx, :],
                     placeholders["V_bw"]: V_bw[:, batch_idx, :],
                     placeholders["B_bin"]: B_bin[:, batch_idx, :],
                     placeholders["L_len"]: L_len[batch_idx],
                     placeholders["V_len"]: V_len[batch_idx]}
                    
        _, l, b, s, t = sess.run([train_step,
                                  L_loss,
                                  B_loss,
                                  share_loss,
                                  loss],
                                 feed_dict=feed_dict)
        print "step:{} total:{}, language:{}, behavior:{}, share:{}".format(step, t, l, b, s)
        if train_conf.test and (step+1) % train_conf.test_interval == 0:
            batch_idx = np.random.permutation(B_shape_u[1])[:batchsize]
            feed_dict = {placeholders["L_fw"]: L_fw_u[:, batch_idx, :],
                         placeholders["B_fw"]: B_fw_u[:, batch_idx, :],
                         placeholders["V_fw"]: V_fw_u[:, batch_idx, :],
                         placeholders["L_bw"]: L_bw_u[:, batch_idx, :],
                         placeholders["B_bw"]: B_bw_u[:, batch_idx, :],
                         placeholders["V_bw"]: V_bw_u[:, batch_idx, :],
                         placeholders["B_bin"]: B_bin_u[:, batch_idx, :],
                         placeholders["L_len"]: L_len_u[batch_idx],
                         placeholders["V_len"]: V_len_u[batch_idx]}
            
            l, b, s, t = sess.run([L_loss, B_loss, share_loss, loss],
                                  feed_dict=feed_dict)
            print "test"
            print "step:{} total:{}, language:{}, behavior:{}, share:{}".format(step, t, l, b, s)
            
        if (step + 1) % train_conf.log_interval == 0:
            saver.save(sess, save_dir)
    past = time.time()
    print past-previous
if __name__ == "__main__":
    main()
