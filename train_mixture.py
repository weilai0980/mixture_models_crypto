#!/usr/bin/python

# local
from utils_libs import *
from utils_data_prep import *
from mixture import *

import sys
import os

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib import rnn

import math
import random


#print 'Number of arguments:', len(sys.argv), 'arguments.'
print '--- Argument List:', str(sys.argv)
method = str(sys.argv[1]) 
run_mode = str(sys.argv[2])
#loss_type = str(sys.argv[2])


# ---- Load pre-processed training and testing data ----
# norm_v_minu_mix, for rnn mixture: neu_norm_v_minu_mix


if run_mode == 'gpu':
    file_postfix = "neu_norm_v_minu_mix"
    xtrain = np.load("../dataset/bk/xtrain_"+file_postfix+".dat")
    xtest  = np.load("../dataset/bk/xtest_" +file_postfix+".dat")
    ytrain = np.load("../dataset/bk/ytrain_"+file_postfix+".dat")
    ytest  = np.load("../dataset/bk/ytest_" +file_postfix+".dat")

elif run_mode == 'local':
    file_postfix = "v_minu_mix"
    xtrain = np.load("../dataset/bitcoin/training_data/xtrain_"+file_postfix+".dat")
    xtest  = np.load("../dataset/bitcoin/training_data/xtest_" +file_postfix+".dat")
    ytrain = np.load("../dataset/bitcoin/training_data/ytrain_"+file_postfix+".dat")
    ytest  = np.load("../dataset/bitcoin/training_data/ytest_" +file_postfix+".dat")


print np.shape(xtrain), np.shape(ytrain), np.shape(xtest), np.shape(ytest)

# split training and testing data into three feature groups      
xtr_v =   np.asarray( [i[0] for i in xtrain] )
xtr_distr = np.asarray( [i[1] for i in xtrain] )

xts_v =   np.asarray( [i[0] for i in xtest] )
xts_distr = np.asarray( [i[1] for i in xtest] )

# !! IMPORTANT: feature normalization

xts_v = conti_normalization_test_dta(  xts_v, xtr_v )
xtr_v = conti_normalization_train_dta( xtr_v )

xts_distr = conti_normalization_test_dta(  xts_distr, xtr_distr )
xtr_distr = conti_normalization_train_dta( xtr_distr )

print np.shape(xtr_v), np.shape(xtr_distr)
print np.shape(xts_v), np.shape(xts_distr)


# ---- common parameters ----

# txt file to record errors in training process 
training_log  = "../bt_results/res/mix_train_log.txt"
res_file    = "../bt_results/res/mix.txt"
model_file = '../bt_results/model/mix'

# initialize the log
with open(training_log, "w") as text_file:
    text_file.close()

bool_train = True

# fixed parameters
para_order_v = len(xtr_v[0])

# ---- Approach specific parameters ----

#-- Mixture linear

para_lr_linear = 0.005
para_n_epoch_linear = 300
para_batch_size_linear = 64
para_l2_linear = 0.001

para_y_log = False
para_pred_exp = False

if len(np.shape(xtr_distr)) > 2:
    para_bool_bilinear = True
    para_order_distr = len(xtr_distr[0][0])
    para_order_steps = len(xtr_distr[0])
    
    print '     !! Time-first order !! '
    
else:
    para_bool_bilinear = False
    para_order_distr = len(xtr_distr[0])
    para_order_steps = 0
    
    print '     !! Flattened features !! '
    
'''
# -- Mixture MLP 
# representability
para_lr_mlp = 0.001
para_n_hidden_list_mlp = [ [32, 16, 4], [32, 16, 4], [32, 4], [16, 8] ]
para_n_epoch_mlp = 400
para_batch_size_mlp = 64

# regularization
para_l2_mlp = 0.005
para_keep_prob_mlp = 1.0

# -- Mixture LSTM 

#para_loss_type = 'norm'
#'lognorm', 'sq'

# representability
para_lstm_dims = [ [8, 4], [32, 16] ]
para_dense_dims = [ [32, 16], [32, 16, 8], [32, 16] ]

para_lr_lstm = 0.002
para_n_epoch_lstm = 300

# regularization 
para_l2_lstm = 0.001
para_batch_size_lstm = 64
para_keep_prob_lstm = 1.0

para_model_check = 10

'''

# ---- main process ----

tmp_test_err = []

if bool_train == True:
    
    with tf.Session() as sess:
        
        if method == 'linear_lk':
            clf = mixture_linear_lk(sess, para_lr_linear, para_l2_linear, para_batch_size_linear, para_order_v, \
                                  para_order_distr, para_order_steps, para_y_log, para_bool_bilinear)
            para_n_epoch = para_n_epoch_linear
            para_batch_size = para_batch_size_linear
            para_keep_prob = 0.0
            
            model_file += '_linear_lk.ckpt'
            
        elif method == 'linear_log':
            clf = mixture_linear_lognorm_lk(sess, para_lr_linear, para_l2_linear, para_batch_size_linear, para_order_v, \
                                 para_order_distr, para_order_steps, para_bool_bilinear)
            para_n_epoch = para_n_epoch_linear
            para_batch_size = para_batch_size_linear
            para_keep_prob = 0.0
            para_y_log = False
            
            model_file += '_linear_lognorm.ckpt'
            
        elif method == 'linear_sq':
            clf = mixture_linear_sq(sess, para_lr_linear, para_l2_linear, para_batch_size_linear, para_order_v, \
                                 para_order_distr, para_pred_exp, para_y_log)
            para_n_epoch = para_n_epoch_linear
            para_batch_size = para_batch_size_linear
            para_keep_prob = 0.0
            
            model_file += '_linear_sq.ckpt'
            
        elif method == 'mlp':
            clf = neural_mixture_dense(sess, para_n_hidden_list_mlp, para_lr_mlp, para_l2_mlp, para_batch_size_mlp,\
                                       para_order_v, para_order_distr )
            para_n_epoch = para_n_epoch_mlp
            para_batch_size = para_batch_size_mlp
            para_keep_prob = para_keep_prob_mlp
            
            model_file += '_linear_mlp.ckpt'
        
        elif method == 'lstm':
            
            model_file += '_lstm.ckpt'
            
            # reshape the data for lstm
            if len(np.shape(xtr_v))==2:
                xts_v = np.expand_dims( xts_v, 2 )
                xtr_v = np.expand_dims( xtr_v,  2 )
            
            print '---- Data shape for Mixture LSTM: '
            print np.shape(xtr_v), np.shape(xtr_distr)
            print np.shape(xts_v), np.shape(xts_distr)
            
            para_n_epoch    = para_n_epoch_lstm
            para_batch_size = para_batch_size_lstm
            para_keep_prob = para_keep_prob_lstm
            
            # fixed parameters
            para_steps = [ len(xtr_v[0]), len(xtr_distr[0]) ]  
            para_dims =  [ 1, len(xtr_distr[0][0])]
            
            clf = neural_mixture_lstm(sess, para_dense_dims, para_lstm_dims, para_lr_lstm, para_l2_lstm, \
                                      para_batch_size_lstm, para_steps, para_dims, loss_type)
        else:
            print "[ERROR] Need to specify a model"
            
    
        # initialize the network                          
        clf.train_ini()
        
        total_cnt   = np.shape(xtrain)[0]
        total_batch = int(total_cnt/para_batch_size)
        total_idx   = range(total_cnt)
                
        #   begin training epochs
        for epoch in range(para_n_epoch):
            
            # shuffle traning instances each epoch
            np.random.shuffle(total_idx)
            
            #  Loop over all batches
            tmpc = 0.0
            for i in range(total_batch):
                batch_idx = total_idx[ i*para_batch_size: (i+1)*para_batch_size ] 
            
                batch_v    =  xtr_v[ batch_idx ]
                batch_distr=  xtr_distr[ batch_idx ]
                
                # log transformation on the target
                if para_y_log == True:
                    batch_y = log(ytrain[batch_idx]+1e-5)
                else:
                    batch_y = ytrain[batch_idx]
            
                tmpc += clf.train_batch( batch_v, batch_distr, batch_y, para_keep_prob )
            
            #?
            tmp_train_acc = clf.inference(xtr_v, xtr_distr, ytrain, para_keep_prob)
            tmp_test_acc  = clf.inference(xts_v, xts_distr, ytest,  para_keep_prob,) 
            
            # record for re-tratin the model 
            tmp_test_err.append( [epoch, sqrt(tmp_train_acc[0]), sqrt(tmp_test_acc[0])] )
            
            print "loss on epoch ", epoch, " : ", 1.0*tmpc/total_batch, sqrt(tmp_train_acc[0]), tmp_train_acc[1],\
            sqrt(tmp_test_acc[0]), tmp_test_acc[1] 
            
            
        print "Optimization Finished!"
        
        # save overall errors
        with open(res_file, "a") as text_file:
            text_file.write( "Mixture %s : %s  \n"%(method, str(min(tmp_test_err, key = lambda x:x[2])) )) 
         
        
        # training the model at the best parameter above
        best_epoch = min(tmp_test_err, key = lambda x:x[2])[0]
        
        print "Re-train the model at epoch ", best_epoch
        
        # reset the model
        clf.model_reset()
        
        # training the model until the best epoch
        for epoch in range(best_epoch):
            
            # shuffle traning instances each epoch
            np.random.shuffle(total_idx)
            
            #  Loop over all batches
            for i in range(total_batch):
                batch_idx = total_idx[ i*para_batch_size: (i+1)*para_batch_size ] 
            
                batch_v    =  xtr_v[ batch_idx ]
                batch_distr=  xtr_distr[ batch_idx ]
                
                if para_y_log == True:
                    batch_y = log(ytrain[batch_idx]+1e-5)
                else:
                    batch_y = ytrain[batch_idx]
            
                _ = clf.train_batch( batch_v, batch_distr, batch_y, para_keep_prob )
            
            print "epoch: ", epoch
                
        print "Model Re-training Finished!"
        
        #?
        py = clf.predict(xts_v, xts_distr, para_keep_prob) 
        np.savetxt("../bt_results/res/pytest_mix.txt",  zip(py, ytest), delimiter=',')
        
        py = clf.predict(xtr_v, xtr_distr, para_keep_prob) 
        np.savetxt("../bt_results/res/pytrain_mix.txt", zip(py, ytrain), delimiter=',')
        
        gates_hat = clf.predict_gates(xts_v, xts_distr, para_keep_prob)
        np.savetxt("../bt_results/res/gate_test.txt", gates_hat, delimiter=',')
        
        gates = clf.predict_gates(xtr_v, xtr_distr, para_keep_prob)
        np.savetxt("../bt_results/res/gate_train.txt", gates, delimiter=',')
        
        # collect the values of all optimized parameters
        print 'prediction \n', clf.collect_coeff_values("pre")
        print 'variance \n', clf.collect_coeff_values("sig")
        print 'gate \n', clf.collect_coeff_values("gate")
        
else:
    
    # --- train the network under the best parameter set-up ---
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        clf = neural_mixture_dense( sess, para_n_hidden_list, para_lr, para_l2, para_batch_size,\
                                   para_order_v, para_order_distr )
    
        # initialize the network
        clf.train_ini()
        
        total_cnt   = np.shape(xtrain)[0]
        total_batch = int(total_cnt/para_batch_size)
        total_idx   = range(total_cnt)
        
        # begin training cycles
        for epoch in range(para_n_epoch):
            
            #  shuffle traning instances each epoch
            np.random.shuffle(total_idx)
            
            #  Loop over all batches
            for i in range(total_batch):
                batch_idx = total_idx[ i*para_batch_size: (i+1)*para_batch_size ] 
                
                batch_v    =  xtr_v[ batch_idx ]
                batch_distr=  xtr_distr[ batch_idx ]
                batch_y = ytrain[ batch_idx ]
            
                clf.train_batch( batch_v, batch_distr, batch_y, para_keep_prob )
        
        # save the model
        save_path = saver.save(sess, model_file)
        print("Model saved in file: %s" %save_path)
