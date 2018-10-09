#!/usr/bin/python

import sys
import os

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib import rnn

import math
import random
from random import shuffle

# local packages 
from utils_libs import *
from utils_data_prep import *
from mixture_neural import *

# ---- parameters from command line

print('--- Argument List:', str(sys.argv))
method = str(sys.argv[1])
train_mode = str(sys.argv[2])
# oneshot, roll, incre
# gpu_id = str(sys.argv[3])

# ---- parameter set-up from parameter-file ----

para_dict = para_parser("para_file.txt")

para_order_minu = para_dict['para_order_minu']
para_order_hour = para_dict['para_order_hour']
bool_feature_selection = para_dict['bool_feature_selection']

interval_len = para_dict['interval_len']
roll_len = para_dict['roll_len']

para_step_ahead = para_dict['para_step_ahead']

# ---- Approach specific parameters ----

# -- LSTM concate 

para_n_epoch = 100

# regularization 
para_batch_size = 64
# para_max_norm = 4.0

# complexity
para_lstm_sizes = [ [32], [64] ]
para_y_log = False

para_loss_type = 'gaussian'
para_activation_type = 'linear'
# linear, relu, leaky_relu, tanh 
para_pos_regu = True

para_x_pointwise = True
para_gate_type = 'logistic-concat'
# ada-ma, softmax-linear,  softmax-mlp, logistic-concat, logistic-partial 
# if logisitics, nothing to do with para_gate_logit_shared
para_gate_logit_shared = True

para_rnn_type = 'gru'
para_decay_step = 500

# ---- training and evalution methods ----
    
def train_validate_mixture( xtr_auto, xtr_x, ytrain, xval_auto, xval_x, yval ):   
    
    # stabilize the network by fixing the random seed
    np.random.seed(1)
    tf.set_random_seed(1)
    # reshape the data for lstm
    if len(np.shape(xtr_auto))==2:
        xval_auto = np.expand_dims( xval_auto, 2 )
        xtr_auto = np.expand_dims( xtr_auto, 2 )
            
    # fixed parameters
    para_steps = [ para_order_hour, para_order_minu ]  
    para_dims =  [ 1, np.shape(xtr_x)[-1] ]
    
    with tf.device('/device:GPU:5'):
        
        config = tf.ConfigProto(allow_soft_placement = True)
        sess = tf.Session(config = config)
        
        #sess = tf.Session()    
    #with tf.Session() as sess:
        
        if method == 'lstm-concat':
            
            clf = lstm_concat(sess, para_lr, para_l2, \
                              para_steps[0], para_dims[1], para_steps[1], \
                              para_dense_num, para_max_norm, para_lstm_sizes)
            
        elif method == 'lstm-mixture':
            
            clf = lstm_mixture(sess, para_lr, para_l2, \
                               para_steps[0], para_dims[1], para_steps[1], \
                               para_dense_num, para_max_norm, para_lstm_sizes, \
                               para_loss_type, para_activation_type, para_pos_regu, \
                               para_gate_type, para_gate_logit_shared, para_x_pointwise, para_rnn_type)
            
        else:
            print("\n --- [ERROR] Need to specify a model --- \n")
            
        
        # initialize the network
        clf.train_ini()
        
        # set up training batch parameters
        total_cnt   = np.shape(xtr_auto)[0]
        total_batch = int(total_cnt/para_batch_size)
        total_idx   = list(range(total_cnt))
        
        # begin training epochs
        
        # epoch-wise errors
        tmp_epoch_err = []
        
        for epoch in range(para_n_epoch):
            
            # shuffle traning instances each epoch
            np.random.shuffle(total_idx)
            
            # loop over all batches
            tmpc = 0.0
            for i in range(total_batch):
                
                batch_idx = total_idx[ i*para_batch_size: (i+1)*para_batch_size ] 
            
                batch_auto = xtr_auto[ batch_idx ]
                batch_x = xtr_x[ batch_idx ]
                
                # log transformation on the target
                if para_y_log == True:
                    batch_y = log(ytrain[batch_idx]+1e-5)
                else:
                    batch_y = ytrain[batch_idx]
                
                # learning rate decay
                if (i + total_batch*epoch)%para_decay_step == 0:
                    
                    tmpc += clf.train_batch(batch_auto, batch_x, batch_y, para_keep_prob, \
                                            True, para_lr*(0.96)**((i + total_batch*epoch)/para_decay_step) )
                else:
                    
                    tmpc += clf.train_batch(batch_auto, batch_x, batch_y, para_keep_prob, \
                                            False, 0.0 )
            #?
            tmp_train_rmse, tmp_train_mae, tmp_train_mape = clf.inference(xtr_auto, xtr_x, ytrain, [1.0, 1.0])
            #?
            tmp_test_rmse, tmp_test_mae, tmp_test_mape = clf.inference(xval_auto, xval_x, yval, [1.0, 1.0]) 
            
            # record for fixing the parameter set-up in testing
            tmp_epoch_err.append( [epoch, tmp_train_rmse, tmp_test_rmse] )
            
            print("loss on epoch ", epoch, " : ", 1.0*tmpc/total_batch, tmp_train_rmse, tmp_test_rmse)
            
        print("Optimization Finished!")
        
        # reset the model
        clf.model_reset()
        clf.train_ini()
    
    # clear the graph in the current session
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    return min(tmp_epoch_err, key = lambda x:x[-1])
    
    
# retrain the model with the best parameter set-up        
def test_mixture( xtr_auto, xtr_x, ytrain, xts_auto, xts_x, ytest, file_addr, model_file, best_epoch ): 
    
    # stabilize the network by fixing the random seed
    # fixing the seed will settle down all the random parameters over the entire routine below
    np.random.seed(1)
    tf.set_random_seed(1)
    
    # reshape the data for lstm
    if len(np.shape(xtr_auto))==2:
        xts_auto = np.expand_dims( xts_auto, 2 )
        xtr_auto = np.expand_dims( xtr_auto,  2 )
    
    # fixed parameters
    para_steps = [ para_order_hour, para_order_minu ]  
    para_dims =  [ 1, np.shape(xtr_x)[-1] ]
    
    '''
    # fixed parameters
    para_steps = [ len(xtr_auto[0]), len(xtr_x[0]) ]  
    para_dims =  [ 1, len(xtr_x[0][0])]
    '''
    
    print("Re-train the model at epoch ", best_epoch)
    
    with tf.device('/device:GPU:5'):
        
        
    
    #with tf.Session() as sess:
        
        sess = tf.Session()
        #config=tf.ConfigProto(log_device_placement=True)
        
        print('---- parameter and epoch set-up in testing:', para_dense_num, para_keep_prob, para_l2, best_epoch)
        
        if method == 'lstm-concat':
            
            clf = lstm_concat(sess, para_lr, para_l2, para_steps[0], \
                              para_dims[1], para_steps[1], para_dense_num, para_max_norm, para_lstm_sizes )
            
        elif method == 'lstm-mixture':
            
            clf = lstm_mixture(sess, para_lr, para_l2, \
                               para_steps[0], para_dims[1], para_steps[1], \
                               para_dense_num, para_max_norm, para_lstm_sizes, \
                               para_loss_type, para_activation_type, para_pos_regu, \
                               para_gate_type, para_gate_logit_shared, para_x_pointwise, para_rnn_type)
            
        else:
            print("     [ERROR] Need to specify a model")
            
        
        # initialize the network
        clf.train_ini()
        
        # setup mini-batch parameters
        total_cnt   = np.shape(xtr_auto)[0]
        total_batch = int(total_cnt/para_batch_size)
        total_idx   = list(range(total_cnt))
        
        # epoch-wise error record
        tmp_epoch_err = []
        
        # training the model until the best epoch
        for epoch in range(best_epoch+1):
            
            
            # shuffle traning instances each epoch
            np.random.shuffle(total_idx)
            
            # Loop over all batches
            tmpc = 0.0
            for i in range(total_batch):
                
                batch_idx = total_idx[ i*para_batch_size: (i+1)*para_batch_size ] 
            
                batch_auto =  xtr_auto[ batch_idx ]
                batch_x =  xtr_x[ batch_idx ]
                
                if para_y_log == True:
                    batch_y = log(ytrain[batch_idx]+1e-5)
                else:
                    batch_y = ytrain[batch_idx]
                    
                    
                # learning rate decay
                if (i + total_batch*epoch)%para_decay_step == 0:
                    
                    tmpc += clf.train_batch(batch_auto, batch_x, batch_y, para_keep_prob,\
                                            True, para_lr*(0.96)**((i + total_batch*epoch)/para_decay_step))
                else:
                    
                    tmpc += clf.train_batch(batch_auto, batch_x, batch_y, para_keep_prob,\
                                            False, 0.0)
            
            #?
            tmp_train_rmse, tmp_train_mae, tmp_train_mape = clf.inference(xtr_auto, xtr_x, ytrain, [1.0, 1.0])
            #?
            tmp_test_rmse, tmp_test_mae, tmp_test_mape = clf.inference(xts_auto, xts_x, ytest, [1.0, 1.0]) 
            
            # record for fixing the parameter set-up in testing
            tmp_epoch_err.append( [epoch, tmp_train_rmse, tmp_test_rmse] )
            
            print("loss on epoch ", epoch, " : ", 1.0*tmpc/total_batch, tmp_train_rmse, tmp_test_rmse)
            
        print("Model Re-training Finished!")
        
        '''
        if method == 'lstm-mixture':
            
            # record prediction and mixture gate values 
            py_test = clf.prlogisticedict(xts_auto, xts_x, para_keep_prob)
            tmp = np.concatenate( [np.expand_dims(ytest, -1), np.transpose(py_test, [1, 0])], 1 )
            np.savetxt( file_addr + "pytest_neumix.txt", tmp, delimiter=',')
        
            py_train = clf.predict(xtr_auto, xtr_x, para_keep_prob)
            tmp = np.concatenate( [np.expand_dims(ytrain, -1), np.transpose(py_train, [1, 0])], 1 )
            np.savetxt( file_addr + "pytrain_neumix.txt", tmp, delimiter=',')
        
            gates_test = clf.predict_gates(xts_auto, xts_x, para_keep_prob)
            np.savetxt( file_addr + "gate_test_neu.txt", gates_test, delimiter=',')
        
            gates_train = clf.predict_gates(xtr_auto, xtr_x, para_keep_prob)
            np.savetxt( file_addr + "gate_train_neu.txt", gates_train, delimiter=',')
        
        
            # collect the values of all optimized parameters
            if train_mode == 'oneshot':
                print 'prediction \n', clf.collect_coeff_values("mean")
                print 'variance \n', clf.collect_coeff_values("sig")
                print 'gate \n', clf.collect_coeff_values("gate")
            
            else:
                w1 = clf.collect_coeff_values("mean")
                np.asarray(w1[1]).dump( file_addr + "weight_pre_mix.dat" )
            
                w2 = clf.collect_coeff_values("gate")
                np.asarray(w2[1]).dump( file_addr + "weight_gate_mix.dat" )
        '''
        
        # reset the model 
        clf.model_reset()
        
    # clear the graph in the current session    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    
    return [ [tmp_test_rmse, tmp_test_mae, tmp_test_mape], min(tmp_epoch_err, key = lambda x:x[-1]) ] 
        
        
def preprocess_feature_mixture(xtrain, xtest):
    
    # split training and testing data into three feature groups      
    xtr_auto = np.asarray( [j[0] for j in xtrain] )
    xtr_x = np.asarray( [j[1] for j in xtrain] )

    xts_auto = np.asarray( [j[0] for j in xtest] )
    xts_x = np.asarray( [j[1] for j in xtest] )
    
    # !! IMPORTANT: feature normalization

    norm_xts_auto = conti_normalization_test_dta(  xts_auto, xtr_auto )
    norm_xtr_auto = conti_normalization_train_dta( xtr_auto )

    norm_xts_x = conti_normalization_test_dta(  xts_x, xtr_x )
    norm_xtr_x = conti_normalization_train_dta( xtr_x )
    
    return np.asarray(norm_xtr_auto), np.asarray(norm_xtr_x), np.asarray(norm_xts_auto), np.asarray(norm_xts_x)
    
    
# ---- main process ----  

if train_mode == 'oneshot':
    
    # result log
    res_file    = "../bt_results/res/mix.txt"
    model_file = '../bt_results/model/mix'
    pred_file = "../bt_results/res/"
    
    # load pre-processed training and testing data
    file_postfix = "v_minu_mix"
    xtrain = np.load("../dataset/bitcoin/training_data/xtrain_"+file_postfix+".dat")
    xtest  = np.load("../dataset/bitcoin/training_data/xtest_" +file_postfix+".dat")
    ytrain = np.load("../dataset/bitcoin/training_data/ytrain_"+file_postfix+".dat")
    ytest  = np.load("../dataset/bitcoin/training_data/ytest_" +file_postfix+".dat")
    print(np.shape(xtrain), np.shape(ytrain), np.shape(xtest), np.shape(ytest))
    
    # extract different groups of features
    xtr, xtr_exter, xts, xts_exter = preprocess_feature_mixture(xtrain, xtest)
    
    print('--- Start one-shot training: ')
    print(np.shape(xtr), np.shape(xtr_exter))
    print(np.shape(xts), np.shape(xts_exter))
    
    tmp_error = train_eval_mixture( xtr, xtr_exter, ytrain, xts, xts_exter, ytest, pred_file, model_file ) 
    
    
elif train_mode == 'roll' or 'incre':
    
    # fix random seed
    np.random.seed(1)
    tf.set_random_seed(1)
    
    # ---- prepare the log file
    
    res_file   = "../bt_results/res/rolling/reg_" + method + ".txt"
    model_file = "../bt_results/model/mix_"
    pred_file = ""
    
    with open(res_file, "a") as text_file:
            text_file.write("\n %s %s %s %s %s %s %s %s %s %s \n\n"%( \
       "\n -------- Rolling --------- \n" if train_mode == 'roll' else "\n -------- Incremental --------- \n",\
                                                            method,\
                                                            str(para_lstm_sizes),\
                                                            str(para_loss_type),\
                                                            str(para_activation_type),\
                                        ' pos_regu ' if para_pos_regu == True else ' NO_pos_regu ',\
                                                            str(para_gate_type),\
                          ' point_wise ' if para_x_pointwise == True else ' last_step ',\
                          ' logit_weight_shared ' if para_gate_logit_shared == True else ' logit_weight_NO_shared ',\
                                                            para_rnn_type))
                           
            
    # ---- prepare the data
    
    # load raw feature and target data
    features_minu = np.load("../dataset/bitcoin/training_data/feature_minu.dat", encoding='latin1')
    rvol_hour = np.load("../dataset/bitcoin/training_data/return_vol_hour.dat", encoding='latin1')
    all_loc_hour = np.load("../dataset/bitcoin/loc_hour.dat", encoding='latin1')
    print('--- Start the ' + train_mode + ' training: \n', np.shape(features_minu), np.shape(rvol_hour))
    
    # prepare the set of pairs of feature - target
    x, y, var_explain = prepare_feature_target(features_minu, rvol_hour, all_loc_hour, \
                                               para_order_minu, para_order_hour, bool_feature_selection, \
                                               para_step_ahead, para_x_pointwise)
    
    # set up the training and evaluation interval 
    interval_num = int(len(y)/interval_len)
    
    
    # ---- the main loop
    #
    for i in range(roll_len + 1, interval_num + 1):
        
        print('\n --- In processing of interval ', i-1, ' --- \n')
        
        # reset the graph
        tf.reset_default_graph()
        sess = tf.InteractiveSession()
        
        # log for predictions in each interval
        pred_file = "../bt_results/res/rolling/" + str(i-1) + "_" + str(para_step_ahead) + '_'
        
        # data within the current time interval
        if train_mode == 'roll':
            tmp_x = x[(i-roll_len-1)*interval_len : i*interval_len]
            tmp_y = y[(i-roll_len-1)*interval_len : i*interval_len]
            para_train_split_ratio = 1.0*(len(tmp_x) - interval_len)/len(tmp_x)
            
        elif train_mode == 'incre':
            tmp_x = x[ : i*interval_len]
            tmp_y = y[ : i*interval_len]
            para_train_split_ratio = 1.0*(len(tmp_x) - interval_len)/len(tmp_x)
            
        else:
            print('[ERROR] training mode')
        
        # training, validation+testing split
        xtrain, ytrain, xtest, ytest = training_testing_mixture_rnn(tmp_x, tmp_y, para_train_split_ratio)
        
        # feature split, normalization READY
        xtr_auto, xtr_x, xts_auto, xts_x = preprocess_feature_mixture(xtrain, xtest)
        
        # build validation and testing data 
        tmp_idx = range(len(xtest))
        tmp_val_idx = []
        tmp_ts_idx = []
        
        # validation and testing data: even sampling 
        for j in tmp_idx:
            if j%2 == 0:
                tmp_val_idx.append(j)
            else:
                tmp_ts_idx.append(j)
        
        xval_auto = xts_auto[tmp_val_idx]
        xval_x = xts_x[tmp_val_idx]
        yval = np.asarray(ytest)[tmp_val_idx]
        
        xts_auto = xts_auto[tmp_ts_idx]
        xts_x = xts_x[tmp_ts_idx]
        yts = np.asarray(ytest)[tmp_ts_idx]
        
        print('Shape of training, validation and testing data: \n')                            
        print(np.shape(xtr_auto), np.shape(xtr_x), np.shape(ytrain))
        print(np.shape(xval_auto), np.shape(xval_x), np.shape(yval))
        print(np.shape(xts_auto), np.shape(xts_x), np.shape(yts))
        
        # -- training and validation phase
        
        para_train_vali = []
        
        for para_lr in [ 0.0005 ]:
            for para_dense_num in [ 0, 1 ]:
                for para_keep_prob in [ [1.0], [0.8] ]:
                    for para_l2 in [ 0.001, 0.005, 0.01, 0.05, 0.1 ]:
                        
                        # apply max_norm contraint only when dropout is used
                        if para_keep_prob[0] <1.0:
                            para_max_norm = 4.0
                        else:
                            para_max_norm = 0.0
                            
                        print("Current parameter set-up: \n", para_lstm_sizes, para_lr, para_dense_num, \
                              para_keep_prob, para_l2, para_decay_step, para_rnn_type, "\n")    
                    
                        # return the parameter set-up and epoch with the lowest validation errors and llk 
                        tmp_best_epoch, tmp_train_err, tmp_vali_err = train_validate_mixture(xtr_auto, 
                                                                                             xtr_x,\
                                                                                             np.asarray(ytrain),\
                                                                                             xval_auto, 
                                                                                             xval_x,\
                                                                                             np.asarray(yval))
                        para_train_vali.append([para_lr,
                                                para_dense_num, 
                                                para_keep_prob, 
                                                para_l2, 
                                                tmp_best_epoch, \
                                                tmp_train_err, 
                                                tmp_vali_err])
                        
                        print("Current erros: \n", para_train_vali[-1], "\n")
                        
        # -- testing phase
        
        # fix the best hyper-parameter, epoch
        final_para = min(para_train_vali, key = lambda x : x[-1])
        
        para_lr = final_para[0]
        para_dense_num = final_para[1]
        para_keep_prob = final_para[2]
        para_l2 = final_para[3]
        best_epoch = final_para[4]
        
        print(' ---- Best parameters : ', final_para, '\n')
        
        test_err = test_mixture(xtr_auto, xtr_x, np.asarray(ytrain), xts_auto, xts_x, np.asarray(yts),\
                                pred_file, model_file, best_epoch)
        
        print(' ---- Training, validation and testing performance : ', final_para[5:], test_err, '\n')
        
        
        # -- log overall errors
        
        # result_tupe: [best epoch, training, validation, testing error]
        result_tupe = [final_para[5], final_para[6], test_err]
        
        with open(res_file, "a") as text_file:
            text_file.write( "Interval %d : %s, %s \n" %(i-1, str(final_para[:5]), str(result_tupe)) )
            
else:
    print('[ERROR] training mode')
