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

import json

# local packages 
from utils_libs import *
from utils_data_prep import *
from mixture import *

# ---- parameters from command line

'''
Arguments:

method: linear, bayes
para_loss_type: lk, sg
train_mode: oneshot, roll, incre
    
'''

print '--- Argument List:', str(sys.argv)
method = str(sys.argv[1])
para_loss_type = str(sys.argv[2])
train_mode = str(sys.argv[3])

# ---- parameter from config ----

#para_dict = para_parser("para_file.txt")

with open('config.json') as f:
    para_dict = json.load(f)
    print(para_dict) 

para_order_minu = para_dict['para_order_minu']
para_order_hour = para_dict['para_order_hour']
bool_feature_selection = False if para_dict['bool_feature_selection'] == 'False' else True

interval_len = para_dict['interval_len']
roll_len = para_dict['roll_len']

para_step_ahead = para_dict['para_step_ahead']


# ---- parameters specific to models ----

para_y_log = False
para_pred_exp = False

para_bool_bilinear = True

# -- Mixture linear
para_lr_linear = 0.001
para_batch_size_linear = 32

para_epoch_linear = 300
#para_l2_linear = 0.001

para_distr_type = 'gaussian'
para_activation_type = 'linear'
# relu, leaky_relu, linear
para_pos_regu = True
para_gate_type = 'softmax'

# -- Bayes
para_lr_bayes = 0.001
para_n_epoch_bayes = 300
para_batch_size_bayes = 32

para_eval_sample_num = 30
para_loss_sample_num = 10

para_infer_type = 'variational_nocondi'


# 'variational-nocondi', 'variational-condi', gibbs 
# ---- training and evalution methods ----
    
def train_validate_mixture( xtr_auto, xtr_x, ytrain, xts_auto, xts_x, ytest ):   
    
    # stabilize the network by fixing the random seed
    np.random.seed(1)
    tf.set_random_seed(1)
    
    with tf.Session() as sess:
        
        if method == 'linear':
            
            '''
            Parameters to tune:
              
              regularization coefficients
            
            '''
            
            clf = mixture_linear(sess, para_lr_linear, para_l2, para_batch_size_linear, para_order_auto, \
                                 para_order_x, para_order_steps, para_y_log, para_bool_bilinear,\
                                 para_loss_type, para_distr_type, para_activation_type, para_pos_regu, para_gate_type)
            
            # global para
            #para_n_epoch = para_n_epoch_linear
            para_batch_size = para_batch_size_linear
            para_keep_prob = 1.0
            
        elif method == 'bayes':
            
            '''
            Parameters to tune:
              
              para_eval_sample_num
              para_loss_sample_num
            
            '''
            
            para_num_iter = para_n_epoch_bayes * (len(xtr_auto)/para_batch_size_bayes)
            
            clf = variational_mixture_linear(sess, para_lr_bayes, para_batch_size_bayes, para_order_auto, para_order_x,\
                                             para_order_steps, para_y_log, para_bool_bilinear, para_distr_type,\
                                             para_eval_sample_num, para_infer_type, para_num_iter, para_loss_sample_num,\
                                             int(np.shape(xtrain)[0]/para_batch_size_bayes))
            # global para
            #para_n_epoch = para_n_epoch_bayes
            para_batch_size = para_batch_size_bayes
            para_keep_prob = 1.0
            
        else:
            print "     [ERROR] Need to specify a model"
            
        # initialize the network
        # reset the model
        clf.train_ini()
        clf.evaluate_ini()
        
        # set up training batch parameters
        total_cnt   = np.shape(xtrain)[0]
        total_batch = int(total_cnt/para_batch_size)
        total_idx   = range(total_cnt)
        
        # log training and validation errors over epoches
        tmp_err_epoch = []
        
        #  begin training epochs
        for epoch in range(para_epoch_linear):
            
            # shuffle traning instances each epoch
            np.random.shuffle(total_idx)
            
            # loop over all batches
            tmpc = 0.0
            for i in range(total_batch):
                
                batch_idx = total_idx[ i*para_batch_size: (i+1)*para_batch_size ] 
            
                batch_auto    =  xtr_auto[ batch_idx ]
                batch_x =  xtr_x[ batch_idx ]
                
                # log transformation on the target
                if para_y_log == True:
                    batch_y = log(ytrain[batch_idx]+1e-5)
                else:
                    batch_y = ytrain[batch_idx]
            
                tmpc = tmpc + float(clf.train_batch( batch_auto, batch_x, batch_y, para_keep_prob ))
            
            #?
            tmp_train_rmse, tmp_train_mae, tmp_train_mape, tmp_train_regu = \
            clf.inference(xtr_auto, xtr_x, ytrain, para_keep_prob)
            #?
            tmp_test_rmse, tmp_test_mae, tmp_test_mape, tmp_test_regu = \
            clf.inference(xts_auto, xts_x, ytest,  para_keep_prob) 
            
            # record for re-training the model afterwards
            tmp_err_epoch.append( [epoch, tmp_train_rmse, tmp_test_rmse] )
            
            # training rmse, training regularization, testing rmse, testing regularization
            print "loss on epoch ", epoch, " : ", 1.0*tmpc/total_batch, tmp_train_rmse, tmp_test_rmse
            
        print "Optimization Finished!"
        
        # reset the model
        clf.model_reset()
        #clf.train_ini()
    
    # clear the graph in the current session 
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    

    return min(tmp_err_epoch, key = lambda x:x[-1]), [tmp_test_rmse, tmp_test_mae, tmp_test_mape]
    
'''    
# retrain the model with the best parameter set-up        
def test_mixture( xtr_auto, xtr_x, ytrain, xts_auto, xts_x, ytest, file_addr, model_file, best_epoch ): 
    
    # stabilize the network by fixing the random seed
    # fixing the seed will settle down all the random parameters over the entire routine below
    np.random.seed(1)
    tf.set_random_seed(1)
    
    print "Re-train the model at epoch ", best_epoch, np.shape(xts_auto), np.shape(xts_x), np.shape(ytest)
    
    with tf.Session() as sess:
        
        if method == 'linear':
            
            clf = mixture_linear(sess, para_lr_linear, para_l2, para_batch_size_linear, para_order_auto, \
                                  para_order_x, para_order_steps, para_y_log, para_bool_bilinear,\
                                  para_loss_type, para_distr_type, para_activation_type, para_pos_regu, para_gate_type)
            
            # overall para 
            para_n_epoch = para_n_epoch_linear
            para_batch_size = para_batch_size_linear
            para_keep_prob = 1.0
            
            model_file += '_linear_lk.ckpt'
            
        elif method == 'bayes':
            
            # use the beat epoch 
            para_num_iter = best_epoch * (len(xtr_auto)/para_batch_size_bayes)
            
            clf = variational_mixture_linear(sess, para_lr_bayes, para_batch_size_bayes, para_order_auto, para_order_x,\
                                             para_order_steps, para_y_log, para_bool_bilinear, para_distr_type,\
                                             para_eval_sample_num, 'variational', para_num_iter, para_loss_sample_num, \
                                             int(np.shape(xtrain)[0]/para_batch_size_bayes))
            # global para
            para_n_epoch = para_n_epoch_bayes
            para_batch_size = para_batch_size_bayes
            para_keep_prob = 1.0
            
        else:
            print "     [ERROR] Need to specify a model"
            
            
        # initialize the network
        clf.train_ini()
        clf.evaluate_ini()
        
        # setup mini-batch parameters
        total_cnt   = np.shape(xtrain)[0]
        total_batch = int(total_cnt/para_batch_size)
        total_idx   = range(total_cnt)
        
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
                batch_x = xtr_x[ batch_idx ]
                
                if para_y_log == True:
                    batch_y = log(ytrain[batch_idx]+1e-5)
                else:
                    batch_y = ytrain[batch_idx]
            
                tmpc += clf.train_batch( batch_auto, batch_x, batch_y, para_keep_prob )
            
            
            tmp_train_rmse, tmp_train_mae, tmp_train_mape, tmp_train_regu = \
            clf.inference(xtr_auto, xtr_x, ytrain, para_keep_prob)
            #?
            tmp_test_rmse, tmp_test_mae, tmp_test_mape, tmp_test_regu = \
            clf.inference(xts_auto, xts_x, ytest,  para_keep_prob) 
            
            # record for fixing the parameter set-up in testing
            tmp_epoch_err.append( [epoch, tmp_train_rmse, tmp_test_rmse] )
            
            print "loss on epoch ", epoch, " : ", 1.0*tmpc/total_batch, tmp_train_rmse, tmp_test_rmse
            
        print "Model Re-training Finished!"
        
        
        # record prediction and mixture gate values 
        # [3, N]
        py_test = clf.predict(xts_auto, xts_x, para_keep_prob)
        tmp = np.concatenate( [np.expand_dims(ytest, -1), np.transpose(py_test, [1, 0])], 1 )
        np.savetxt( file_addr + "pytest_mix.txt", tmp, delimiter=',')
        
        py_train = clf.predict(xtr_auto, xtr_x, para_keep_prob)
        tmp = np.concatenate( [np.expand_dims(ytrain, -1), np.transpose(py_train, [1, 0])], 1 )
        np.savetxt( file_addr + "pytrain_mix.txt", tmp, delimiter=',')
        
        gates_test = clf.predict_gates(xts_auto, xts_x, para_keep_prob)
        np.savetxt( file_addr + "gate_test.txt", gates_test, delimiter=',')
        
        gates_train = clf.predict_gates(xtr_auto, xtr_x, para_keep_prob)
        np.savetxt( file_addr + "gate_train.txt", gates_train, delimiter=',')
        
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
        
        
        # reset the model 
        clf.model_reset()
        
    # clear the graph in the current session    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    
    return [ [tmp_test_rmse, tmp_test_mae, tmp_test_mape], min(tmp_epoch_err, key = lambda x:x[-1]) ]
'''

def preprocess_feature_mixture(xtrain, xtest):
    
    # split training and testing data into three feature groups      
    xtr_vol =   np.asarray( [j[0] for j in xtrain] )
    xtr_feature = np.asarray( [j[1] for j in xtrain] )

    xts_vol =   np.asarray( [j[0] for j in xtest] )
    xts_feature = np.asarray( [j[1] for j in xtest] )

    # !! IMPORTANT: feature normalization

    xts = conti_normalization_test_dta(  xts_vol, xtr_vol )
    xtr = conti_normalization_train_dta( xtr_vol )

    xts_exter = conti_normalization_test_dta(  xts_feature, xtr_feature )
    xtr_exter = conti_normalization_train_dta( xtr_feature )
    
    return np.asarray(xtr), np.asarray(xtr_exter), np.asarray(xts), np.asarray(xts_exter)
    
    
# ---- main process ----  

if train_mode == 'oneshot':
    
    # result log
    log_error   = "../bt_results/res/oneshot/mix.txt"
    model_file = '../bt_results/model/mix'
    pred_file  = "../bt_results/res/oneshot/"
    
    # load pre-processed training and testing data
    file_postfix = "v_minu_mix"
    xtrain = np.load("../dataset/bitcoin/training_data/xtrain_"+file_postfix+".dat")
    xtest  = np.load("../dataset/bitcoin/training_data/xtest_" +file_postfix+".dat")
    ytrain = np.load("../dataset/bitcoin/training_data/ytrain_"+file_postfix+".dat")
    ytest  = np.load("../dataset/bitcoin/training_data/ytest_" +file_postfix+".dat")
    print np.shape(xtrain), np.shape(ytrain), np.shape(xtest), np.shape(ytest)
    
    # extract different groups of features
    xtr, xtr_exter, xts, xts_exter = preprocess_feature_mixture(xtrain, xtest)
    
    print '--- Start one-shot training: '
    print np.shape(xtr), np.shape(xtr_exter)
    print np.shape(xts), np.shape(xts_exter)
    
    # fixed parameters
    para_order_auto = len(xtr[0])
    
    if len(np.shape(xtr_exter)) > 2:
        para_bool_bilinear = True
        para_order_x = len(xtr_exter[0][0])
        para_order_steps = len(xtr_exter[0])
        print '     !! Time-first order !! '

    else:
        para_bool_bilinear = False
        para_order_x = len(xtr_exter[0])
        para_order_steps = 0
        print '     !! Flattened features !! '
    
    tmp_error = train_eval_mixture( xtr, xtr_exter, ytrain, xts, xts_exter, ytest, pred_file, model_file ) 
    
    # save overall errors
    with open(log_error, "a") as text_file:
        text_file.write( "Mixture %s  %s %s %s : %s  \n"%(method, para_loss_type, para_distr_type, 
                                                          'bi-linear' if para_bool_bilinear else 'linear', \
                                                          str(tmp_error)) ) 
elif train_mode == 'roll' or 'incre':
    
    # fix random seed
    np.random.seed(1)
    tf.set_random_seed(1)
    
    # ---- prepare the log
    
    log_error = "../bt_results/res/rolling/log_error_mix.txt"
    log_epoch  = "../bt_results/res/rolling/log_epoch_mix.txt"
    
    roll_title = "\n -------- Rolling --------- \n"
    incre_title = "\n -------- Incremental --------- \n"
    
    # prepare log files
    if method == 'linear':
        
        with open(log_error, "a") as text_file:
            text_file.write("\n %s Mixture %s  %s %s %s %s %s \n\n"%(roll_title if train_mode == 'roll' else incre_title,
                                                                     method,
                                                                     para_loss_type, 
                                                                     para_distr_type, 
                                                                     'bi-linear' if para_bool_bilinear == True else 'linear', 
                                                                     para_activation_type, 
                                                                     'pos_regu' if para_pos_regu == True else 'no_pos_regu'))
    
    elif method == 'bayes':
        
        with open(log_error, "a") as text_file:
            text_file.write("\n %s Mixture %s  %s %s %s : \n"%(roll_title if train_mode == 'roll' else incre_title,
                                                               method, 
                                                               para_loss_type, 
                                                               para_distr_type, 
                                                               'bi-linear' if para_bool_bilinear == True else 'linear'))
    
    
    # ---- prepare the data
    
    # load raw feature and target data
    features_minu = np.load("../dataset/bitcoin/training_data/feature_minu.dat" )
    rvol_hour = np.load("../dataset/bitcoin/training_data/return_vol_hour.dat" )
    all_loc_hour = np.load("../dataset/bitcoin/loc_hour.dat" )
    print '--- Start the ' + train_mode + ' training: \n', np.shape(features_minu), np.shape(rvol_hour)
    
    # prepare the set of pairs of features and targets
    x, y, var_explain = prepare_feature_target(features_minu, rvol_hour, all_loc_hour, \
                                               para_order_minu, para_order_hour, \
                                               bool_feature_selection, para_step_ahead, False)
    
    # set up the training and evaluation interval 
    interval_num = int(len(y)/interval_len)
    
    
    # ---- the main loop
    
    # interval 
    for i in range(roll_len + 1, interval_num + 1):
        
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
            print '[ERROR] training mode'
        
        print '\n --- In processing of interval ', i-1, ' --- \n'
        
        
        # training, validation+testing split 
        if para_bool_bilinear == True:
            xtrain, ytrain, xtest, ytest = training_testing_mixture_rnn(tmp_x, tmp_y, para_train_split_ratio)
        else:
            xtrain, ytrain, xtest, ytest = training_testing_mixture_mlp(tmp_x, tmp_y, para_train_split_ratio)
            
        
        # feature split, normalization READY
        xtr, xtr_exter, xtest, xtest_exter = preprocess_feature_mixture(xtrain, xtest)
        
        # build validation and testing data 
        tmp_idx = range(len(xtest))
        tmp_val_idx = []
        tmp_ts_idx = []
        
        # even sampling the validation and testing data
        for j in tmp_idx:
            if j%2 == 0:
                tmp_val_idx.append(j)
            else:
                tmp_ts_idx.append(j)
        
        xval = xtest[tmp_val_idx]
        xval_exter = xtest_exter[tmp_val_idx]
        yval = np.asarray(ytest)[tmp_val_idx]
        
        xts = xtest[tmp_ts_idx]
        xts_exter = xtest_exter[tmp_ts_idx]
        yts = np.asarray(ytest)[tmp_ts_idx]
        
        print 'shape of training, validation and testing data: \n'                            
        print np.shape(xtr), np.shape(xtr_exter), np.shape(ytrain)
        print np.shape(xval), np.shape(xval_exter), np.shape(yval)
        print np.shape(xts), np.shape(xts_exter), np.shape(yts)
        
        # dump training and testing data in one interval to disk 
        '''
        np.asarray(xtrain).dump("../dataset/bitcoin/training_data/rolling/" + str(i-1) + "_xtrain_mix.dat")
        np.asarray(xtest ).dump("../dataset/bitcoin/training_data/rolling/" + str(i-1) + "_xtest_mix.dat")
        np.asarray(ytrain).dump("../dataset/bitcoin/training_data/rolling/" + str(i-1) + "_ytrain_mix.dat")
        np.asarray(ytest ).dump("../dataset/bitcoin/training_data/rolling/" + str(i-1) + "_ytest_mix.dat")
        '''
        
        # parameter set-up
        para_order_auto = para_order_hour
        
        if para_bool_bilinear == True:
            para_order_x = len(xtr_exter[0][0])
            para_order_steps = len(xtr_exter[0])
            print '     !! Time-first order !! '

        elif para_bool_bilinear == False:
            para_order_x = len(xtr_exter[0])
            para_order_steps = 0
            print '     !! Flattened features !! '
        else:
            print ' [ERROR]  bi-linear '
        
        
        # -- training and validation phase
        
        para_train_vali = []
        
        for para_lr_linear in [0.001, 0.005, 0.01]:
            for para_l2 in [0.0001, 0.001, 0.01, 0.1]:
                
                tmp_epoch, tmp_tr_rmse, tmp_val_rmse, _ = train_validate_mixture(xtr, 
                                                                                 xtr_exter, 
                                                                                 np.asarray(ytrain), 
                                                                                 xval,
                                                                                 xval_exter, 
                                                                                 np.asarray(yval))
                
                para_train_vali.append( [para_lr_linear, para_l2, tmp_epoch, tmp_tr_rmse, tmp_val_rmse] )
            
                print 'Current parameter set-up: \n', para_train_vali[-1], '\n'
            
        
        # -- testing phase
        
        # best global hyper-parameter
        final_para = min(para_train_vali, key = lambda x:x[-1])
        
        para_lr_linear = final_para[0]
        para_l2 = final_para[1]
        para_epoch_linear = final_para[2]
        
        print ' ---- Best parameters : ', final_para, '\n'
        
        result_tuple = [final_para[3], final_para[4]]
        
        if len(xts) == 0:
            test_error = train_validate_mixture(xtr, 
                                                xtr_exter, 
                                                np.asarray(ytrain), 
                                                xval, 
                                                xval_exter, 
                                                np.asarray(yval))
            result_tuple.append(None)
            
        else:
            test_error = train_validate_mixture(xtr, 
                                                xtr_exter, 
                                                np.asarray(ytrain), 
                                                xts, 
                                                xts_exter, 
                                                np.asarray(yts))
            result_tuple.append(test_error)
        
        print ' ---- Training, validation and testing performance: ', final_para, test_error, '\n'
        
        
        # -- log overall errors
        
        with open(log_error, "a") as text_file:
            text_file.write("Interval %d : %s, %s \n" %(i-1, str(final_para[:2]), str(result_tuple)))
        
else:
    print '[ERROR] training mode'
