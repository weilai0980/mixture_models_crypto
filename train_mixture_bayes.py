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
from mixture import *


# ONLY USED FOR ROLLING EVALUATION
# ---- parameter set-up for preparing trainning and testing data ----
para_order_minu = 30
para_order_hour = 16
bool_feature_selection = False

para_bool_bilinear = True
# ----


print '--- Argument List:', str(sys.argv)
method = str(sys.argv[1])
# linear, mlp, rnn
para_loss_type = str(sys.argv[2])
#'lk', 'sg'
para_distr_type = str(sys.argv[3])
#'log', 'norm'
train_mode = str(sys.argv[4])
# oneshot, roll, incre

# ---- common parameters ----

# log files for training process 
training_log  = "../bt_results/res/mix_train_log.txt"
# initialize the log
with open(training_log, "w") as text_file:
    text_file.close()



# ---- Approach specific parameters ----

# -- variational  
para_lr_linear = 0.001
para_n_epoch_linear = 300
para_batch_size_linear = 1440
para_l2_linear = 0.001

para_y_log = False
para_pred_exp = False

para_step_gap = 0
para_eval_sample = 100

# ---- training and evalution methods ----
    
def train_mixture( xtr_v, xtr_distr, ytrain, xts_v, xts_distr, ytest ):   
    
    tmp_test_err = []
            
    # stabilize the network by fixing the random seed
    np.random.seed(1)
    tf.set_random_seed(1)
        
    with tf.Session() as sess:
        
        if method == 'bayes-linear':
            
            clf = variational_mixture_linear(sess, para_lr_linear, para_l2_linear, para_batch_size_linear, para_order_v, \
                                  para_order_distr, para_order_steps, para_y_log, para_bool_bilinear,\
                                  para_loss_type, para_distr_type, para_eval_sample)
            
            para_n_epoch = para_n_epoch_linear
            para_batch_size = para_batch_size_linear
            para_keep_prob = 1.0
        
        else:
            print "     [ERROR] Need to specify a model"
            
            
        # initialize the network
        # reset the model
        clf.train_ini()
        clf.evaluate_variational_ini()
        
        
        # set up training batch parameters
        total_cnt   = np.shape(xtrain)[0]
        total_batch = int(total_cnt/para_batch_size)
        total_idx   = range(total_cnt)
        
        #  begin training epochs
        for epoch in range(para_n_epoch):
            
            # shuffle traning instances each epoch
            np.random.shuffle(total_idx)
            
            # loop over all batches
            tmpc = 0.0
            for i in range(total_batch):
                
                batch_idx = total_idx[ i*para_batch_size: (i+1)*para_batch_size ] 
            
                batch_v     = xtr_v[ batch_idx ]
                batch_distr = xtr_distr[ batch_idx ]
                
                batch_y = ytrain[batch_idx]
            
                #tmpc += clf.train_batch( batch_v, batch_distr, batch_y, para_keep_prob )
                
                # bayesian test
                #print '----- variable to optimize: ', clf.test( batch_v, batch_distr, batch_y, para_keep_prob )
                clf.train_varitational( batch_v, batch_distr, batch_y, para_keep_prob )
            
            #?
            tmp_train_acc = clf.evaluate_metric(xtr_v, xtr_distr, ytrain, para_keep_prob)
            #?
            tmp_test_acc  = clf.evaluate_metric(xts_v, xts_distr, ytest,  para_keep_prob) 
            
            y_hat = clf.predict(xts_v, xts_distr,  para_keep_prob) 
            
            print ytest[:10]
            print y_hat[0][:10]
            
            # record for re-tratining the model afterwards
            #tmp_test_err.append( [epoch, sqrt(tmp_train_acc), sqrt(tmp_test_acc), tmpc] )
            
            print "loss on epoch ", epoch, " : ", sqrt(tmp_train_acc), sqrt(tmp_test_acc)
            
        print "Optimization Finished!"
        
        #clf.varitational_eval(xts_v[:10], xts_distr[:10], ytest[:10],  para_keep_prob) 
        
        
        # the model at the best parameter above
        #best_epoch = min(tmp_test_err, key = lambda x:x[2])[0]
        
        # reset the model
        clf.model_reset()
        clf.train_ini()
    
        
    # clear the graph in the current session
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    
    return 0
    #return best_epoch, min(tmp_test_err, key = lambda x:x[2])
    
    
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
    res_file    = "../bt_results/res/mix.txt"
    model_file = '../bt_results/model/mix'
    pred_file = "../bt_results/res/"
    
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
    para_order_v = len(xtr[0])
    
    if len(np.shape(xtr_exter)) > 2:
        para_bool_bilinear = True
        para_order_distr = len(xtr_exter[0][0])
        para_order_steps = len(xtr_exter[0])
        print '     !! Time-first order !! '

    else:
        para_bool_bilinear = False
        para_order_distr = len(xtr_exter[0])
        para_order_steps = 0
        print '     !! Flattened features !! '
    
    tmp_error = train_eval_mixture( xtr, xtr_exter, ytrain, xts, xts_exter, ytest, pred_file, model_file ) 
    
    # save overall errors
    with open(res_file, "a") as text_file:
        text_file.write( "Mixture %s  %s %s %s : %s  \n"%(method, para_loss_type, para_distr_type, 
                                                          'bi-linear' if para_bool_bilinear else 'linear', \
                                                          str(tmp_error)) ) 
    
elif train_mode == 'roll' or 'incre':
    
    # fix random seed
    np.random.seed(1)
    tf.set_random_seed(1)
    
    # result logs
    res_file   = "../bt_results/res/rolling/reg_mix.txt"
    model_file = "../bt_results/model/mix_"
    pred_file = ""
    
    # load raw feature and target data
    features_minu = np.load("../dataset/bitcoin/training_data/feature_minu.dat")
    rvol_hour = np.load("../dataset/bitcoin/training_data/return_vol_hour.dat")
    all_loc_hour = np.load("../dataset/bitcoin/loc_hour.dat")
    print '--- Start the ' + train_mode + ' training: \n', np.shape(features_minu), np.shape(rvol_hour)
    
    # prepare the set of pairs of features and targets
    x, y, var_explain = prepare_feature_target( features_minu, rvol_hour, all_loc_hour, \
                                                para_order_minu, para_order_hour, bool_feature_selection, para_step_gap )
    
    
    print 'test : ', np.shape(x), np.shape(y) 
    
    # set up the training and evaluation interval 
    interval_len = 30*24
    interval_num = int(len(y)/interval_len)
    print np.shape(x), np.shape(y), interval_len, interval_num
    roll_len = 2
    
    
    # note down prediction errors 
    with open(res_file, "a") as text_file:
            text_file.write( "\n" )
    
    
    # the main loop
    #interval_num + 1
    for i in range(roll_len + 1, roll_len + 2):
        
        # reset the graph
        tf.reset_default_graph()
        sess = tf.InteractiveSession()
        
        # log for predictions in each interval
        pred_file = "../bt_results/res/rolling/" + str(i-1) + "_" + str(para_step_gap) + '_'
        
        print '\n --- In processing of interval ', i-1, ' --- \n'
        with open(res_file, "a") as text_file:
            text_file.write( "Interval %d :\n" %(i-1) )
        
        # extract the data within the current time interval
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
            
        # process the data within the current interval, flatten or 
        if para_bool_bilinear == True:
            xtrain, ytrain, xtest, ytest = training_testing_mixture_rnn(tmp_x, tmp_y, para_train_split_ratio)
        else:
            xtrain, ytrain, xtest, ytest = training_testing_mixture_mlp(tmp_x, tmp_y, para_train_split_ratio)
        
        # dump training and testing data in one interval to disk 
        '''
        np.asarray(xtrain).dump("../dataset/bitcoin/training_data/rolling/" + str(i-1) + "_xtrain_mix.dat")
        np.asarray(xtest ).dump("../dataset/bitcoin/training_data/rolling/" + str(i-1) + "_xtest_mix.dat")
        np.asarray(ytrain).dump("../dataset/bitcoin/training_data/rolling/" + str(i-1) + "_ytrain_mix.dat")
        np.asarray(ytest ).dump("../dataset/bitcoin/training_data/rolling/" + str(i-1) + "_ytest_mix.dat")
        '''
        
        # feature split, normalization READY
        xtr, xtr_exter, xts, xts_exter = preprocess_feature_mixture(xtrain, xtest)
        print '\n In processing of interval ', i-1, '\n'
        print np.shape(xtr), np.shape(xtr_exter)
        print np.shape(xts), np.shape(xts_exter)
        
        # parameter set-up
        para_order_v = para_order_hour
        if para_bool_bilinear == True:
            para_order_distr = len(xtr_exter[0][0])
            para_order_steps = len(xtr_exter[0])
            print '     !! Time-first order !! '

        elif para_bool_bilinear == False:
            para_order_distr = len(xtr_exter[0])
            para_order_steps = 0
            print '     !! Flattened features !! '
        else:
            print '[ERROR]  bi-linear '
            
            
        # training begins
        # return lowest prediction errors and llk 
        best_epoch, tmp_error = train_mixture( xtr, xtr_exter, np.asarray(ytrain), xts, xts_exter, np.asarray(ytest) )
        
        
        '''
        print tmp_error
        
        # save overall errors
        with open(res_file, "a") as text_file:
            text_file.write( "Mixture %s  %s %s %s : %s  \n"%(method, para_loss_type, para_distr_type, 
                                                              'bi-linear' if para_bool_bilinear else 'linear', \
                                                              str(tmp_error)) )
       '''
        
else:
    print '[ERROR] training mode'
