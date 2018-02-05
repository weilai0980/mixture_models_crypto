#!/usr/bin/python

import sys
import os

# local packages
from utils_libs import *
from utils_data_prep import *
from regression_models import *


# ONLY USED FOR ROLLING EVALUATION
# ---- parameter set-up for preparing trainning and testing data ----
para_order_minu = 50
para_order_hour = 16
bool_feature_selection = False
bool_add_feature = True
# ----


# ---- model and training log set-up ----
result_file = "../bt_results/res/rolling/reg_v_minu.txt"
model_file = "../bt_results/model/v_minu_inter"
bool_clf = False

# clean the log
#with open(result_file, "w") as text_file:
#    text_file.close()

load_file_postfix = "v_minu_reg"
model_list = ['gbt', 'rf', 'xgt', 'gp', 'bayes', 'enet', 'ridge', 'ewma']
# 'gbt', 'rf', 'xgt', 'gp', 'bayes', 'enet', 'ridge', 'lasso'

def train_eval_models( xtrain, ytrain, xtest, ytest ):
    
    best_err_ts = []
    
    
    # GBT gradient boosted tree
    tmperr = gbt_train_validate(xtrain, ytrain, xtest, ytest, 0.0, bool_clf, result_file, model_file + '_gbt.sav', file_path)
    best_err_ts.append(tmperr)
    
    # Random forest performance
    tmperr = rf_train_validate(xtrain, ytrain, xtest, ytest, bool_clf, result_file, model_file + '_rf.sav', file_path)
    best_err_ts.append(tmperr)
    
    # XGBoosted extreme gradient boosted
    tmperr = xgt_train_validate(xtrain, ytrain, xtest, ytest, bool_clf, 0, result_file, model_file + '_xgt.sav', file_path)
    best_err_ts.append(tmperr)
    
   
    # log transformation of y
    log_ytrain = []
    #log(ytrain+1e-5)
    
    # note: remove the training error calculation
    # Gaussain process 
    tmperr = gp_train_validate(xtrain, ytrain, xtest, ytest, result_file, model_file + '_gp.sav', log_ytrain, file_path)
    best_err_ts.append(tmperr)
    
    
    # Bayesian regression
    #tmperr = bayesian_reg_train_validate(xtrain, ytrain, xtest, ytest, result_file, model_file + '_bayes.sav', log_ytrain,\
    #                                    file_path)
    #best_err_ts.append(tmperr)
    
    # ElasticNet
    tmperr = elastic_net_train_validate(xtrain, ytrain, xtest, ytest, result_file, model_file + '_enet.sav', log_ytrain,\
                                       file_path)
    best_err_ts.append(tmperr)
    
    #Ridge regression
    #tmperr = ridge_reg_train_validate(xtrain, ytrain, xtest, ytest, result_file, model_file + '_ridge.sav', log_ytrain,\
    #                                  file_path)
    #best_err_ts.append(tmperr)
    
    # Lasso 
    #tmperr = lasso_train_validate(xtrain, ytrain, xtest, ytest, result_file, model_file + '_lasso.sav', log_ytrain, file_path)
    #best_err_ts.append(tmperr)
    
    # EWMA
    #tmperr = ewma_validate(ytrain, ytest, result_file, file_path)
    #best_err_ts.append(tmperr)
    
    
    return best_err_ts


# --- main process ---
# oneshot, roll, incre
train_mode = str(sys.argv[1])

if train_mode == 'oneshot':
    
    # log file address
    result_file = "../bt_results/res/reg_v_minu.txt"
    model_file = "../bt_results/model/v_minu"
    
    # log for predictions
    file_path = "../bt_results/res/"
    
    # load data ready for training and testing 
    xtrain = np.load("../dataset/bitcoin/training_data/xtrain_"+load_file_postfix+".dat")
    xtest  = np.load("../dataset/bitcoin/training_data/xtest_" +load_file_postfix+".dat")
    ytrain = np.load("../dataset/bitcoin/training_data/ytrain_"+load_file_postfix+".dat")
    ytest  = np.load("../dataset/bitcoin/training_data/ytest_" +load_file_postfix+".dat")

    # all feature vectors are already normalized
    print '--- Start one-shot training: ', np.shape(xtrain), np.shape(ytrain), np.shape(xtest), np.shape(ytest)
    
    train_eval_models( xtrain, ytrain, xtest, ytest )
    
elif train_mode == 'roll' or 'incre':
    
    
    # load raw feature data
    features_minu = np.load("../dataset/bitcoin/training_data/feature_minu.dat")
    rvol_hour = np.load("../dataset/bitcoin/training_data/return_vol_hour.dat")
    all_loc_hour = np.load("../dataset/bitcoin/loc_hour.dat")
    print '--- Start the ' + train_mode + ' training: \n', np.shape(features_minu), np.shape(rvol_hour)
    
    # prepare pairs of features and targets
    if bool_add_feature == True:
        x, y, var_explain = prepare_feature_target( features_minu, rvol_hour, all_loc_hour, \
                                                        para_order_minu, para_order_hour, bool_feature_selection )
    else:
        x, y, var_explain = prepare_feature_target( [], rvol_hour, all_loc_hour, \
                                                        para_order_minu, para_order_hour, bool_feature_selection )
        
    # set up the interval parameters
    interval_len = 30*24
    interval_num = int(len(y)/interval_len)
    print np.shape(x), np.shape(y), interval_len, interval_num
    roll_len = 2
    '''
    # the main loop 
    for i in range(roll_len + 1, interval_num+1):
        
        # log for predictions in each interval
        file_path = "../bt_results/res/rolling/" + str(i-1) + "_"
        
        print '\n --- In processing of interval ', i-1, ' --- \n'
        with open(result_file, "a") as text_file:
            text_file.write( "Interval %d :\n" %(i-1) )
        
        if train_mode == 'roll':
            tmp_x = x[(i-roll_len-1)*interval_len: i*interval_len]
            tmp_y = y[(i-roll_len-1)*interval_len: i*interval_len]
            para_train_split_ratio = 1.0*(len(tmp_x) - interval_len)/len(tmp_x)
            
        elif train_mode == 'incre':
            tmp_x = x[ : i*interval_len]
            tmp_y = y[ : i*interval_len]
            para_train_split_ratio = 1.0*(len(tmp_x) - interval_len)/len(tmp_x)
            
        else:
            print '[ERROR] training mode'
        
        
        # split into training and testin data, normalization
        xtrain, ytrain, xtest, ytest = training_testing_plain_regression(tmp_x, tmp_y, para_train_split_ratio)
        print np.shape(xtrain), np.shape(ytrain), np.shape(xtest), np.shape(ytest)
        
        
        # train and evaluate models
        # arguments: numpy array 
        tmp_errors = train_eval_models( np.asarray(xtrain), np.asarray(ytrain), np.asarray(xtest), np.asarray(ytest) )
        
        print list(zip(model_list, tmp_errors))
    '''
else:
    print '[ERROR] training mode'

