#!/usr/bin/python

import sys
import os

# local packages
from utils_libs import *
from utils_data_prep import *
from regression_models import *

# --- parameter set-up from parameter-file ---

# load the parameter file
para_dict = para_parser("para_file.txt")

para_order_minu = para_dict['para_order_minu']
para_order_hour = para_dict['para_order_hour']
bool_feature_selection = para_dict['bool_feature_selection']

# ONLY USED FOR ROLLING EVALUATION
interval_len = para_dict['interval_len']
roll_len = para_dict['roll_len']

para_step_ahead = para_dict['para_step_ahead']

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
model_list = ['rf', 'xgt', 'gbt', 'enet', 'ewma', 'bayes', 'ridge', 'lasso']
# 'gbt', 'rf', 'xgt', 'gp', 'bayes', 'enet', 'ridge', 'lasso', 'ewma'

def train_eval_models( xtrain, ytrain, xval, yval, xtest, ytest, autotrain, autoval, autotest ):
    
    '''
    Argu: numpy array
    
    '''
    
    best_err_ts = []
   
    # log transformation of y
    log_ytrain = []
    #log(ytrain+1e-5)
    
    # note: remove the training error calculation
    # Gaussain process 
    if 'gp' in model_list:
        tmperr = gp_train_validate(xtrain, ytrain, xval, yval, xtest, ytest, result_file, model_file + '_gp.sav', log_ytrain,\
                                   file_path)
        best_err_ts.append(tmperr)
    
    
    # Bayesian regression
    if 'bayes' in model_list:
        tmperr = bayesian_reg_train_validate(xtrain, ytrain, xval, yval, xtest, ytest, result_file, model_file + '_bayes.sav',\
                                             log_ytrain, file_path)
        best_err_ts.append(tmperr)
    
    # ElasticNet
    if 'enet' in model_list:
        tmperr = elastic_net_train_validate(xtrain, ytrain, xval, yval, xtest, ytest, result_file, model_file + '_enet.sav',\
                                            log_ytrain, file_path)
        best_err_ts.append(tmperr)
    
    #Ridge regression
    if 'ridge' in model_list:
        tmperr = ridge_reg_train_validate(xtrain, ytrain, xval, yval, xtest, ytest, result_file, model_file + '_ridge.sav',\
                                          log_ytrain, file_path)
        best_err_ts.append(tmperr)
    
    # Lasso 
    if 'lasso' in model_list:
        tmperr = lasso_train_validate(xtrain, ytrain, xval, yval, xtest, ytest, result_file, model_file + '_lasso.sav', \
                                      log_ytrain, file_path)
        best_err_ts.append(tmperr)
    
    # EWMA
    if 'ewma' in model_list:
        if para_step_ahead != 0 or len(xtest) != 0: 
            tmperr = ewma_instance_validate(autotrain, ytrain, autoval, yval, autotest, ytest, result_file, file_path)
        
        else:
            tmperr = ewma_validate(ytrain, yval, result_file, file_path)
            
        best_err_ts.append(tmperr)
        
    
    # GBT gradient boosted tree
    if 'gbt' in model_list:
        tmperr = gbt_train_validate(xtrain, ytrain, xval, yval, xtest, ytest, 0.0, bool_clf, result_file, \
                                    model_file +'_gbt.sav', file_path)
        best_err_ts.append(tmperr)
    
    # Random forest performance
    if 'rf' in model_list:
        tmperr = rf_train_validate(xtrain, ytrain, xval, yval, xtest, ytest, bool_clf, result_file, model_file + '_rf.sav',
                                   file_path)
        best_err_ts.append(tmperr)
    
    # XGBoosted extreme gradient boosted
    if 'xgt' in model_list:
        tmperr = xgt_train_validate(xtrain, ytrain, xval, yval, xtest, ytest, bool_clf, 0, result_file, 
                                    model_file + '_xgt.sav', file_path)
        best_err_ts.append(tmperr)
    
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
    # auto-regressive feature first
    if bool_add_feature == True:
        x, y, var_explain = prepare_feature_target( features_minu, rvol_hour, all_loc_hour, \
                                                    para_order_minu, para_order_hour, bool_feature_selection, para_step_ahead)
    else:
        x, y, var_explain = prepare_feature_target( [], rvol_hour, all_loc_hour, \
                                                    para_order_minu, para_order_hour, bool_feature_selection, para_step_ahead)
    
    # set up the training and evaluation interval 
    interval_num = int(len(y)/interval_len)
    print np.shape(x), np.shape(y), interval_len, interval_num
    
    # prepare the log
    with open(result_file, "a") as text_file:
        text_file.write("\n %s "%( "\n -------- Rolling --------- \n" if train_mode == 'roll' else "\n -------- Incremental --------- \n"))
    
    # the main loop 
    for i in range(roll_len + 1, interval_num+1):
        
        # log for predictions in each interval
        file_path = "../bt_results/res/rolling/" + str(i-1) + "_"
        
        print '\n --- In processing of interval ', i-1, ' --- \n'
        
        with open(result_file, "a") as text_file:
            text_file.write( "\n Interval %d :\n" %(i-1) )
        
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
        
        # auto-regressive part
        auto = [ j[0] for j in tmp_x ]
        auto_tr = auto[:int(para_train_split_ratio*len(tmp_y))] 
        auto_test = auto[int(para_train_split_ratio*len(tmp_y)):]
        
        # split data, normalization
        xtr, ytr, xtest, ytest = training_testing_plain_regression(tmp_x, tmp_y, para_train_split_ratio)
        #print np.shape(xtr), np.shape(ytr), np.shape(xtest), np.shape(ytest)
        
        # build validation and testing data 
        tmp_idx = range(len(xtest))
        tmp_val_idx = []
        tmp_ts_idx = []
        
        for j in tmp_idx:
            if j%2 == 0:
                tmp_val_idx.append(j)
            else:
                tmp_ts_idx.append(j)
        
        xval = xtest[tmp_val_idx]
        yval = np.asarray(ytest)[tmp_val_idx]
        
        xts = xtest[tmp_ts_idx]
        yts = np.asarray(ytest)[tmp_ts_idx]
        
        auto_val = np.asarray(auto_test)[tmp_val_idx] 
        auto_ts  = np.asarray(auto_test)[tmp_ts_idx]
        
        print 'shape of training, validation and testing data: \n',
        print np.shape(xtr),  np.shape(auto_tr), np.shape(ytr)
        print np.shape(xval), np.shape(auto_val), np.shape(yval)
        print np.shape(xts),  np.shape(auto_ts), np.shape(yts)
        
        # train, validate and test models
        tmp_errors = train_eval_models( np.asarray(xtr), np.asarray(ytr), np.asarray(xval), np.asarray(yval),
                                        np.asarray(xts), np.asarray(yts), np.asarray(auto_tr), np.asarray(auto_val),\
                                        np.asarray(auto_ts) )
        
        print list(zip(model_list, tmp_errors))
    
else:
    print '[ERROR] training mode'
