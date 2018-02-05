#!/usr/bin/python
from utils_libs import *
from utils_data_prep import *

from numpy import prod
import math

import sys
import os

# statiscal models
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.api import VAR, DynamicVAR
from statsmodels.stats import diagnostic

import pyflux as pf


# ONLY USED FOR ROLLING EVALUATION
# ---- parameter set-up for preparing trainning and testing data ----
para_order_minu = 20
para_order_hour = 16
bool_feature_selection = False
# ----


# ---- model and training log set-up ----
result_file = "../bt_results/res/rolling/reg_v_minu_stats.txt"
log_file = "../bt_results/res/arima"
model_file = "../bt_results/model/v_minu"


# ---- training methods ----
def oneshot_prediction_arimax( xtr, extr, xts, exts, arima_order, bool_add_ex ):
    
    xts_hat = []
        
    if bool_add_ex == True:
        mod = sm.tsa.statespace.SARIMAX(endog = xtr, exog = extr, order = arima_order)
            
        fit_res = mod.fit(disp=False)
        predict = fit_res.get_forecast( len(xts), exog = exts )
        predict_ci = predict.conf_int()
    
    else:
        mod = sm.tsa.statespace.SARIMAX(endog = xtr, order = arima_order)
            
        fit_res = mod.fit(disp=False )
        predict = fit_res.get_forecast( len(xts) )
        predict_ci = predict.conf_int()
        
    tr_predict = fit_res.get_prediction()
    tr_predict_ci = predict.conf_int()
    
    xtr_hat = tr_predict.predicted_mean 
    xts_hat = predict.predicted_mean
    
    # out-sample forecast, in-sample rmse, out-sample rmse      
    return xts_hat, xtr_hat, sqrt(mean((xts - np.asarray(xts_hat))*(xts - np.asarray(xts_hat)))), \
sqrt(mean((xtr - np.asarray(xtr_hat))*(xtr - np.asarray(xtr_hat))))

def oneshot_prediction_strx( xtr, extr, xts, exts, bool_add_ex ):
    
    xts_hat = []
    
    if bool_add_ex == True:
        roll_mod = sm.tsa.UnobservedComponents(endog = xtr, exog = extr, level= 'local linear trend', trend = True )
            
        fit_res = roll_mod.fit(disp=False)
        predict = fit_res.get_forecast(len(xts), exog = exts)
        predict_ci = predict.conf_int()
            
    else:
        roll_mod = sm.tsa.UnobservedComponents(endog = xtr, level= 'local linear trend', trend = True )
            
        fit_res = roll_mod.fit(disp=False)
        predict = fit_res.get_forecast(len(xts))
        predict_ci = predict.conf_int()
            
    tr_predict = fit_res.get_prediction()
    tr_predict_ci = predict.conf_int()
            
    xts_hat = predict.predicted_mean
    xtr_hat = tr_predict.predicted_mean    
    
    # out-sample forecast, in-sample rmse, out-sample rmse      
    return xts_hat, xtr_hat, sqrt(mean((xts - np.asarray(xts_hat))*(xts - np.asarray(xts_hat)))), \
sqrt(mean((xtr - np.asarray(xtr_hat))*(xtr - np.asarray(xtr_hat))))

def roll_prediction_arimax( xtr, extr, xts, exts, training_order, arima_order, bool_add_ex, log_file ):
    roll_x = xtr
    roll_ex= extr
    
    exdim = len(exts[0])
    xts_hat = []
    
    for i in range(len(xts)):
        
        tmp_x = roll_x[-training_order:]
        tmp_ex = roll_ex[-training_order:]
        
        print 'test on: ', i 
        
        if bool_add_ex == True:
            roll_mod = sm.tsa.statespace.SARIMAX(endog = tmp_x, exog = tmp_ex, order = arima_order)
            
            fit_res = roll_mod.fit(disp=False)
            predict = fit_res.get_forecast(1, exog = np.reshape(exts[i], [1, exdim]))
            predict_ci = predict.conf_int()
            
        else:
            roll_mod = sm.tsa.statespace.SARIMAX(endog = tmp_x, order = arima_order)
            
            fit_res = roll_mod.fit(disp=False)
            predict = fit_res.get_forecast(1)
            predict_ci = predict.conf_int()
            
        xts_hat.append(predict.predicted_mean)
        
        roll_x  = np.concatenate( (roll_x,   xts[i:i+1]) )
        roll_ex = np.concatenate( (roll_ex, exts[i:i+1]) )
        
        with open(log_file, "a") as text_file:
            text_file.write( "%f %f\n"%(xts[i], predict.predicted_mean) )
        
    # return rooted mse    
    return xts_hat, sqrt(mean((xts - np.asarray(xts_hat))*(xts - np.asarray(xts_hat))))

def roll_prediction_strx( xtr, extr, xts, exts, training_order, bool_add_ex ):
    roll_x = xtr
    roll_ex= extr
    
    exdim = len(exts[0])
    xts_hat = []
    
    for i in range(len(xts)):
        
        tmp_x = roll_x[-training_order:]
        tmp_ex = roll_ex[-training_order:]
        
        # --- test 
        tmp_x1  = np.expand_dims(np.asarray(roll_x[-training_order-100:-100]), 0)
        tmp_ex1 = np.asarray(roll_ex[-training_order-100:-100])
        tmp_x   = np.expand_dims(np.asarray(tmp_x), 0)
        tmp_ex  = np.asarray(tmp_ex)
        
        tmp_x = np.concatenate( [tmp_x, tmp_x1], 0 )
        print '!! test shape: ', np.shape(tmp_x)
        # ---
        
        print 'test on: ', i 
        
        if bool_add_ex == True:
            
            roll_mod = sm.tsa.UnobservedComponents(endog = tmp_x, exog = tmp_ex, level= 'local linear trend', trend = True )
            
            fit_res = roll_mod.fit(disp=False)
            predict = fit_res.get_forecast(1, exog = np.reshape(exts[i], [1, exdim]))
            predict_ci = predict.conf_int()
            
        else:
            roll_mod = sm.tsa.UnobservedComponents(endog = tmp_x, level= 'local linear trend', trend = True )
            
            fit_res = roll_mod.fit(disp=False)
            predict = fit_res.get_forecast(1)
            predict_ci = predict.conf_int()
            
        xts_hat.append(predict.predicted_mean)
        
        roll_x  = np.concatenate( (roll_x,   xts[i:i+1]) )
        roll_ex = np.concatenate( (roll_ex, exts[i:i+1]) )
        
    # return rooted mse    
    return xts_hat, sqrt(mean((xts - np.asarray(xts_hat))*(xts - np.asarray(xts_hat))))

def oneshot_prediction_egarch( return_tr, vol_tr, return_ts, vol_ts, method ):
    
    if method == 'garch1':
        model = pf.GARCH(return_tr, p=1, q=1)
        x = model.fit()
        
        tr_sigma2, _, ___ = model._model(model.latent_variables.get_z_values())
        vol_tr_hat = tr_sigma2**0.5
        
    elif method == 'egarch1':
        model = pf.EGARCH(return_tr, p=1, q=1)
        x = model.fit()
    
        tr_sigma2, _, ___ = model._model(model.latent_variables.get_z_values())
        vol_tr_hat = np.exp(tr_sigma2/2.0)

    tmp_pre = np.asarray(model.predict(len(vol_ts)))
    vol_ts_hat = []
    for i in tmp_pre:
        vol_ts_hat.append(i[0])
    
    return vol_ts_hat, vol_tr_hat, sqrt(mean((vol_ts - np.asarray(vol_ts_hat))*(vol_ts - np.asarray(vol_ts_hat)))), \
sqrt(mean((vol_tr[1:] - np.asarray(vol_tr_hat))*(vol_tr[1:] - np.asarray(vol_tr_hat))))
    
def roll_prediction_egarch( return_tr, vol_tr, return_ts, vol_ts, training_order, method ):
    
    roll_x = return_tr
    vol_hat = []
    
    for i in range(len(vol_ts)):
        
        print 'now processing', i
        
        tmp_x = roll_x[-training_order:]
        
        if method == 'garch':
            model = pf.GARCH(tmp_x, p=1, q=1)
            x = model.fit()
            
        elif method == 'egarch':
            model = pf.EGARCH(tmp_x, p=1, q=1)
            x = model.fit()
           
        vol_hat.append(np.asarray(model.predict(1))[0][0])
        roll_x = np.concatenate((roll_x, return_ts[i:i+1]))
        
    # return rooted mse    
    return vol_hat, sqrt(mean((vol_ts - np.asarray(vol_hat))*(vol_ts - np.asarray(vol_hat))))

    
def train_armax_strx( xtrain, extrain, xtest, extest, result_file, ar_order, pred_file ):
    # only rolling methods ouput both training and testing errors
    
    #yhat_ts, rmse = roll_prediction_arimax( xtrain, extrain, xtest, extest, 1026, (16,0,0), False, log_file )
    #yhat_ts, rmse = roll_prediction_arimax( xtrain, extrain, xtest, extest, 1026, (16,0,0), True,  log_file )
    #yhat_ts, rmse = roll_prediction_strx( xtrain, extrain, xtest, extest, 1026, False )
    #yhat_ts, rmse = 
    
    #roll_prediction_strx( xtrain, extrain, xtest, extest, 1026,  False )
    
    '''
    yhat_ts, yhat_tr, ts_rmse, tr_rmse = oneshot_prediction_arimax( xtrain, extrain, xtest, extest, (ar_order,1,0), False )
    with open(result_file, "a") as text_file:
        text_file.write( "%s : %f  %f \n"%("ARIMA", tr_rmse, ts_rmse) ) 
        
    np.savetxt(pred_file + "pytest_arima.txt", zip(xtest, yhat_ts), delimiter=',')  
    np.savetxt(pred_file + "pytrain_arima.txt", zip(xtrain, yhat_tr), delimiter=',')  
    '''    
    
    '''
    yhat_ts, yhat_tr, ts_rmse, tr_rmse = oneshot_prediction_arimax( xtrain, extrain, xtest, extest, (ar_order,1,0), True )
    with open(result_file, "a") as text_file:
        text_file.write( "%s : %f  %f \n"%("ARIMAX", tr_rmse, ts_rmse) ) 
    
    np.savetxt(pred_file + "pytest_arimax.txt", zip(xtest, yhat_ts), delimiter=',')    
    np.savetxt(pred_file + "pytrain_arimax.txt", zip(xtrain, yhat_tr), delimiter=',')  
    print 'ARIMAX : ', tr_rmse, ts_rmse
    '''
    
    '''
    yhat_ts, yhat_tr, ts_rmse, tr_rmse = oneshot_prediction_strx( xtrain, extrain, xtest, extest, False )
    with open(result_file, "a") as text_file:
        text_file.write( "%s : %f  %f \n"%("STR", tr_rmse, ts_rmse) )
        
    np.savetxt(pred_file + "pytest_str.txt", zip(xtest, yhat_ts), delimiter=',')    
    np.savetxt(pred_file + "pytrain_str.txt", zip(xtrain, yhat_tr), delimiter=',')
    '''
    
    yhat_ts, yhat_tr, ts_rmse, tr_rmse = oneshot_prediction_strx( xtrain, extrain, xtest, extest, True )
    #with open(result_file, "a") as text_file:
    #    text_file.write( "%s : %f  %f \n"%("STRX", tr_rmse, ts_rmse) )
    
    np.savetxt(pred_file + "pytest_strx.txt", zip(xtest, yhat_ts), delimiter=',')
    np.savetxt(pred_file + "pytrain_strx.txt", zip(xtrain, yhat_tr), delimiter=',')
    print 'STRX : ', tr_rmse, ts_rmse
    
    return 0

def train_garch( rt_train, xtrain, rt_test, xtest, result_file, pred_file ):
    
    #yhat_ts, rmse = roll_prediction_egarch( rt_train, xtrain, rt_test, xtest, 1000, 'garch' )
    #yhat_ts, rmse = roll_prediction_egarch( rt_train, xtrain, rt_test, xtest, 1000, 'egarch' )
    
    yhat_ts, yhat_tr, ts_rmse, tr_rmse = oneshot_prediction_egarch( rt_train, xtrain, rt_test, xtest, 'garch1' )
    with open(result_file, "a") as text_file:
        text_file.write( "%s : %f  %f \n"%("GARCH", tr_rmse, ts_rmse) )
    
    np.savetxt(pred_file + "pytest_garch.txt", zip(xtest, yhat_ts), delimiter=',')
    np.savetxt(pred_file + "pytrain_garch.txt", zip(xtrain, yhat_tr), delimiter=',')
    
    
    yhat_ts, yhat_tr, ts_rmse, tr_rmse = oneshot_prediction_egarch( rt_train, xtrain, rt_test, xtest, 'egarch1' )
    with open(result_file, "a") as text_file:
        text_file.write( "%s : %f  %f \n"%("EGARCH", tr_rmse, ts_rmse) )
    
    np.savetxt(pred_file + "pytest_egarch.txt", zip(xtest, yhat_ts), delimiter=',')
    np.savetxt(pred_file + "pytrain_egarch.txt", zip(xtrain, yhat_tr), delimiter=',')
    
    return 0
    
        

# --- main process ---

#print 'Number of arguments:', len(sys.argv), 'arguments.'
print '--- Argument List:', str(sys.argv)

# oneshot, roll, incre
train_mode = str(sys.argv[1])

if train_mode == 'oneshot':
    
    # log file address
    result_file = "../bt_results/res/reg_v_minu.txt"
    model_file = "../bt_results/model/v_minu"
    
    # log for predictions
    file_path = "../bt_results/res/"
    
    if method in ['garch', 'garch1', 'egarch', 'egarch1'] :
        
        # --- Load pre-processed training and testing data ---
        file_postfix = "garch"
        xtrain = np.load("../dataset/bitcoin/training_data/voltrain_"+file_postfix+".dat")
        rt_train  = np.load("../dataset/bitcoin/training_data/rttrain_" +file_postfix+".dat")
        xtest = np.load("../dataset/bitcoin/training_data/voltest_"+file_postfix+".dat")
        rt_test  = np.load("../dataset/bitcoin/training_data/rttest_" +file_postfix+".dat")
        
        print np.shape(xtrain), np.shape(rt_train), np.shape(xtest), np.shape(rt_test)
        
    else:
        # --- Load pre-processed training and testing data ---
        file_postfix = "stat"
        xtrain = np.load("../dataset/bitcoin/training_data/xtrain_"+file_postfix+".dat")
        extrain  = np.load("../dataset/bitcoin/training_data/extrain_" +file_postfix+".dat")
        xtest = np.load("../dataset/bitcoin/training_data/xtest_"+file_postfix+".dat")
        extest  = np.load("../dataset/bitcoin/training_data/extest_" +file_postfix+".dat")
        
        print np.shape(xtrain), np.shape(extrain), np.shape(xtest), np.shape(extest)

    # all feature vectors are already normalized
    print '--- Start one-shot training: ', np.shape(xtrain), np.shape(ytrain), np.shape(xtest), np.shape(ytest)
    
    train_eval_models( xtrain, ytrain, xtest, ytest )
    
elif train_mode == 'roll' or 'incre':
    
    # load raw feature data
    features_minu = np.load("../dataset/bitcoin/training_data/feature_minu.dat")
    rvol_hour = np.load("../dataset/bitcoin/training_data/return_vol_hour.dat")
    all_loc_hour = np.load("../dataset/bitcoin/loc_hour.dat")
    
    all_dta_minu = np.load("../dataset/bitcoin/dta_minu.dat")
    price_minu, req_minu = cal_price_req_minu(all_dta_minu)
    
    
    print '--- Start the ' + train_mode + ' training: \n', np.shape(features_minu), np.shape(rvol_hour)
    
    # shift to align 
    rvol_hour = rvol_hour[1:]
    all_loc_hour = all_loc_hour[1:]
    
    # set up the interval parameters
    interval_len = 30*24
    interval_num = int(len(rvol_hour)/interval_len)
    roll_len = 2
    print interval_len, interval_num
    
    for i in range(roll_len + 1, interval_num+1):
        
        # log for predictions in each interval
        pred_file_path = "../bt_results/res/rolling/" + str(i-1) + "_"
        
        print '\n --- In processing of interval ', i-1, ' --- \n'
        with open(result_file, "a") as text_file:
            text_file.write( "Interval %d :\n" %(i-1) )
            
        
        if train_mode == 'roll':
            
            tmp_vol_hour = rvol_hour[(i-roll_len-1)*interval_len: i*interval_len]
            tmp_loc_hour  = all_loc_hour[(i-roll_len-1)*interval_len: i*interval_len]
            para_train_split_ratio = 1.0*(len(tmp_vol_hour) - interval_len)/len(tmp_vol_hour)
            
            
            # -- AR order
            # adaptivly change the auto-regressive order, check PACF order of the current training data
            full_pacf = sm.tsa.stattools.pacf(tmp_vol_hour)
            tmp_pacf = full_pacf[2:]
            pacf_order = list(tmp_pacf).index(max(tmp_pacf))

            tmp_ar_order = 2
            for i in range(len(full_pacf)):
                if full_pacf[i]>0.1:
                    tmp_ar_order = i
            
            if tmp_ar_order < 16:
                tmp_ar_order = 16
                
            print 'PACF order: ', tmp_ar_order, '\n', full_pacf
            # --

            # prepare the data 
            xtrain, extrain, xtest, extest = training_testing_statistic(features_minu, tmp_vol_hour, tmp_loc_hour, \
                                para_order_minu, para_order_hour, para_train_split_ratio, bool_feature_selection)
            
            vol_train, rt_train, vol_test, rt_test = training_testing_garch(tmp_vol_hour, tmp_loc_hour, para_order_hour, \
                                                                para_train_split_ratio, price_minu)
            
            print np.shape(tmp_vol_hour), np.shape(tmp_loc_hour)
            print np.shape(xtrain), np.shape(extrain), np.shape(xtest), np.shape(extest)
            print np.shape(vol_train), np.shape(rt_train), np.shape(vol_test), np.shape(rt_test)
            
            
            # begin to train the model
            train_armax_strx( xtrain, extrain, xtest, extest, result_file, tmp_ar_order, pred_file_path )
            #train_garch( np.asarray(rt_train), np.asarray(vol_train), np.asarray(rt_test), \
            #            np.asarray(vol_test), result_file, pred_file_path )
            
        
        elif train_mode == 'incre':
            
            tmp_vol_hour = rvol_hour[ : i*interval_len]
            tmp_loc_hour  = all_loc_hour[ : i*interval_len]
            para_train_split_ratio = 1.0*(len(tmp_vol_hour) - interval_len)/len(tmp_vol_hour)
            
            
            # -- AR order
            # adaptivly change the auto-regressive order, check PACF order of the current training data
            full_pacf = sm.tsa.stattools.pacf(tmp_vol_hour)
            tmp_pacf = full_pacf[2:]
            pacf_order = list(tmp_pacf).index(max(tmp_pacf))

            tmp_ar_order = 2
            for i in range(len(full_pacf)):
                if full_pacf[i]>0.1:
                    tmp_ar_order = i
            
            if tmp_ar_order < 16:
                tmp_ar_order = 16
                
            print 'PACF order: ', tmp_ar_order, '\n', full_pacf
            # --
            
            # prepare the data
            xtrain, extrain, xtest, extest = training_testing_statistic(features_minu, tmp_vol_hour, tmp_loc_hour, \
                                para_order_minu, para_order_hour, para_train_split_ratio, bool_feature_selection)
            
            vol_train, rt_train, vol_test, rt_test = training_testing_garch(tmp_vol_hour, tmp_loc_hour, para_order_hour, \
                                                                para_train_split_ratio, price_minu)
            
            
            print np.shape(tmp_vol_hour), np.shape(tmp_loc_hour)
            print np.shape(xtrain), np.shape(extrain), np.shape(xtest), np.shape(extest)
            print np.shape(vol_train), np.shape(rt_train), np.shape(vol_test), np.shape(rt_test)
            
            # begin to train the model
            train_armax_strx( xtrain, extrain, xtest, extest, result_file, tmp_ar_order, pred_file_path )
            train_garch( np.asarray(rt_train), np.asarray(vol_train), np.asarray(rt_test), \
                        np.asarray(vol_test), result_file, pred_file_path )
            
            
        else:
            print '[ERROR] training mode'
        
        
else:
    print '[ERROR] training mode'

    
    
    