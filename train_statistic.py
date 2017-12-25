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

# ---- model and training log set-up ----
result_file = "res/reg_v_minu.txt"
log_file = "res/arima"
model_file = "model/v_minu"

def oneshot_prediction_arimax( xtr, extr, xts, exts, training_order, arima_order, bool_add_ex ):
    roll_x = xtr
    roll_ex= extr
    
    exdim = len(exts[0])
    xts_hat = []
        
    if bool_add_ex == True:
        mod = sm.tsa.statespace.SARIMAX(endog = xtr, exog = extr, order = arima_order)
            
        fit_res = mod.fit(disp=False)
        predict = fit_res.get_forecast( len(xts), exog = exts )
        predict_ci = predict.conf_int()
    
    else:
        mod = sm.tsa.statespace.SARIMAX(endog = xtr, order = arima_order)
            
        fit_res = mod.fit(disp=False)
        predict = fit_res.get_forecast( len(xts) )
        predict_ci = predict.conf_int()
        
    tr_predict = fit_res.get_prediction()
    tr_predict_ci = predict.conf_int()
    
    xtr_hat = tr_predict.predicted_mean 
    xts_hat = predict.predicted_mean
    
    # out-sample forecast, in-sample rmse, out-sample rmse      
    return xts_hat, sqrt(mean((xts - np.asarray(xts_hat))*(xts - np.asarray(xts_hat)))), \
sqrt(mean((xtr - np.asarray(xtr_hat))*(xtr - np.asarray(xtr_hat))))

def oneshot_prediction_strx( xtr, extr, xts, exts, training_order, bool_add_ex ):
    roll_x = xtr
    roll_ex= extr
    
    exdim = len(exts[0])
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
    return xts_hat, sqrt(mean((xts - np.asarray(xts_hat))*(xts - np.asarray(xts_hat)))), \
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

def oneshot_prediction_egarch( return_tr, vol_tr, return_ts, vol_ts, training_order, method ):
    
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
    
    return vol_ts_hat, sqrt(mean((vol_ts - np.asarray(vol_ts_hat))*(vol_ts - np.asarray(vol_ts_hat)))), \
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

# ---- main process ----

#print 'Number of arguments:', len(sys.argv), 'arguments.'
print '--- Argument List:', str(sys.argv)
method = str(sys.argv[1]) 
run_mode = str(sys.argv[2])

if run_mode == 'local':
    
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
    
elif run_mode == 'server':
    
    # --- Load pre-processed training and testing data ---
    file_postfix = "stat"
    xtrain = np.load("../dataset/bk/xtrain_"+file_postfix+".dat")
    extrain  = np.load("../dataset/bk/extrain_" +file_postfix+".dat")
    xtest = np.load("../dataset/bk/xtest_"+file_postfix+".dat")
    extest  = np.load("../dataset/bk/extest_" +file_postfix+".dat")

    print np.shape(xtrain), np.shape(extrain), np.shape(xtest), np.shape(extest)

# only rolling methods ouput both training and testing errors
if method == 'arima':
    log_file = log_file +'.txt'
    
    # initialize the log
    with open(log_file, "w") as text_file:
        text_file.close()
        
    yhat_ts, rmse = roll_prediction_arimax( xtrain, extrain, xtest, extest, 1026, (16,0,0), False, log_file )
    
elif method == 'arimax':
    log_file = log_file +'x.txt'
    
    # initialize the log
    with open(log_file, "w") as text_file:
        text_file.close()
    
    yhat_ts, rmse = roll_prediction_arimax( xtrain, extrain, xtest, extest, 1026, (16,0,0), True,  log_file )
    
elif method == 'str':
    yhat_ts, rmse = roll_prediction_strx( xtrain, extrain, xtest, extest, 1026, False )
    
elif method == 'strx':
    yhat_ts, rmse = roll_prediction_strx( xtrain, extrain, xtest, extest, 1026,  True )
    
elif method == 'arima1':
    yhat_ts, ts_rmse, tr_rmse = oneshot_prediction_arimax( xtrain, extrain, xtest, extest, 1026, (16,0,0), False )
    
elif method == 'arimax1':
    yhat_ts, ts_rmse, tr_rmse = oneshot_prediction_arimax( xtrain, extrain, xtest, extest, 1026, (16,0,0), True )
    
elif method == 'str1':
    yhat_ts, ts_rmse, tr_rmse = oneshot_prediction_strx( xtrain, extrain, xtest, extest, 1026, False )
    
elif method == 'strx1':
    yhat_ts, ts_rmse, tr_rmse = oneshot_prediction_strx( xtrain, extrain, xtest, extest, 1026, True )
    
elif method == 'garch1' or method == 'egarch1' :
    
    print 'using one-shot garch model: '
    yhat_ts, ts_rmse, tr_rmse = oneshot_prediction_egarch( rt_train, xtrain, rt_test, xtest, 1000, method )

elif method == 'garch' or method == 'egarch':
    
    print 'using garch model: '
    yhat_ts, rmse = roll_prediction_egarch( rt_train, xtrain, rt_test, xtest, 1000, method )
    
    
# ---- record traning and testing errors ----    

np.savetxt("res/pytest_" + method + ".txt", zip(xtest, yhat_ts), delimiter=',')

if method in ['arima', 'arimax', 'str', 'strx', 'garch', 'egarch']:
    
    print 'RMSE : ', rmse
    
    with open(result_file, "a") as text_file:
        text_file.write( "%s : %f  \n"%(method, rmse) ) 
        
else:
    
    print 'RMSE : ', tr_rmse, ts_rmse
    
    with open(result_file, "a") as text_file:
        text_file.write( "%s : %f  %f \n"%(method, tr_rmse, ts_rmse) ) 
    
    
    