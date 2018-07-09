import numpy as np   
import pandas as pd 

from pandas import *
from numpy import *
from scipy import *
 
import random

# machine leanring packages
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.ensemble import *
#GradientBoostingRegressor
from sklearn.ensemble import *
#RandomForestRegressor
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.externals import joblib

import xgboost as xgb

# ++++ GBT ++++

# https://www.analyticsvidhya.com/blog/2016/02/
# complete-guide-parameter-tuning-gradient-boosting-gbm-python/

#----Boosting parameters:
#   learnning rate: 0.05 - 0.2
#   n_estimators: 40-70

#----Tree parameters:
#   max_depth: 3-10
#   max_leaf_nodes
#   num_samples_split: 0.5-1% of total number 
#   min_samples_leaf
#   max_features

#   subsample: 0.8
#   min_weight_fraction_leaf

#----Order of tuning: max_depth and num_samples_split, min_samples_leaf, max_features

def utils_result_comparison(res1, res2, bool_clf):
    # clf: accuracy, regression: error
    if bool_clf:
        return True if res1>res2 else False
    else:
        return True if res1<res2 else False
    
def utils_evaluation_score(x, y, bool_clf, model):
    # clf: accuracy, regression: rmse
    if bool_clf:
        return model.score(x,y) 
    else:
        y_hat  = model.predict(x)
        return sqrt(sum((y_hat-y)*(y_hat-y))/len(y))
    
def utils_evaluation_full_score(x, y, bool_clf, model):
    # clf: accuracy, regression: rmse, mae, mape 
    if bool_clf:
        return model.score(x,y) 
    else:
        y_hat = model.predict(x)
        return [ sqrt(mean((y_hat-y)**2)), mean(abs(y_hat-y)), mean(abs(y_hat-y)/(y+1e-10)) ]      
    
    
            
def gbt_n_estimatior(maxnum, X, Y, xtest, ytest, fix_lr, bool_clf ):
    
    tmpy = Y.reshape( (len(Y),) )
    score = []
    
    tmp_err = 0.0 if bool_clf else np.inf 
    
    for i in range(10,maxnum+1,10):
        
        if bool_clf == False:
            clf = GradientBoostingRegressor(n_estimators = i,learning_rate = fix_lr, max_depth=3, max_features ='sqrt')
        else:
            clf = GradientBoostingClassifier(n_estimators = i,learning_rate = fix_lr, max_depth=3, max_features ='sqrt')

        clf.fit( X, tmpy )
        pytest  = clf.predict(xtest)
        
        if bool_clf == False:
            tmp_ts = sqrt(sum((pytest-ytest)*(pytest-ytest))/len(ytest))
            score.append( (i, tmp_ts) )
            
            if tmp_ts<tmp_err:
                best_pytest = pytest
                best_model  = clf
                
                tmp_err = tmp_ts
        else:
            tmp_ts = clf.score(xtest, ytest)
            score.append( (i, tmp_ts) )
            
            if tmp_ts>tmp_err:
                best_pytest = pytest
                best_model  = clf
                
                tmp_err = tmp_ts
    
    return min(score, key = lambda x: x[1]) if bool_clf == False else max(score, key = lambda x: x[1]) ,\ 
           best_model, utils_evaluation_score(X, Y, bool_clf, best_model) 

def gbt_tree_para( X, Y, xtest, ytest, depth_range, fix_lr, fix_n_est, bool_clf ):
    
    tmpy = Y.reshape( (len(Y),) )
    score = []
    
    tmp_err = 0.0 if bool_clf else np.inf 
    
    for i in depth_range:
        
        if bool_clf == False:
            clf=GradientBoostingRegressor(n_estimators = fix_n_est, learning_rate = fix_lr,max_depth = i, max_features ='sqrt')
        else:
            clf=GradientBoostingClassifier(n_estimators = fix_n_est,learning_rate = fix_lr,max_depth = i, max_features ='sqrt')
            
        clf.fit( X, tmpy )
        pytest = clf.predict(xtest)
        
        # regression
        if bool_clf == False:
            
            tmp_ts = sqrt(sum((pytest-ytest)*(pytest-ytest))/len(ytest))
            score.append( (i, tmp_ts) )
            
            if tmp_ts<tmp_err:
                best_pytest = pytest
                best_model  = clf
                
                tmp_err = tmp_ts
                
        # classification
        else:
            tmp_ts = clf.score(xtest, ytest)
            score.append( (i, tmp_ts) )
            
            if tmp_ts>tmp_err:
                best_pytest = pytest
                best_model  = clf
                
                tmp_err = tmp_ts
    
    return min(score, key = lambda x: x[1]) if bool_clf == False else max(score, key = lambda x: x[1]),\
           best_model, utils_evaluation_score(X, Y, bool_clf, best_model) 
        
def gbt_train_validate(xtrain, ytrain, xval, yval, xtest, ytest, fix_lr, bool_clf, result_file, model_file, pred_file):
    
    print "\nStart to train GBT"

    fix_lr = 0.25

    n_err, model0, train_err0 = gbt_n_estimatior( 250, xtrain, ytrain, xval, yval, fix_lr, bool_clf)
    print "n_estimator, RMSE:", train_err0, n_err

    depth_err, model1, train_err1 = gbt_tree_para( xtrain, ytrain, xval, yval, range(3,16), fix_lr, n_err[0], bool_clf )
    print "depth, RMSE:", train_err1, depth_err
    
    
    # training performance
    best_train = min(train_err0, train_err1) if bool_clf == False else max(train_err0, train_err1)  
    
    # fix the model with the best validation performance
    if utils_result_comparison(n_err[1], depth_err[1], bool_clf):
        
        joblib.dump(model0, model_file)
        
        best_model = model0
        best_vali = n_err[1]
        
        # result_tuple [ # iteration, depth, train error, validation error, test error ]
        # hyper-parameters: depth = 3 has the best validation error
        result_tuple = [ n_err[0], 3, best_train, best_vali ]
        
        
    else:
        
        joblib.dump(model1, model_file)
        
        best_model = model1
        best_vali = depth_err[1]
        
        # hyper-parameters with the best validation error 
        result_tuple = [ n_err[0], depth_err[0], best_train, best_vali ]
        
        
    # save training prediction under the best model
    py = best_model.predict( xtrain )
    np.savetxt(pred_file + "pytrain_gbt.txt", zip(ytrain, py), delimiter=',')
    
    # save testing prediction under the best model
    if len(xtest)!=0:
        py = best_model.predict( xtest )
        np.savetxt(pred_file + "pytest_gbt.txt", zip(ytest, py), delimiter=',')
            
        result_tuple.append( utils_evaluation_score(xtest, ytest, bool_clf, best_model) )
    
    # save validation prediction under the best model
    else:
        py = best_model.predict( xval )
        np.savetxt(pred_file + "pytest_gbt.txt", zip(yval, py), delimiter=',')
            
        result_tuple.append( None )
    
    # log overall errors 
    with open(result_file, "a") as text_file:
        text_file.write( "GBT: %s \n" %(str(result_tuple)) )
    
    return result_tuple
    
    # load the model from disk
    #loaded_model = joblib.load(filename)
    #result = loaded_model.score(X_test, Y_test)
    #print(result)
    
    
    
# ++++ XGBoosted ++++

# https://www.analyticsvidhya.com/blog/2016/03/
#     complete-guide-parameter-tuning-xgboost-with-codes-python/

#----General Parameters

#   eta(learning rate): 0.05 - 0.3
#   number of rounds: 

#----Booster Parameters

#   max_depth 3-10
#   max_leaf_nodes
#   gamma: mininum loss reduction
#   min_child_weight: 1 by default

#   max_delta_step: not needed in general, for unbalance in logistic regression
#   subsample: 0.5-1
#   colsample_bytree: 0.5-1
#   colsample_bylevel: 

#   lambda: l2 regularization 
#   alpha: l1 regularization
#   scale_pos_weight: >>1, for high class imbalance

# Learning Task Parameters
def xgt_evaluation_score(xg_tuple, y, bool_clf, model):
    # clf: accuracy, regression: error
    pred = model.predict( xg_tuple )
    
    if bool_clf == True:
        tmplen = len(y)
        tmpcnt = 0.0
        for i in range(tmplen):
            if y[i] == pred[i]:
                tmpcnt += 1.0
        return tmpcnt*1.0/tmplen
                
    else:
        return sqrt(mean( [(pred[i] - y[i])**2 for i in range(len(y))] )) 
    

def xgt_n_depth( lr, max_depth, max_round, xtrain, ytrain, xtest, ytest, bool_clf, num_class ):
    
    score = []
    xg_train = xgb.DMatrix(xtrain, label = ytrain)
    xg_test  = xgb.DMatrix(xtest,  label = ytest)

# setup parameters for xgboost
    param = {}
# use softmax multi-class classification

    if bool_clf == True:
        param['objective'] = 'multi:softmax'
        param['num_class'] = num_class
    else:
        param['objective'] = "reg:linear" 
#   'multi:softmax'
    
# scale weight of positive examples
    #   param['gamma']
    param['eta'] = lr
    param['max_depth'] = 0
    param['silent'] = 1
    param['nthread'] = 8
    
    tmp_err = 0.0 if bool_clf else np.inf 
    
    for depth_trial in range(2, max_depth):
        
        for num_round_trial in range(2, max_round):

            param['max_depth'] = depth_trial
            bst  = xgb.train( param, xg_train, num_round_trial )
            pred = bst.predict( xg_test )
            
            if bool_clf == True:
                tmplen = len(ytest)
                tmpcnt = 0.0
                for i in range(tmplen):
                    if ytest[i] == pred[i]:
                        tmpcnt +=1
                tmp_accur = tmpcnt*1.0/tmplen
                
                if tmp_accur > tmp_err:
                    best_model = bst
                    best_pytest = pred
                    
                    tmp_err = tmp_accur
            else:
                tmp_accur = sqrt(mean( [(pred[i] - ytest[i])**2 for i in range(len(ytest))] )) 
                
                if tmp_accur < tmp_err:
                    best_model = bst
                    best_pytest = pred
                    
                    tmp_err = tmp_accur
            
            score.append( (depth_trial, num_round_trial, tmp_accur) )
            
    return min(score, key = lambda x: x[2]) if bool_clf == False else max(score, key = lambda x: x[2]),\
           best_model, xgt_evaluation_score(xg_train, ytrain, bool_clf, best_model) 


def xgt_l2( fix_lr, fix_depth, fix_round, xtrain, ytrain, xtest, ytest, l2_range, bool_clf, num_class ):
    
    score = []
    xg_train = xgb.DMatrix(xtrain, label = ytrain)
    xg_test  = xgb.DMatrix(xtest,  label = ytest)

# setup parameters for xgboost
    param = {}
# use softmax multi-class classification
    if bool_clf == True:
        param['objective'] = 'multi:softmax'
        param['num_class'] = num_class
    else:
        param['objective'] = "reg:linear" 

# scale weight of positive examples
    param['eta'] = fix_lr
    param['max_depth'] = fix_depth
    param['silent'] = 1
    param['nthread'] = 8
    
    param['lambda'] = 0.0
#     param['alpha']
    
    tmp_err = 0.0 if bool_clf else np.inf 
    
    for l2_trial in l2_range:
        
        param['lambda'] = l2_trial
        
        bst = xgb.train(param, xg_train, fix_round )
        pred = bst.predict( xg_test )
        
        if bool_clf == True:
            tmplen = len(ytest)
            tmpcnt = 0.0
            for i in range(tmplen):
                if ytest[i] == pred[i]:
                    tmpcnt += 1
            tmp_accur = tmpcnt*1.0/tmplen
            
            if tmp_accur > tmp_err:
                best_model = bst
                best_pytest = pred
                    
                tmp_err = tmp_accur
        else:
            tmp_accur = sqrt(mean( [(pred[i] - ytest[i])**2 for i in range(len(ytest))] ))
            
            if tmp_accur < tmp_err:
                best_model = bst
                best_pytest = pred
                    
                tmp_err = tmp_accur
                    
        score.append( (l2_trial, tmp_accur) )
            
    return min(score, key = lambda x: x[1]) if bool_clf == False else max(score, key = lambda x: x[1]),\
           best_model, xgt_evaluation_score(xg_train, ytrain, bool_clf, best_model)
    
    
def xgt_train_validate(xtrain, ytrain, xval, yval, xtest, ytest, bool_clf, num_class, result_file, model_file, pred_file):
    
    print "\nStart to train XGBoosted"
    
    fix_lr = 0.2

    n_depth_err, model0, train_err0 = xgt_n_depth( fix_lr, 16, 41, xtrain, ytrain, xval, yval, bool_clf, num_class)
    print " depth, number of rounds, RMSE:", train_err0, n_depth_err

    l2_err, model1, train_err1 = xgt_l2( fix_lr, n_depth_err[0], n_depth_err[1], xtrain, ytrain, xval, yval,\
                    [0.0001, 0.001, 0.01, 0.1, 1, 10, 100], bool_clf, num_class)
    print " l2, RMSE:", train_err1, l2_err


    # specific for XGBoosted
    xg_train = xgb.DMatrix(xtrain, label = ytrain)
    
    
    # training performance
    best_train = min(train_err0, train_err1) if bool_clf == False else max(train_err0, train_err1)  
    
    # fix the model with the best validation performance
    if utils_result_comparison(n_depth_err[2], l2_err[1], bool_clf):
        
        joblib.dump(model0, model_file)
        
        # result_tuple [ # iteration, depth, l2, train error, validation error, test error ]
        result_tuple = [ n_depth_err[0], n_depth_err[1], 0, best_train, best_vali ]
        best_model = model0
        best_vali = n_depth_err[2]
        
    else:
        joblib.dump(model1, model_file)
        
        result_tuple = [ n_depth_err[0], n_depth_err[1], l2_err[0], best_train, best_vali ]
        best_model = model1
        best_vali = l2_err[1]
        
    # save training prediction under the best model
    py = best_model.predict( xg_train )
    np.savetxt(pred_file + "pytrain_xgt.txt", zip(ytrain, py), delimiter=',')
    
    
    # save testing prediction under the best model
    if len(xtest)!=0:
        xg_test  = xgb.DMatrix(xtest, label = ytest)
        
        py = best_model.predict( xg_test )
        np.savetxt(pred_file + "pytest_xgt.txt", zip(ytest, py), delimiter=',')
            
        result_tuple.append( xgt_evaluation_score(xg_test, ytest, bool_clf, best_model) )
    
    # save validation prediction under the best model
    else:
        xg_val = xgb.DMatrix(xval, label = yval)
        
        py = best_model.predict( xg_val )
        np.savetxt(pred_file + "pytest_xgt.txt", zip(yval, py), delimiter=',')
            
        result_tuple.append( None )
    
    # log overall errors 
    with open(result_file, "a") as text_file:
        text_file.write( "XGT: %s \n" %(str(result_tuple)) )
    
    return result_tuple
    
# TO DO: def xgt_l1 for very high dimensional features    
    
    
# ++++ Random forest ++++

#https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/

# max_features:
# n_estimators
# max_depth

def rf_n_depth_estimatior(maxnum, maxdep, X, Y, xtest, ytest, bool_clf):
        
    tmpy = Y
    score = []
        
    tmp_err = 0.0 if bool_clf else np.inf 

    for n_trial in range(10,maxnum+1,10):
        for dep_trial in range(2, maxdep+1):
            
            if bool_clf == True:
                clf = RandomForestClassifier(n_estimators = n_trial, max_depth = dep_trial, max_features = "sqrt")
            else:
                clf = RandomForestRegressor(n_estimators = n_trial, max_depth = dep_trial, max_features = "sqrt")
            
            clf.fit( X, tmpy )
            pytest = clf.predict(xtest)
            
            if bool_clf == False:
                tmp_ts = sqrt(sum((pytest-ytest)*(pytest-ytest))/len(ytest))
                score.append((n_trial, dep_trial, tmp_ts )) 
                
                if tmp_ts<tmp_err:
                    best_pytest = pytest
                    best_model  = clf
                
                    tmp_err = tmp_ts
                
            else:
                tmp_ts = clf.score(xtest, ytest)
                score.append( (n_trial, dep_trial, tmp_ts ) )
            
                if tmp_ts>tmp_err:
                    best_pytest = pytest
                    best_model  = clf
                
                    tmp_err = tmp_ts
                                
    return min(score, key = lambda x: x[2]) if bool_clf==False else max(score, key = lambda x: x[2]),\
           best_pytest, best_model, utils_evaluation_score(X, Y, bool_clf, best_model)


def rf_train_validate(xtrain, ytrain, xval, yval, xtest, ytest, bool_clf, result_file, model_file, pred_file):
    
    print "\nStart to train Random Forest"

    n_err, y_hat, best_model, train_err = rf_n_depth_estimatior( 100, 20, xtrain, ytrain, xval, yval, bool_clf )
    
    
    # save the best model
    joblib.dump(best_model, model_file)
    
    # [tree number, tree depth, train error, validation error, test error]
    result_tuple = [ n_err[0], n_err[1], train_err, n_err[-1] ]
    
    # save training prediction under the best model
    py = best_model.predict( xtrain )
    np.savetxt(pred_file + "pytrain_rf.txt", zip(ytrain, py), delimiter=',')
    
    # save testing or validation prediction under the best model
    if len(xtest)!=0:
        py = model.predict( xtest )
        np.savetxt(pred_file + "pytest_rf.txt", zip(ytest, py), delimiter=',')
        
        result_tuple.append( utils_evaluation_score(xtest, ytest, bool_clf, best_model) )
    
    else:
        py = model.predict( xval )
        np.savetxt(pred_file + "pytest_rf.txt", zip(yval, py), delimiter=',')
        
        result_tuple.append( None )
    
    
    print "number trees, depth, RMSE:", result_tuple
    
    # log overall errors
    with open(result_file, "a") as text_file:
        text_file.write( "Random Forest: %s \n" %(str(result_tuple)) )
    
    # return the result tuple 
    return result_tuple

# ++++ ElasticNet ++++

from sklearn.linear_model import ElasticNet

def enet_alpha_l1(alpha_range, l1_range, xtrain, ytrain, xtest, ytest, orig_ytrain):
    
    res = []
    tmp_err = np.inf
    
    for i in alpha_range:
        for j in l1_range:
            
            enet = ElasticNet(alpha = i, l1_ratio = j, normalize  = True, fit_intercept = True )
            enet.fit(xtrain, ytrain)
            
            pytrain = enet.predict( xtrain ) 
            pytest  = enet.predict( xtest )
            
            if len(orig_ytrain)==0:
                tmp_tr = sqrt(mean((pytrain-ytrain)*(pytrain-ytrain)))
                tmp_ts = sqrt(mean((pytest-ytest)*(pytest-ytest)))
            else:
                tmp_tr = sqrt(mean((exp(pytrain)-orig_ytrain)*(exp(pytrain)-orig_ytrain)))
                tmp_ts = sqrt(mean((exp(pytest)-ytest)*(exp(pytest)-ytest)))
            
            res.append( (i, j, tmp_ts) )
            
            if tmp_ts<tmp_err:
                best_model  = enet
                best_pytest = pytest
                tmp_err = tmp_ts
            
    
    py_train = best_model..predict( xtrain ) 
    train_err = sqrt(mean((pytrain-ytrain)*(pytrain-ytrain)))
    
    return min(res, key = lambda x:x[3]), train_err, best_model


def elastic_net_train_validate(xtrain, ytrain, xval, yval, xtest, ytest, result_file, model_file, trans_ytrain, pred_file):
    print "\nStart to train ElasticNet"
    
    # training and validation phase
    alpha_range = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    l1_range = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    
    # train_err is derived with the model under the best validation performance
    if len(trans_ytrain) != 0:
        err_min, err_list, train_err, best_model = enet_alpha_l1( alpha_range, l1_range, xtrain, trans_ytrain, xval, yval,\
                                                                 ytrain)
    else:
        err_min, err_list, train_err, best_model = enet_alpha_l1( alpha_range, l1_range, xtrain, ytrain, xval, yval, [])
    
    # save the best model
    joblib.dump(best_model, model_file)
    
    # [l2, l1, train error, validation error, test error]
    result_tuple = [ err_min[0], err_min[1], train_err, n_err[-1] ]
    
    # save training resutls under the best model
    py = model.predict( xtrain )
    np.savetxt(pred_file + "pytrain_enet.txt", zip(ytrain, py), delimiter=',')
    
    
    # save testing prediction under the best model  
    if len(xtest)!=0:
        pytest  = model.predict( xtest )
        np.savetxt(pred_file + "pytest_enet.txt", zip(ytest, py), delimiter=',')
        
        result_tuple.append( sqrt( mean((pytest - ytest)*(pytest - ytest)) ) )
    
    # save validation prediction under the best model     
    else:
        pytest  = model.predict( xval )
        np.savetxt(pred_file + "pytest_enet.txt", zip(yval, py), delimiter=',')
        
        result_tuple.append( None )
    
    # output
    print "ENET RMSE:", result_tuple
    
    # save overall errors
    with open(result_file, "a") as text_file:
        text_file.write( "Elastic Net %s \n"%( str(err_min)) )
    
    # return the least validation error 
    return result_tuple

    
# ++++ Bayesian regression ++++

# BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,
#        fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300,
#        normalize=False, tol=0.001, verbose=False)

from sklearn import linear_model

def bayesian_reg_train_validate(xtrain, ytrain, xval, yval, xtest, ytest, result_file, model_file, trans_ytrain, pred_file):
    
    print "\nStart to train Bayesian Regression"
    
    if len(trans_ytrain) ==0:
        bayesian_reg = linear_model.BayesianRidge( normalize=True, fit_intercept=True )
        bayesian_reg.fit(xtrain, ytrain)
    else:
        bayesian_reg = linear_model.BayesianRidge( normalize=True, fit_intercept=True )
        bayesian_reg.fit(xtrain, trans_ytrain)
    
    pytrain = bayesian_reg.predict( xtrain )
    pyval = bayesian_reg.predict( xval )
    
    if len(trans_ytrain) ==0:
        tmp_tr = sqrt(mean((pytrain-ytrain)*(pytrain-ytrain)))
        tmp_val = sqrt(mean((pyval-yval)**2))
    else:
        tmp_tr = sqrt(mean((exp(pytrain)-ytrain)*(exp(pytrain)-ytrain)))
        tmp_val = sqrt(mean((exp(pyval)-yval)**2))
    
    print "Bayes RMSE: ", tmp_tr, tmp_val 
    
    # [train error, validation error, test error]
    result_tuple = [ tmp_tr, tmp_val ]
    
    if len(xtest) != 0:
        py = bayesian_reg.predict( xtest )
        
        # save testing prediction under the best model
        if len(trans_ytrain) ==0:
            result_tuple.append( sqrt(mean((py-ytest)**2)) )
            np.savetxt(pred_file + "pytest_bayes.txt", zip(ytest, py), delimiter=',')
            
        else:
            result_tuple.append( sqrt(mean((exp(py)-ytest)**2)) )
            np.savetxt(pred_file + "pytest_bayes.txt", zip(ytest, exp(py)), delimiter=',')
            
    else:
        result_tuple.append(None)
        
        if len(trans_ytrain) ==0:
            np.savetxt(pred_file + "pytest_bayes.txt", zip(yval, py), delimiter=',')
        else:
            np.savetxt(pred_file + "pytest_bayes.txt", zip(yval, exp(py)), delimiter=',')
        
    
    # save training prediction under the best model
    py = bayesian_reg.predict( xtrain )
    if len(trans_ytrain) ==0:
        np.savetxt(pred_file + "pytrain_bayes.txt", zip(ytrain, py), delimiter=',')
    else:
        np.savetxt(pred_file + "pytrain_bayes.txt", zip(ytrain, exp(py)), delimiter=',')
    
    
    # save the best model
    joblib.dump(bayesian_reg, model_file)
    
    # outpout
    print "Bayes RMSE: ", result_tuple
    
    # log the overall errors
    with open(result_file, "a") as text_file:
        text_file.write( "Bayeisan regression: %s \n"%(str(result_tuple)) )
    
    # return the lowest validation error 
    return result_tuple
    
        
# ++++ Ridge regression ++++

from sklearn.linear_model import Ridge

def ridge_reg_train_validate(xtrain, ytrain, xval, yval, xtest, ytest, result_file, model_file, trans_ytrain, pred_file):
    
    print "\nStart to train Ridge Regression"
    
    tmp_range = [0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2]
    tmp_err = []
    
    best_err = np.inf
    best_l2 = 0.0
   
    for alpha_trial in tmp_range:
        
        if len(trans_ytrain) ==0:
            clf = Ridge(alpha = alpha_trial, fit_intercept = True, normalize= True)
            clf.fit(xtrain, ytrain)
        else:
            clf = Ridge(alpha = alpha_trial, fit_intercept = True, normalize= True)
            clf.fit(xtrain, trans_ytrain)
        
        py = clf.predict( xval )
        
        if len(trans_ytrain) ==0:
            tmp_val = sqrt(mean((py-yval)**2))
        else:
            tmp_val = sqrt(mean((exp(py)-yval)**2))
        
        if tmp_val < best_err:
            
            best_err = tmp_val
            best_model = clf
            best_l2 = alpha_trial
    
    # save training resutls under the best model
    py = best_model.predict( xtrain )
    
    if len(trans_ytrain) ==0:
        train_err = sqrt(mean((py-ytrain)**2))
        np.savetxt(pred_file + "pytrain_ridge.txt", zip(ytrain, py), delimiter=',')
    else:
        train_err = sqrt(mean((exp(py)-ytrain)**2))
        np.savetxt(pred_file + "pytrain_ridge.txt", zip(ytrain, exp(py)), delimiter=',')
    
    # [ l2, train error, validation error, test error]
    result_tuple = [ best_l2, train_err, best_err ]
    
    # save testing resutls under the best model
    if len(xtest) != 0:
        
        py = best_model.predict( xtest )
        if len(trans_ytrain) ==0:
            result_tuple.append( sqrt(mean((py-ytest)*(py-ytest))) )
            np.savetxt(pred_file + "pytest_ridge.txt", zip(ytest, py), delimiter=',')
        
        else:
            result_tuple.append( sqrt(mean((exp(py)-ytest)*(exp(py)-ytest))) )
            np.savetxt(pred_file + "pytest_ridge.txt", zip(ytest, exp(py)), delimiter=',')
        
    else:
        py = best_model.predict( xval )
        
        if len(trans_ytrain) ==0: 
            np.savetxt(pred_file + "pytest_ridge.txt", zip(yval, py), delimiter=',')
        else:
            np.savetxt(pred_file + "pytest_ridge.txt", zip(yval, exp(py)), delimiter=',')
            
        result_tuple.append( None )
        
    
    # save best model 
    joblib.dump(best_model, model_file)
    
    # output 
    print "Ridge RMSE: ", result_tuple
    
    # log the overall errors
    with open(result_file, "a") as text_file:
        text_file.write( "Ridge regression: %s \n"%( str(result_tuple) ) )
    
    # return the least validation error 
    return result_tuple
    
# ++++ gaussian process ++++

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *

def gp_train_validate(xtrain, ytrain, xval, yval, xtest, ytest, result_file, model_file, trans_ytrain, pred_file):
    
    print "\nStart to train Gaussian process"
    
    # specify dimension-specific length-scale parameter
    kernel = 1.0*RBF(length_scale=41.8) + 1.0*RBF(length_scale=180) * ExpSineSquared(length_scale=1.44, periodicity=1) \
+ RationalQuadratic(alpha=17.7, length_scale=0.957) +  RBF(length_scale=0.138) + WhiteKernel(noise_level=0.0336)
    
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, optimizer = 'fmin_l_bfgs_b', normalize_y = True )
    
    gp.fit(xtrain, ytrain)
    # if fit() method is not used, GP uses the parameters as arguments to do prediction 
    # gp.kernel_, gp.alpha_
    
    print "Begin to evaluate Gaussian process regression"
    
    #pytrain, sigma_train = gp.predict(xtrain, return_std=True)
    #tmp_tr = sqrt(mean((pytrain-ytrain)*(pytrain-ytrain)))
    pytrain = -1
    tmp_tr = -1
    
    py, sigma_test = gp.predict(xtrain, return_std=True)
    tmp_tr = sqrt(mean((py-ytrain)**2))
    
    # save training prediction under the best model
    np.savetxt(pred_file + "pytrain_gp.txt", zip(ytrain, py), delimiter=',')
    
    py_val, sigma_test = gp.predict(xval, return_std=True)
    tmp_val = sqrt(mean((py-yval)**2))
    
    # [ train error, validation error, test error ]
    result_tuple = [ tmp_tr, tmp_val ]
    
    # save testing prediction under the best model
    if len(xtest)!=0:
        py, sigma_test = gp.predict(xtest, return_std=True)
        np.savetxt(pred_file + "pytest_gp.txt", zip(ytest, py), delimiter=',')
        
        result_tuple.append( sqrt(mean((py-ytest)**2)) )
        
    else:
        np.savetxt(pred_file + "pytest_gp.txt", zip(yval, py_val), delimiter=',')
        result_tuple.append( None )
    
    # output
    print "GP RMSE: ", result_tuple
    
    # save best model 
    joblib.dump(gp, model_file)
    
    # log the overall errors
    with open(result_file, "a") as text_file:
        text_file.write( "Gaussian process regression: %s \n"%(str(result_tuple)) )
    
    return result_tuple

# ++++ lasso regression ++++

from sklearn import linear_model

def lasso_train_validate(xtrain, ytrain, xval, yval, xtest, ytest, result_file, model_file, trans_ytrain, pred_file):
    
    print "\nStart to train Lasso regression"
    
    tmp_range = [0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 4, 6]
    tmp_err = []
       
    for alpha_trial in tmp_range:
        
        reg = linear_model.Lasso(alpha = alpha_trial, fit_intercept = True, normalize = True) 
        reg.fit(xtrain, ytrain)
        #clf.coef_, clf.intercept_
        
        py = reg.predict(xval)
        tmp_val = sqrt(mean((py-yval)*(py-yval)))
        
        tmp_err.append([alpha_trial, tmp_val, reg])
    
    # save best model 
    best_model = min(tmp_err, key = lambda x:x[1])[2]
    l1 = min(tmp_err, key = lambda x:x[1])[0]
    tmp_val = min(tmp_err, key = lambda x:x[1])[1]
    
    joblib.dump(best_model, model_file)
    
    # save training prediction under the best model
    py = best_model.predict(xtrain)
    np.savetxt(pred_file + "pytrain_lasso.txt", zip(ytrain, py), delimiter=',')
    
    tmp_tr = sqrt(mean((py-ytrain)*(py-ytrain)))
    
    # [ l1, train error, validation error, test error ]
    result_tuple = [ l1, tmp_tr, tmp_val ]
    
    # save testing prediction under the best model
    if len(xtest)!=0:
        
        py = reg.predict( xtest )
        np.savetxt(pred_file + "pytest_lasso.txt", zip(ytest, py), delimiter=',')
        
        result_tuple.append(sqrt(mean((py-ytest)**2)))
        
    else:
        py = reg.predict( xval )
        np.savetxt(pred_file + "pytest_lasso.txt", zip(yval, py), delimiter=',')
        
        result_tuple.append(None)
        
    # output 
    print "Lasso RMSE: ", result_tuple
    
    # log the overall errors
    with open(result_file, "a") as text_file:
        text_file.write( "Lasso regression: %s \n"%(result_tuple) )
        
    return result_tuple


# ++++ exponential weighted moving average ++++

def ewma_validate(ytrain, yval, ytest, result_file, pred_file):
    
    rmse = []
    pred_tr = []
    pred_ts = []
    
    for alpha in [0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        
        diff_tr = []
        diff_ts = []
        
        tmp_pred = (1.0-alpha)*ytrain[0]
        
        tmp_pred_tr = []
        tmp_pred_ts = []
        
        # training phase
        for i in ytrain:
            diff_tr.append(i-tmp_pred)
            tmp_pred_tr.append(tmp_pred)
            tmp_pred = (1.0-alpha)*i + alpha*tmp_pred
            
        # testing phase
        for i in ytest:
            diff_ts.append(i-tmp_pred)
            tmp_pred_ts.append(tmp_pred)
            tmp_pred = (1.0-alpha)*i + alpha*tmp_pred
        
        diff_tr = np.asarray(diff_tr)
        diff_ts = np.asarray(diff_ts)
        
        rmse.append( [alpha, sqrt(mean(diff_tr*diff_tr)), sqrt(mean(diff_ts*diff_ts))] )
        pred_tr.append( tmp_pred_tr )
        pred_ts.append( tmp_pred_ts )
    
    best = min(rmse, key = lambda x:x[2]) 
    best_alpha = min(rmse, key = lambda x:x[2])[0]
    
    # retrain with the best parameter
        
    tmp_pred = (1.0-alpha)*ytrain[0]
    tmp_pred_tr = []
    tmp_pred_ts = []
        
    # training phase
    for i in ytrain:
        tmp_pred_tr.append(tmp_pred)
        tmp_pred = (1.0-best_alpha)*i + best_alpha*tmp_pred
        
    # testing phase
    for i in ytest:
        tmp_pred_ts.append(tmp_pred)
        tmp_pred = (1.0-best_alpha)*i + best_alpha*tmp_pred
        
    # save the overall errors
    with open(result_file, "a") as text_file:
        text_file.write( "EWMA: %s, \n"%( str(best) ))
        
    # save testing resutls under the best model
    np.savetxt(pred_file + "pytest_ewma.txt", zip(ytest, tmp_pred_ts), delimiter=',')
    
    # save training resutls under the best model
    np.savetxt(pred_file + "pytrain_ewma.txt", zip(ytrain, tmp_pred_tr), delimiter=',')
        
    return best[2]


def ewma_instance_validate(auto_train, ytrain, auto_val, yval, auto_test, ytest, result_file, pred_file):
    
    tmp_val_err = []
    for alpha in [0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        
        # validation phase
        py = []
        for i in range(len(yval)):
            
            tmplen = len(auto_val[i])
            tmp_py = (1.0-alpha)*auto_val[i][0]
            
            for j in range(tmplen):
                tmp_py = (1.0-alpha)*auto_val[i][j] + alpha*tmp_py
        
            py.append(tmp_py)
            
        py = np.asarray(py)
        tmp_val_err.append( [ alpha, sqrt(mean(py-yval)**2) ] )
    
    val_err = min(tmp_val_err, key = lambda x:x[1])[1] 
    best_alpha = min(tmp_val_err, key = lambda x:x[1])[0]
    
    
    # save training prediction under the best model
    py = []
    for i in range(len(ytrain)):
        
        tmplen = len(auto_train[i])
        tmp_py = (1.0-best_alpha)*auto_train[i][0]
        
        for j in range(tmplen):
            tmp_py = (1.0-best_alpha)*auto_train[i][j] + best_alpha*tmp_py
        
        py.append(tmp_py)
            
    py = np.asarray(py)
    train_err = sqrt(mean(py-yval)**2)
    np.savetxt(pred_file + "pytrain_ewma.txt", zip(ytrain, py), delimiter=',')
    
    
    # [ alpha, train error, validation error, test error ]
    result_tuple = [ best_alpha, training_err, val_err ]
    
    # save testing prediction under the best model
    if len(ytest)!=0:

        py = []
        for i in range(len(ytest)):
            
            tmplen = len(auto_test[i])
            tmp_py = (1.0-best_alpha)*auto_test[i][0]
            
            for j in range(tmplen):
                tmp_py = (1.0-best_alpha)*auto_test[i][j] + best_alpha*tmp_py
        
            py.append(tmp_py)
            
        py = np.asarray(py)
        result_tuple.append( sqrt(mean(py-ytest)**2) )
        np.savetxt(pred_file + "pytest_ewma.txt", zip(ytest, py), delimiter=',')
        
    else:
        
        py = []
        for i in range(len(yval)):
            
            tmplen = len(auto_val[i])
            tmp_py = (1.0-best_alpha)*auto_val[i][0]
            
            for j in range(tmplen):
                tmp_py = (1.0-best_alpha)*auto_val[i][j] + best_alpha*tmp_py
        
            py.append(tmp_py)
            
        py = np.asarray(py)
        result_tuple.append( None )
        np.savetxt(pred_file + "pytest_ewma.txt", zip(yval, py), delimiter=',')
        
        
    # save the overall errors
    with open(result_file, "a") as text_file:
        text_file.write( "EWMA: %s, \n"%( str(result_tuple) ))
        
    return result_tuple

