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

file_path = "../bt_results/"


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
    # clf: accuracy, regression: error
    if bool_clf:
        return model.score(x,y) 
    else:
        y_hat  = model.predict(x)
        return sqrt(sum((y_hat-y)*(y_hat-y))/len(y))
    
            
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
score, best_pytest, best_model, utils_evaluation_score(X, Y, bool_clf, best_model) 

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
            
            print tmp_ts
            
            if tmp_ts>tmp_err:
                best_pytest = pytest
                best_model  = clf
                
                tmp_err = tmp_ts
    
    return min(score, key = lambda x: x[1]) if bool_clf == False else max(score, key = lambda x: x[1]), score, \
best_pytest, best_model,utils_evaluation_score(X, Y, bool_clf, best_model) 
        
def gbt_train_validate(xtrain, ytrain, xtest, ytest, fix_lr, bool_clf, result_file, model_file):
    
    print "\nStart to train GBT"

    fix_lr = 0.25

    n_err, n_err_list, y0_hat, model0, train_err0 = gbt_n_estimatior(200, xtrain, ytrain, xtest, ytest, fix_lr, bool_clf)

    print "n_estimator, RMSE:", train_err0, n_err

    depth_err, depth_err_list, y1_hat, model1, train_err1 = gbt_tree_para( xtrain, ytrain, xtest, ytest, range(3,16), \
                                                                          fix_lr, n_err[0], bool_clf )
    print "depth, RMSE:", train_err1, depth_err
    
    # save overall errors 
    with open(result_file, "a") as text_file:
        text_file.write( "GBT %f, %s, %f, %s \n" %(train_err0, str(n_err), train_err1, str(depth_err)) )
    
    # save model and testing results
    if utils_result_comparison(n_err[1], depth_err[1], bool_clf):
        joblib.dump(model0, model_file)
        
        # save testing resutls under the best model
        py = model0.predict( xtest )
        np.savetxt(file_path + "res/pytest_gbt.txt", zip(ytest, py), delimiter=',')
        
        # save training resutls under the best model
        py = model0.predict( xtrain )
        np.savetxt(file_path + "res/pytrain_gbt.txt", zip(ytrain, py), delimiter=',')
        
        return model0
    else:
        joblib.dump(model1, model_file)
        
        # save testing resutls under the best model
        py = model1.predict( xtest )
        np.savetxt(file_path + "res/pytest_gbt.txt", zip(ytest, py), delimiter=',')
        
        # save training resutls under the best model
        py = model1.predict( xtrain )
        np.savetxt(file_path + "res/pytrain_gbt.txt", zip(ytrain, py), delimiter=',')
        
        return model1
    
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
score, best_model, xgt_evaluation_score(xg_train, ytrain, bool_clf, best_model) 


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
score, best_model, xgt_evaluation_score(xg_train, ytrain, bool_clf, best_model)

  
def xgt_train_validate(xtrain, ytrain, xtest, ytest, bool_clf, num_class, result_file, model_file):
    
    print "\nStart to train XGBoosted"
    
    fix_lr = 0.2

    n_depth_err, n_depth_err_list, model0, train_err0 = xgt_n_depth( fix_lr, 16, 41, xtrain, ytrain, xtest, ytest, bool_clf,\
                                                                    num_class)
    print " depth, number of rounds, RMSE:", train_err0, n_depth_err

    l2_err, l2_err_list, model1, train_err1 = xgt_l2( fix_lr, n_depth_err[0], n_depth_err[1], xtrain, ytrain, xtest, ytest,\
                    [0.0001, 0.001, 0.01, 0.1, 1, 10, 100], bool_clf, num_class)
    print " l2, RMSE:", train_err1, l2_err

    
    # specific for XGBoosted
    xg_test  = xgb.DMatrix(xtest,  label = ytest)
    xg_train = xgb.DMatrix(xtrain, label = ytrain)
    
    # save best model and the testing results
    if utils_result_comparison(n_depth_err[2], l2_err[1], bool_clf):
        
        joblib.dump(model0, model_file)
        
        # save testing resutls under the best model
        py = model0.predict( xg_test )
        np.savetxt(file_path + "res/pytest_xgt.txt", zip(ytest, py), delimiter=',')
        
        # save training resutls under the best model
        py = model0.predict( xg_train )
        np.savetxt(file_path + "res/pytrain_xgt.txt", zip(ytrain, py), delimiter=',')
        
    else:
        joblib.dump(model1, model_file)
        
        # save testing resutls under the best model
        py = model1.predict( xg_test )
        np.savetxt(file_path + "res/pytest_xgt.txt", zip(ytest, py), delimiter=',')
        
        # save training resutls under the best model
        py = model1.predict( xg_train )
        np.savetxt(file_path + "res/pytrain_xgt.txt", zip(ytrain, py), delimiter=',')
    
    # save overall errors
    with open(result_file, "a") as text_file:
        text_file.write( "XG-boosted %f, %s, %f, %s\n" %(train_err0, str(l2_err), train_err1, str(n_depth_err)) )
        
#  def xgt_l1 for very high dimensional features    
    
    
# ++++ Random forest ++++

#https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/

# max_features:
# n_estimators
# max_depth

def rf_n_depth_estimatior(maxnum, maxdep, X, Y, xtest, ytest, bool_clf ):
        
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
score, best_pytest, best_model, utils_evaluation_score(X, Y, bool_clf, best_model)


def rf_train_validate(xtrain, ytrain, xtest, ytest, bool_clf, result_file, model_file):
    
    print "\nStart to train Random Forest"

    n_err, n_err_list, y_hat, model, train_err = rf_n_depth_estimatior( 100, 20, xtrain, ytrain, xtest, ytest, bool_clf )
    print "n_estimator, RMSE:", train_err, n_err
    
    # save the best model
    joblib.dump(model, model_file)
    
    # save overall errors
    with open(result_file, "a") as text_file:
        text_file.write( "Random forest %f, %s \n" %(train_err, str(n_err)) )
        
    # save testing resutls under the best model
    py = model.predict( xtest )
    np.savetxt(file_path + "res/pytest_rf.txt", zip(ytest, py), delimiter=',')
    
    # save training resutls under the best model
    py = model.predict( xtrain )
    np.savetxt(file_path + "res/pytrain_rf.txt", zip(ytrain, py), delimiter=',')


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
            
            res.append( (i, j, tmp_tr, tmp_ts) )
            
            if tmp_ts<tmp_err:
                best_model  = enet
                best_pytest = pytest
                tmp_err = tmp_ts
            
    
    return min(res, key = lambda x:x[3]), res, best_pytest, best_model


def elastic_net_train_validate(xtrain, ytrain, xtest, ytest, result_file, model_file, trans_ytrain):
    print "\nStart to train ElasticNet"
    
    tmp_range = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    
    if len(trans_ytrain) != 0:
        err_min, err_list, y_hat, model = enet_alpha_l1( tmp_range, tmp_range, \
                                                    xtrain, trans_ytrain, xtest, ytest, ytrain)
    else:
        err_min, err_list, y_hat, model = enet_alpha_l1( tmp_range, tmp_range, \
                                                    xtrain, ytrain, xtest, ytest, [])
    # save the best model
    joblib.dump(model, model_file)
        
    print "RMSE:", err_min
    
    # save overall errors
    with open(result_file, "a") as text_file:
        text_file.write( "Elastic net %s \n"%( str(err_min)) )
        
    # save testing resutls under the best model
    py = model.predict( xtest )
    np.savetxt(file_path + "res/pytest_enet.txt", zip(ytest, py), delimiter=',')
    
    # save training resutls under the best model
    py = model.predict( xtrain )
    np.savetxt(file_path + "res/pytrain_enet.txt", zip(ytrain, py), delimiter=',')
    
    
# ++++ Bayesian regression ++++

# BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,
#        fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300,
#        normalize=False, tol=0.001, verbose=False)

from sklearn import linear_model

def bayesian_reg_train_validate(xtrain, ytrain, xtest, ytest, result_file, model_file, trans_ytrain):
    
    print "\nStart to train Bayesian Regression"
    
    if len(trans_ytrain) ==0:
        bayesian_reg = linear_model.BayesianRidge( normalize=True, fit_intercept=True )
        bayesian_reg.fit(xtrain, ytrain)
    else:
        bayesian_reg = linear_model.BayesianRidge( normalize=True, fit_intercept=True )
        bayesian_reg.fit(xtrain, trans_ytrain)
    
    pytrain = bayesian_reg.predict( xtrain )
    pytest = bayesian_reg.predict( xtest )
    
    if len(trans_ytrain) ==0:
        tmp_tr = sqrt(mean((pytrain-ytrain)*(pytrain-ytrain)))
        tmp_ts = sqrt(mean((pytest-ytest)*(pytest-ytest)))
    else:
        tmp_tr = sqrt(mean((exp(pytrain)-ytrain)*(exp(pytrain)-ytrain)))
        tmp_ts = sqrt(mean((exp(pytest) -ytest)*(exp(pytest)-ytest)))
    
    print "RMSE: ", tmp_tr, tmp_ts 
    
    # save the best model
    joblib.dump(bayesian_reg, model_file)
    
    # save the overall errors
    with open(result_file, "a") as text_file:
        text_file.write( "Bayeisan regression: %f, %f \n"%(tmp_tr, tmp_ts) )
        
    # save testing resutls under the best model
    py = bayesian_reg.predict( xtest )
    np.savetxt(file_path + "res/pytest_bayes.txt", zip(ytest, py), delimiter=',')
    
    # save training resutls under the best model
    py = bayesian_reg.predict( xtrain )
    np.savetxt(file_path + "res/pytrain_bayes.txt", zip(ytrain, py), delimiter=',')

        
# ++++ Ridge regression ++++

from sklearn.linear_model import Ridge

def ridge_reg_train_validate(xtrain, ytrain, xtest, ytest, result_file, model_file, trans_ytrain):
    
    print "\nStart to train Ridge Regression"
    
    tmp_range = [0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2]
    tmp_err = []
    
    best_err = np.inf
   
    for alpha_trial in tmp_range:
        
        if len(trans_ytrain) ==0:
            clf = Ridge(alpha = alpha_trial, fit_intercept = True, normalize= True)
            clf.fit(xtrain, ytrain)
        else:
            clf = Ridge(alpha = alpha_trial, fit_intercept = True, normalize= True)
            clf.fit(xtrain, trans_ytrain)
        
        pytest = clf.predict( xtest )
        
        if len(trans_ytrain) ==0:
            tmp_ts = sqrt(mean((pytest-ytest)*(pytest-ytest)))
        else:
            tmp_ts = sqrt(mean((exp(pytest)-ytest)*(exp(pytest)-ytest)))
        
        if tmp_ts < best_err:
            best_err = tmp_ts
            best_model = clf
        
        tmp_err.append( [alpha_trial, tmp_ts] )
    
    pytrain = best_model.predict( xtrain )
    if len(trans_ytrain) ==0:
        tmp_tr = sqrt(mean((pytrain-ytrain)*(pytrain-ytrain)))
    else:
        tmp_tr = sqrt(mean((exp(pytrain)-ytrain)*(exp(pytrain)-ytrain)))
    
    print "RMSE: ", min(tmp_err, key = lambda x:x[1])[0], tmp_tr, best_err
    
    # save best model 
    joblib.dump(best_model, model_file)
    
    # save the overall errors
    with open(result_file, "a") as text_file:
        text_file.write( "Ridge regression: %f, %f, %f \n"%(min(tmp_err, key = lambda x:x[1])[0], tmp_tr, best_err) )
    
    # save testing resutls under the best model
    py = best_model.predict( xtest )
    np.savetxt(file_path + "res/pytest_ridge.txt", zip(ytest, py), delimiter=',')
    
    # save training resutls under the best model
    py = best_model.predict( xtrain )
    np.savetxt(file_path + "res/pytrain_ridge.txt", zip(ytrain, py), delimiter=',')

# ++++ gaussian process ++++

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *

def gp_train_validate(xtrain, ytrain, xtest, ytest, result_file, model_file, trans_ytrain):
    
    print "\nStart to train Gaussian process"
    
    kernel = 1.0*RBF(length_scale=41.8) + 1.0*RBF(length_scale=180) * ExpSineSquared(length_scale=1.44, periodicity=1) \
+ RationalQuadratic(alpha=17.7, length_scale=0.957) +  RBF(length_scale=0.138) + WhiteKernel(noise_level=0.0336)
    
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, optimizer = 'fmin_l_bfgs_b', normalize_y = True )
    
    gp.fit(xtrain, ytrain)
    #gp.kernel_, gp.alpha_
    
    print "Begin to evaluate Gaussian process regression"
    
    pytrain, sigma_train = gp.predict(xtrain, return_std=True)
    pytest,  sigma_test  = gp.predict(xtest, return_std=True)
    
    tmp_tr = sqrt(mean((pytrain-ytrain)*(pytrain-ytrain)))
    tmp_ts = sqrt(mean((pytest-ytest)*(pytest-ytest)))
    
    print "RMSE: ", tmp_tr, tmp_ts
    
    # save best model 
    joblib.dump(gp, model_file)
    
    # save the overall errors
    with open(result_file, "a") as text_file:
        text_file.write( "Gaussian process regression: %f, %f \n"%(tmp_tr, tmp_ts) )
    
    # save testing resutls under the best model
    py = gp.predict( xtest )
    np.savetxt(file_path + "res/pytest_gp.txt", zip(ytest, py), delimiter=',')
    
    # save training resutls under the best model
    py = gp.predict( xtrain )
    np.savetxt(file_path + "res/pytrain_gp.txt", zip(ytrain, py), delimiter=',')
    

# ++++ lasso regression ++++

from sklearn import linear_model

def lasso_train_validate(xtrain, ytrain, xtest, ytest, result_file, model_file, trans_ytrain):
    
    print "\nStart to train Lasso regression"
    
    tmp_range = [0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 4, 6]
    tmp_err = []
       
    for alpha_trial in tmp_range:
        
        reg = linear_model.Lasso(alpha = alpha_trial, fit_intercept = True, normalize = True) 
        reg.fit(xtrain, ytrain)
        #clf.coef_, clf.intercept_
        
        # test
        #print reg.coef_
        
        pytest = reg.predict(xtest)
        tmp_ts = sqrt(mean((pytest-ytest)*(pytest-ytest)))
        
        tmp_err.append([alpha_trial, tmp_ts, reg])
        
        print alpha_trial, tmp_ts
        
    pytrain = reg.predict(xtrain)
    tmp_tr = sqrt(mean((pytrain-ytrain)*(pytrain-ytrain)))
    
    print "RMSE: ", min(tmp_err, key = lambda x:x[1])[0], tmp_tr, min(tmp_err, key = lambda x:x[1])[1]
    
    # save best model 
    best_model = min(tmp_err, key = lambda x:x[1])[2]
    joblib.dump(best_model, model_file)
    
    # save the overall errors
    with open(result_file, "a") as text_file:
        text_file.write( "Lasso regression: %f, %f, %f \n"%(min(tmp_err, key = lambda x:x[1])[0],\
                                                                       tmp_tr, min(tmp_err, key = lambda x:x[1])[1]))
    # save testing resutls under the best model
    py = reg.predict( xtest )
    np.savetxt(file_path + "res/pytest_lasso.txt", zip(ytest, py), delimiter=',')
    
    # save training resutls under the best model
    py = reg.predict( xtrain )
    np.savetxt(file_path + "res/pytrain_lasso.txt", zip(ytrain, py), delimiter=',')
    


