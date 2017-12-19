#!/usr/bin/python

from utils_libs import *
from regression_models import *

# --- model and training log set-up ---
result_file = "res/reg_v_minu.txt"
# test and training errors 

model_file = "model/v_minu"
bool_clf = False

# clean the 
#with open(result_file, "w") as text_file:
#    text_file.close()

# --- Load pre-processed training and testing data ---
file_postfix = "v_minu_reg"
xtrain = np.load("../dataset/bitcoin/training_data/xtrain_"+file_postfix+".dat")
xtest  = np.load("../dataset/bitcoin/training_data/xtest_" +file_postfix+".dat")
ytrain = np.load("../dataset/bitcoin/training_data/ytrain_"+file_postfix+".dat")
ytest  = np.load("../dataset/bitcoin/training_data/ytest_" +file_postfix+".dat")

# all feature vectors are already normalized
print np.shape(xtrain), np.shape(ytrain), np.shape(xtest), np.shape(ytest)

# --- start training different models ---

'''
# GBT gradient boosted tree
gbt_train_validate(xtrain, ytrain, xtest, ytest, 0.0, bool_clf, result_file, model_file + '_gbt.sav')

# Random forest performance
rf_train_validate(xtrain, ytrain, xtest, ytest, bool_clf, result_file, model_file + '_rf.sav')

# XGBoosted extreme gradient boosted
xgt_train_validate(xtrain, ytrain, xtest, ytest, bool_clf, 0, result_file, model_file + '_xgt.sav')
'''

# log transformation of y
log_ytrain = []
#log(ytrain+1e-5)

'''
# Bayesian regression
bayesian_reg_train_validate(xtrain, ytrain, xtest, ytest, result_file, model_file + '_bayes.sav', log_ytrain)

# ElasticNet
elastic_net_train_validate(xtrain, ytrain, xtest, ytest, result_file, model_file + '_enet.sav', log_ytrain)

#Ridge regression
ridge_reg_train_validate(xtrain, ytrain, xtest, ytest, result_file, model_file + '_ridge.sav', log_ytrain)
'''

# Gaussain process 
#gp_train_validate(xtrain, ytrain, xtest, ytest, result_file, model_file + '_gp.sav', log_ytrain)

# Lasso 
lasso_train_validate(xtrain, ytrain, xtest, ytest, result_file, model_file + '_lasso.sav', log_ytrain)










