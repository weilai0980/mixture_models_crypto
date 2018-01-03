#!/usr/bin/python

from utils_libs import *
from regression_models import *

# --- model and training log set-up ---
result_file = "../bt_results/res/reg_v.txt"
# test and training errors 

# ?
model_file = "../bt_results/model/v_"
bool_clf = False

# clean the 
#with open(result_file, "w") as text_file:
#    text_file.close()

# --- Load pre-processed training and testing data ---
file_postfix = "v_reg"

xtrain = np.load("../dataset/bitcoin/training_data/xtrain_"+file_postfix+".dat")
xtest  = np.load("../dataset/bitcoin/training_data/xtest_" +file_postfix+".dat")
ytrain = np.load("../dataset/bitcoin/training_data/ytrain_"+file_postfix+".dat")
ytest  = np.load("../dataset/bitcoin/training_data/ytest_" +file_postfix+".dat")

# all feature vectors are already normalized
print np.shape(xtrain), np.shape(ytrain), np.shape(xtest), np.shape(ytest)


# --- start training different models ---


# GBT gradient boosted tree
gbt_train_validate(xtrain, ytrain, xtest, ytest, 0.0, bool_clf, result_file, model_file + '_gbt.sav')

# Random forest performance
rf_train_validate(xtrain, ytrain, xtest, ytest, bool_clf, result_file, model_file + '_rf.sav')


# XGBoosted extreme gradient boosted
xgt_train_validate(xtrain, ytrain, xtest, ytest, bool_clf, 0, result_file, model_file + '_xgt.sav')


# log transformation of y
log_ytrain = []
#log(ytrain+1e-5)

# Gaussain process 
gp_train_validate(xtrain, ytrain, xtest, ytest, result_file, model_file + '_gp.sav', log_ytrain)


# Bayesian regression
bayesian_reg_train_validate(xtrain, ytrain, xtest, ytest, result_file, model_file + '_bayes.sav', log_ytrain)

# ElasticNet
elastic_net_train_validate(xtrain, ytrain, xtest, ytest, result_file, model_file + '_enet.sav', log_ytrain)

#Ridge regression
ridge_reg_train_validate(xtrain, ytrain, xtest, ytest, result_file, model_file + '_ridge.sav', log_ytrain)


# Lasso 
lasso_train_validate(xtrain, ytrain, xtest, ytest, result_file, model_file + '_lasso.sav', log_ytrain)










