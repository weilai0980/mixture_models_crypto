#!/usr/bin/python

import gzip
import os
import tempfile

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib import rnn

import math
import random

import collections
import hashlib
import numbers

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn_cell_impl import * 

# library for bayesian inference
import edward as ed
from edward.models import Categorical, Dirichlet, Empirical, InverseGamma, MultivariateNormalDiag, Normal, ParamMixture, Beta, Bernoulli, Mixture

# local packages
from utils_libs import *
from ts_mv_rnn_basics import *

# reproducibility by fixing the random seed
np.random.seed(1)
tf.set_random_seed(1)

# ---- utilities functions ----

def linear( x, dim_x, scope, bool_bias ):
    
    with tf.variable_scope(scope):
        w = tf.Variable( tf.random_normal([dim_x, 1], stddev = math.sqrt(1.0/float(dim_x))) )
        b = tf.Variable( tf.zeros([1,]))
        
        if bool_bias == True:
            h = tf.matmul(x, w) + b
        else:
            h = tf.matmul(x, w)
           
        #l2
        regularizer = tf.nn.l2_loss(w)
        
    return tf.squeeze(h), regularizer

'''
def linear_predict( x, dim_x, scope, bool_bias ):
    
    with tf.variable_scope(scope):
        w = tf.Variable( tf.random_normal([dim_x, 1], stddev = math.sqrt(1.0/float(dim_x))) )
        b = tf.Variable( tf.zeros([1,]))
        
        if bool_bias == True:
            h = tf.matmul(x, w) + b
        else:
            h = tf.matmul(x, w)
            
        #l2
        regularizer = tf.nn.l2_loss(w)
        
        # hinge regularization
        vec_zero = tf.zeros([dim_x, ]) 
        hinge_w = tf.maximum(vec_zero, w)
        
    #regularizer +     
    # ?
    return tf.squeeze(h), regularizer
'''

# shape of x: [b, t, v]
def bilinear( x, shape_one_x, scope, bool_bias ):
    
    with tf.variable_scope(scope):
        
        w_t = tf.Variable(name = 'temp', initial_value = \
                          tf.random_normal([shape_one_x[0], 1], stddev = math.sqrt(1.0/float(shape_one_x[0]))) )
        w_v = tf.Variable(name = 'vari', initial_value = \
                          tf.random_normal([shape_one_x[1], 1], stddev = math.sqrt(1.0/float(shape_one_x[1]))) )
        
        b = tf.Variable( tf.zeros([1,]) )
        
        tmph = tf.tensordot(x, w_v, 1)
        tmph = tf.squeeze(tmph, [2])
        
        if bool_bias == True:
            h = tf.matmul(tmph, w_t) + b
        else:
            h = tf.matmul(tmph, w_t)
            
    # l2, l1 regularization
    #tf.nn.l2_loss(w_t) tf.reduce_sum(tf.abs(w_v)
    return tf.squeeze(h), [ tf.reduce_mean(tf.square(w_t)), tf.reduce_mean(tf.abs(w_v)) ] 

'''
# shape of x: [b, t, v]
def bilinear_with_external( x, shape_one_x, exter, exter_dim, scope, bool_bias ):
    
    with tf.variable_scope(scope):
        
        w_t = tf.Variable(name = 'temp', initial_value = \
                          tf.random_normal([shape_one_x[0], 1], stddev = math.sqrt(1.0/float(shape_one_x[0]))) )
        w_v = tf.Variable(name = 'vari', initial_value = \
                          tf.random_normal([shape_one_x[1], 1], stddev = math.sqrt(1.0/float(shape_one_x[1]))) )
        
        w_ex = tf.Variable(name = 'ex', initial_value = \
                          tf.random_normal([exter_dim, 1], stddev = math.sqrt(1.0/float(exter_dim))) )
        
        b = tf.Variable( tf.zeros([1,]) )
        
        tmph = tf.tensordot(x, w_v, 1)
        tmph = tf.squeeze(tmph, [2])
        
        
        w_mat_l = tf.Variable(name = 'i_mat_left', initial_value = \
                          tf.random_normal([exter_dim, 1], stddev = math.sqrt(1.0/float(exter_dim))) )
        
        w_mat_r = tf.Variable(name = 'i_mat_right', initial_value = \
                          tf.random_normal([1, shape_one_x[0]], stddev = math.sqrt(1.0/float(exter_dim))) )
        w_mat = tf.matmul(w_mat_l, w_mat_r)
        
        w_vec = tf.Variable(name = 'i_vec', initial_value = \
                          tf.random_normal([shape_one_x[1], 1], stddev = math.sqrt(1.0/float(exter_dim))) )
        
        i_h = tf.matmul(exter, w_mat)
        i_h = tf.expand_dims(i_h, -1)
        tmp_i_h = tf.reduce_sum(i_h*x, 1)
        i_b = tf.matmul(tmp_i_h, w_vec)
        
        
        if bool_bias == True:
            h = tf.matmul(tmph, w_t) + tf.matmul(exter, w_ex) + b 
        else:
            h = tf.matmul(tmph, w_t) + tf.matmul(exter, w_ex) 
            
    # l2, l1 regularization
    
    #tf.nn.l2_loss(w_t) tf.reduce_sum(tf.abs(w_v)
    return tf.squeeze(h), [ tf.nn.l2_loss(w_t) + tf.nn.l2_loss(w_v), tf.nn.l2_loss(w_ex) ] 
'''

# ---- Mixture linear ----

class mixture_linear():

    def __init__(self, session, lr, l2, batch_size, steps_auto, dim_x, steps_x, bool_log, bool_bilinear, \
                 loss_type, distr_type, activation_type, pos_regu, gate_type):
        
        # build the network graph 
        self.LEARNING_RATE = lr
                
        self.N_BATCH = batch_size
        self.L2 = l2
        
        self.MAX_NORM = 0.0
        self.epsilon = 1e-3
        
        self.sess = session
        
        self.bool_log = bool_log
        self.loss_type = loss_type
        self.distr_type = distr_type
        
        # initialize placeholders
        self.auto = tf.placeholder(tf.float32, [None, steps_auto])
        self.y = tf.placeholder(tf.float32, [None, ])
        self.keep_prob = tf.placeholder(tf.float32)
        
        if bool_bilinear == True:
            self.x = tf.placeholder(tf.float32, [None, steps_x, dim_x])
        else:
            self.x = tf.placeholder(tf.float32, [None, dim_x])
        
        
        # --- prediction of individual models

        # models on individual feature groups
        mean_v, regu_v_mean = linear(self.auto, steps_auto, 'mean_v', True)
        
        if bool_bilinear == True:
            mean_x, regu_x_mean = bilinear(self.x, [steps_x, dim_x], 'mean_distr', True)
        else:
            mean_x, regu_x_mean = linear(self.x, dim_x, 'mean_distr', True)

        # activation 
        if activation_type == 'relu':
            mean_x = tf.nn.relu(mean_x)
            mean_v = tf.nn.relu(mean_v)
            
        elif activation_type == 'leaky_relu':
            mean_x = tf.maximum( mean_x, 0.2*mean_x )
            mean_v = tf.maximum( mean_v, 0.2*mean_v )
        
        elif activation_type == 'linear':
            mean_x = mean_x
            mean_v = mean_v
        else:
            print ' ------- [ERROR] activation type'
            
        
        # concatenate individual means 
        self.mean_stack = tf.stack( [mean_v, mean_x], 1 )
        mean_stack = tf.stack( [mean_v, mean_x], 1 )
        
        # --- variance of individual models
        
        # variance depedent on features 
        varv, regu_v_var = linear(self.auto, steps_auto, 'sig_v', True)
        
        # only linear 
        vardistr, regu_x_var = linear( tf.reshape(self.x, [-1, steps_x*dim_x]), steps_x*dim_x, 'sig_distr', True)
        
        var_v = tf.square(varv)
        var_x = tf.square(vardistr)
        
        # concatenate individual variance 
        var_stack = tf.stack( [var_v, var_x], 1 )
        
        # variance constant for each expert
        #varv =    tf.Variable(tf.random_normal([1,], stddev = math.sqrt(2.0/float(1.0))) )
        #varreq =  tf.Variable(tf.random_normal([1,], stddev = math.sqrt(2.0/float(1.0))) )
        #vardistr= tf.Variable(tf.random_normal([1,], stddev = math.sqrt(2.0/float(1.0))) )
        #regu_sig = tf.nn.l2_loss(varv)+tf.nn.l2_loss(vardistr)
        
        # inverse variance standardize 
        #inv_var_v = tf.square(inv_varv)
        #inv_var_d = tf.square(inv_vardistr)
        
        
        # --- gate weights of individual models
        
        # logstic
        if gate_type == 'logstic-partial':
            
            if bool_bilinear == True:
                self.logit, regu_gate = bilinear(self.x, [steps_x, dim_x], 'gate_x', True)
            
            else:
                self.logit, regu_gate = linear(self.x, dim_x, 'gate_x', True)
        
            self.gates = tf.stack( [1.0 - tf.sigmoid(self.logit), tf.sigmoid(self.logit)] ,1 )
        
        elif gate_type == 'logstic-concat':
            
            logit_v, regu_v_gate = linear(self.auto, steps_auto, 'gate_v', True)
        
            if bool_bilinear == True:
                logit_x, regu_x_gate = bilinear(self.x, [steps_x, dim_x], 'gate_x', True)
            
            else:
                logit_x, regu_x_gate = linear(self.x, dim_x, 'gate_x', True)
            
            self.logit = logit_v + logit_x
            
            self.gates = tf.stack( [1.0 - tf.sigmoid(self.logit), tf.sigmoid(self.logit)] ,1 )
            
        # softmax 
        elif gate_type == 'softmax':
            
            logit_v, regu_v_gate = linear(self.auto, steps_auto, 'gate_v', True)
        
            if bool_bilinear == True:
                logit_x, regu_x_gate = bilinear(self.x, [steps_x, dim_x], 'gate_x', True)
            
            else:
                logit_x, regu_x_gate = linear(self.x, dim_x, 'gate_x', True)
        
            self.logit = tf.squeeze( tf.stack([logit_v, logit_x], 1) )
            self.gates = tf.nn.softmax(self.logit)
        
        else:
            
            print '\n ----- [ERROR] gate type \n'
            
        # --- negative log likelihood 
        
        if distr_type == 'gaussian':
            tmpllk_v = tf.exp(-0.5*tf.square(self.y - mean_v)/(var_v + 1e-5))/(2.0*np.pi*(var_v + 1e-5))**0.5
            tmpllk_x = tf.exp(-0.5*tf.square(self.y - mean_x)/(var_x + 1e-5))/(2.0*np.pi*(var_x + 1e-5))**0.5
            
        elif distr_type == 'log':
            
            tmpllk_v = tf.exp(-0.5*tf.square(tf.log(self.y+1e-10) - mean_v)/(var_v + 1e-10))/(2.0*np.pi*(var_v + 1e-10))**0.5/(self.y+1e-10)
            
            tmpllk_x = tf.exp(-0.5*tf.square(tf.log(self.y+1e-10) - mean_x)/(var_x + 1e-10))/(2.0*np.pi*(var_x + 1e-10))**0.5/(self.y+1e-10)
        
        else:
            print '[ERROR] distribution type'
        
        llk = tf.multiply( (tf.stack([tmpllk_v, tmpllk_x], 1)), self.gates ) 
        self.neg_logllk = tf.reduce_sum( -1.0*tf.log(tf.reduce_sum(llk, 1)+1e-5) )
        
        
        # --- regularization
        
        # temporal coherence, diversity 
        
        # gate smoothness 
        logit_v_diff = logit_v[1:]-logit_v[:-1]
        logit_x_diff = logit_x[1:]-logit_x[:-1]
        
        def tf_var(tsr):
            return tf.reduce_mean(tf.square(tsr - tf.reduce_mean(tsr)))
        
        regu_gate_smooth = tf_var(logit_v_diff) + tf_var(logit_x_diff)  
        
        # gate diversity
        gate_diff = logit_v - logit_x
        regu_gate_diver = tf_var(gate_diff)
        
        # mean diversity
        regu_mean_diver = tf.reduce_mean( tf.maximum(0.0, (mean_v - self.y)*(mean_x - self.y)) )
        
        # mean non-negative  
        regu_mean_pos = tf.reduce_sum( tf.maximum(0.0, -1.0*mean_v) + tf.maximum(0.0, -1.0*mean_x) )
        
        
        if bool_bilinear == True:
            
            if loss_type == 'sq' and distr_type == 'gaussian':
                
                self.regu = 0.01*(regu_v_mean) + 0.001*(regu_x_mean[0]) + 0.001*(regu_x_mean[1])+\
                            0.001*(regu_v_gate) + 0.0001*(regu_x_gate[0] + regu_x_gate[1])\
                            + 0.001*regu_mean_pos\
                            #+ 0.0001*regu_mean_diver
                            
            elif loss_type == 'lk' and distr_type == 'gaussian':
                
                
                #self.regu = 0.001*(regu_v_mean) + 0.001*(regu_x_mean[0]) + 0.001*(regu_x_mean[1])\
                #        + 0.0001*(regu_v_gate) + 0.00001*(regu_x_gate[0] + regu_x_gate[1])\
                #        + 0.00001*(regu_v_var + regu_x_var)
                        #+ 0.001*regu_mean_diver   
                
                # activation and hinge regularization 
                #if pos_regu == True:
                #    self.regu += 0.001*regu_mean_pos
                    
                    
                self.regu = l2*regu_v_mean + l2*regu_x_mean[0] + l2*regu_x_mean[1]\
                            + 0.1*l2*regu_v_gate + 0.01*l2*(regu_x_gate[0] + regu_x_gate[1])\
                            + 0.01*l2*(regu_v_var + regu_x_var)
                        #+ 0.001*regu_mean_diver   
                
                # activation and hinge regularization 
                if pos_regu == True:
                    self.regu += l2*regu_mean_pos
                    
                
            elif loss_type == 'sq' and distr_type == 'log':
                
                self.regu = 0.01*(regu_v_mean) + 0.001*(regu_x_mean[0]) + 0.00001*(regu_x_mean[1])+\
                        0.001*(regu_v_gate) + 0.00001*(regu_x_gate[0] + regu_x_gate[1])
            
            elif loss_type == 'lk' and distr_type == 'log':
                
                self.regu = 0.001*(regu_v_mean) + 0.001*(regu_x_mean[0]) + 0.001*(regu_x_mean[1])+\
                            0.0001*(regu_v_gate) + 0.0001*(regu_x_gate[0] + regu_x_gate[1])\
                            #+ 0.0001*regu_mean_diver
            else:
                print '[ERROR] loss type'
        
        else:
            
            if loss_type == 'sq' and distr_type == 'gaussian':
                
                self.regu = 0.01*regu_v_mean + 0.01*regu_x_mean\
                          + 0.0001*(regu_v_gate + regu_x_gate) - 0.001*regu_gate_diver
            
            elif loss_type == 'lk' and distr_type == 'gaussian':
                
                self.regu = 0.01*regu_v_mean + 0.0001*regu_x_mean + 0.0001*(regu_v_gate + regu_x_gate)
                        
            elif loss_type == 'sq' and distr_type == 'log':
                
                self.regu = 0.01*regu_v_mean + 0.0001*regu_x_mean + 0.0001*(regu_v_gate + regu_x_gate)
            
            elif loss_type == 'lk' and distr_type == 'log':
                
                self.regu = 0.001*regu_v_mean + 0.0001*regu_x_mean + 0.0001*(regu_v_gate + regu_x_gate)
            
            else:
                print '[ERROR] loss type'
        
        
        # --- mixture prediction, mean and variance
        
        if distr_type == 'gaussian':
            
            # mean and prediction
            self.y_hat = tf.reduce_sum(tf.multiply(mean_stack, self.gates), 1)
            
            self.pre_v = mean_v
            self.pre_x = mean_x
            
            # variance 
            sq_mean_stack =  var_stack - tf.square(mean_stack)
            mix_sq_mean = tf.reduce_sum(tf.multiply(sq_mean_stack, self.gates), 1)
            
            self.var_hat = mix_sq_mean - tf.square(self.y_hat)
            
        elif distr_type == 'log':
            
            # mean and prediction
            self.y_hat = tf.reduce_sum(tf.multiply(tf.exp(mean_stack), self.gates), 1)
            
            self.pre_v = tf.exp(mean_v)
            self.pre_x = tf.exp(mean_x)
            
            self.orig = mean_v
            
            # variance 
            # TO DO
            
        else:
            print '[ERROR] distribution type'
        
    
    def evaluate_ini(self):
        
        # --- error metric
        
        # RMSE
        self.mse  = tf.losses.mean_squared_error( self.y, self.y_hat )
        self.rmse = tf.sqrt(self.mse)
        # MAE
        self.mae  = tf.reduce_mean( tf.abs(self.y - self.y_hat) )
        # MAPE
        
        mask = tf.greater(tf.abs(self.y), 0.00001)
        
        y_mask = tf.boolean_mask(self.y, mask)
        y_hat_mask = tf.boolean_mask(self.y_hat, mask)
        
        self.mape = tf.reduce_mean( tf.abs((y_mask - y_hat_mask)/(y_mask+1e-10)) )
        
        #self.mape = tf.reduce_mean( tf.abs((self.y - self.y_hat)/(self.y+1e-10)) )
        
    def model_reset(self):
        self.init = tf.global_variables_initializer()
        self.sess.run( self.init )
        
#   initialize loss and optimization operations for training
    def train_ini(self):
        
        # loss
        if self.loss_type == 'sq':
            self.loss = self.mse + self.regu
            
        elif self.loss_type == 'lk':
            self.loss = self.neg_logllk + self.regu
        
        else:
            print '[ERROR] loss type'
        
        self.train = tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE)
        self.optimizer =  self.train.minimize(self.loss)
        
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        
        # !
        #self.optimizer = tf.train.ProximalAdagradOptimizer(learning_rate = 0.05, \
        #                                                  l2_regularization_strength = 0.0002, \
        #                                                  l1_regularization_strength = 0.005).minimize(self.lk_loss) 
        
        #self.optimizer = tf.train.FtrlOptimizer(learning_rate = 0.03, \
        #                                        l2_regularization_strength = 0.001, \
        #                                        l1_regularization_strength = 1.0).minimize(self.neg_logllk)
        
        
    #   training on batch of data
    def train_batch(self, auto_train, x_train, y_train, keep_prob ):
        
        # !
        _, c = self.sess.run([self.optimizer, self.neg_logllk],\
                             feed_dict={self.auto: auto_train, \
                                        self.x:x_train, self.y:y_train, self.keep_prob:keep_prob })
        return c
    
    #   infer givn testing data
    def inference(self, auto_test, x_test, y_test, keep_prob):
        
        return self.sess.run([self.rmse, self.mae, self.mape, self.regu], feed_dict = {self.auto:auto_test, \
                                                    self.x:x_test,  self.y:y_test, self.keep_prob:keep_prob })
    
    #   prediction
    def predict(self, auto_test, x_test, keep_prob):
        
        return self.sess.run( [self.y_hat, self.pre_v, self.pre_x], feed_dict = {self.auto:auto_test, \
                                                       self.x:x_test,  self.keep_prob:keep_prob })
    
    #   mixture gates
    def predict_gates(self, auto_test, x_test, keep_prob):
        return self.sess.run( self.gates , feed_dict = {self.auto:auto_test, \
                                                        self.x:x_test,  self.keep_prob:keep_prob })
    #   mixture logits
    def predict_logit(self, auto_test, x_test, keep_prob):
        return self.sess.run( self.logit , feed_dict = {self.auto:auto_test, \
                                                        self.x:x_test,  self.keep_prob:keep_prob })
    # collect the optimized variable values
    def collect_coeff_values(self, vari_keyword):
        return [ tf_var.name for tf_var in tf.trainable_variables() if (vari_keyword in tf_var.name) ],\
    [ tf_var.eval() for tf_var in tf.trainable_variables() if (vari_keyword in tf_var.name) ]
        
    def test(self, auto_train, x_train, y_train, keep_prob ):
        
        #return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mean_v')
        # [var.name for var in tf.trainable_variables()],
        return self.sess.run([tf.shape(self.mean_stack)],\
                             feed_dict={self.auto:auto_train, \
                                        self.x:x_train, self.y:y_train, self.keep_prob:keep_prob })
    
    
# ---- Bayesian variational mixture linear Edward ----


scope_name = ['mean_v', 'mean_distr', 'sig_v', 'sig_distr', 'gate_v', 'gate_distr']

posterior_id = ['sig_distr/b/', 'mean_v/b/', 'gate_distr/w_l/', 'mean_distr/w_l/', 'gate_distr/w_r/', 'sig_v/b/', 'mean_distr/b/', 'mean_v/w/', 'gate_v/w/', 'mean_distr/w_r/', 'sig_v/w/', 'gate_v/b/', 'gate_distr/b/', 'sig_distr/w/']

# can be used for posterior expectation
def bayesian_linear(x, dim_x, scope, bool_bias, dict_var, w_sample, b_sample, infer_type):
    
    with tf.variable_scope(scope):
        
        # for training phase
        if w_sample == None and b_sample == None:
            # prior
            w = Normal(loc=tf.zeros([dim_x, 1]), scale=tf.ones([dim_x, 1]) )
            b = Normal(loc=tf.zeros([1,]), scale=tf.ones([1,]) )    
            
            # variational posterior
            if infer_type == 'variational_nocondi':
                
                qW = Normal(name = 'w', loc=tf.get_variable("qW/loc", [dim_x, 1]), \
                    scale= tf.ones([dim_x, 1]))
                
                #tf.nn.softplus(tf.get_variable("qW/scale", [dim_x, 1])))
                qb = Normal(name = 'b', loc=tf.get_variable("qb/loc", [1,]), \
                        scale=tf.ones([1,]))
                    #scale=tf.nn.softplus(tf.get_variable("qb/scale", [1,])))
        
                dict_var.update({w:qW})
                dict_var.update({b:qb})
                
            
            # gibbs posterior
            elif infer_type == 'gibbs':
                
                T = 500  # number of MCMC samples
                qW = Empirical(tf.get_variable("qW/loc", [T, dim_x, 1], initializer = tf.constant_initializer(1.0 / K)))
                qb = Empirical(tf.get_variable("qb/loc", [T, 1], initializer = tf.zeros_initializer()))
                
                dict_var.update({w:qW})
                dict_var.update({b:qb})
                
            else:
                print '---- [ERROR] inference type error'
                
        
        # for the inference phase
        # given weight samples
        else:
            w = w_sample
            b = b_sample
        
        if bool_bias == True:
            h = tf.matmul(x, w) + b
        else:
            h = tf.matmul(x, w)
        
    return h
    
def bayesian_bilinear(x, shape_one_x, scope, bool_bias, dict_var, wl_sample, wr_sample, b_sample, infer_type):
    
    # for training phase
    with tf.variable_scope(scope):
        
        if wl_sample == None and wr_sample == None and b_sample == None:
            
            # prior
            w_l = Normal(loc=tf.zeros([shape_one_x[0], 1]), scale=tf.ones([shape_one_x[0], 1]))
            w_r = Normal(loc=tf.zeros([shape_one_x[1], 1]), scale=tf.ones([shape_one_x[1], 1]))
            b = Normal(loc=tf.zeros([1,]), scale=tf.ones([1,])) 
            
            
            # variational posterior
            if infer_type == 'variational_nocondi':
                
                
                # no conditions
                
                qW_l = Normal(name = 'w_l',\
                              loc=tf.get_variable("qW_l/loc", [shape_one_x[0], 1]), \
                              scale=tf.ones([shape_one_x[0], 1]))
                
                      #scale=tf.nn.softplus(tf.get_variable("qW_t/scale",[shape_one_x[0], 1])))
                qW_r = Normal(name = 'w_r',\
                              loc=tf.get_variable("qW_r/loc", [shape_one_x[1], 1]), \
                              scale=tf.ones([shape_one_x[1], 1]) )
                      #scale=tf.nn.softplus(tf.get_variable("qW_v/scale", [shape_one_x[1], 1])))
                
                qb   = Normal(name = 'b',\
                              loc=tf.get_variable("qb/loc", [1,]),\
                              scale=tf.ones([1,]))
                      #scale=tf.nn.softplus(tf.get_variable("qb/scale", [1,])))
                    
                dict_var.update({w_l : qW_l})
                dict_var.update({w_r : qW_r})
                dict_var.update({b : qb})
                    
            # gibbs posterior
            elif infer_type == 'gibbs':
                
                T = 500  # number of MCMC samples
                qW_l = Empirical(tf.get_variable("qW_l/loc", [T, shape_one_x[0], 1], \
                                                 initializer = tf.constant_initializer(1.0 / K)))
                qW_r = Empirical(tf.get_variable("qW_r/loc", [T, shape_one_x[1], 1], \
                                                 initializer = tf.constant_initializer(1.0 / K)))
                qb   = Empirical(tf.get_variable("qb/loc", [T, 1], initializer = tf.zeros_initializer()))
                
                dict_var.update({w_l : qW_l})
                dict_var.update({w_r : qW_r})
                dict_var.update({b : qb})
                
            else:
                print '---- [ERROR] inference type error'
                
        # for inference phase
        # given weight samples
        else:
            w_l = wl_sample
            w_r = wr_sample
            b   = b_sample
            
        tmph = tf.tensordot(x, w_r, 1)
        tmph = tf.squeeze(tmph, [2])
        
        if bool_bias == True:
            h = tf.matmul(tmph, w_l) + b
        else:
            h = tf.matmul(tmph, w_l)
            
    return h


'''
TO DO:
posterior parameter variable
inference method

'''

class variational_mixture_linear():

    def __init__(self, session, lr, batch_size, steps_auto, dim_x, steps_x, bool_log, bool_bilinear, \
                 distr_type, eval_sample_num, infer_type, num_iter, loss_sample_num, num_batch_per_epoch):
        
        # build the network graph 
        self.LEARNING_RATE = lr
                 
        self.N_BATCH = batch_size
   
        self.epsilon = 1e-3
        
        self.sess = session
        
        self.bool_log = bool_log
        self.distr_type = distr_type
        
        self.dict_var = {}
        
        self.eval_sample_num = eval_sample_num
        self.loss_sample_num = loss_sample_num
        self.num_iter = num_iter
        self.infer_type = infer_type
        self.num_batch_per_epoch = num_batch_per_epoch
        
        # initialize placeholders
        self.auto = tf.placeholder(tf.float32, [None, steps_auto])
        self.y = tf.placeholder(tf.float32, [None, ])
        self.keep_prob = tf.placeholder(tf.float32)
        
        if bool_bilinear == True:
            self.x = tf.placeholder(tf.float32, [None, steps_x, dim_x])
        else:
            self.x = tf.placeholder(tf.float32, [None, dim_x])
        
        
        # --- prediction of individual models

        # models on individual feature groups
        mean_auto = bayesian_linear(self.auto, steps_auto, 'mean_v', True, self.dict_var, None, None, infer_type)
        
        if bool_bilinear == True:
            mean_x = bayesian_bilinear(self.x, [steps_x, dim_x], 'mean_distr', True, self.dict_var, None,\
                                       None, None, infer_type)
        else:
            mean_x = bayesian_linear(self.x, dim_x, 'mean_distr', True, self.dict_var, None, None, infer_type)

        # concatenate individual means 
        # [N, 2]
        mean_stack = tf.concat( [mean_auto, mean_x], 1 )
        
        
        # --- variance of individual models
        
        # variance depedent on features 
        varv = bayesian_linear(self.auto, steps_auto, 'sig_v', True, self.dict_var, None, None, infer_type)
        
        flatten_d = tf.reshape(self.x, [-1, steps_x*dim_x])
        vardistr = bayesian_linear(flatten_d, steps_x*dim_x, 'sig_distr', True, self.dict_var, None, None,\
                                   infer_type)
            
        var_auto = tf.square(varv)
        var_x = tf.square(vardistr)
        
        # concatenate individual variance 
        # [N, 2]
        var_stack = tf.concat( [var_auto, var_x], 1 )
        
        
        # --- gate weights of individual models
        
        # gate based on softmax        
        logit_auto = bayesian_linear(self.auto, steps_auto, 'gate_v', True, self.dict_var, None, None, infer_type)
        
        if bool_bilinear == True:
            logit_x = bayesian_bilinear(self.x, [steps_x, dim_x], 'gate_distr', True, self.dict_var, None, None,\
                                        None, infer_type)
        else:
            logit_x = bayesian_linear(self.x, dim_x, 'gate_distr', True, self.dict_var, None, None, infer_type)
        
        # [N, 2]
        self.logits = tf.concat([logit_auto, logit_x], 1)
        self.gates = tf.nn.softmax(self.logits)
        
        # --- bayesian mixture
        cat = ed.models.Categorical(logits = self.logits)
        components = [Normal(loc = loc, scale = scale) for loc, scale in zip(tf.unstack(tf.transpose(mean_stack)),\
                                                                         tf.unstack(tf.transpose(var_stack)))]
        # random variable
        self.bayes_y = ed.models.Mixture(cat=cat, components=components )
        
    
    #   initialize loss and optimization operations for training
    def train_ini(self):
        
        #self.inference = ed.ReparameterizationKLKLqp( latent_vars = self.dict_var, data = {self.bayes_y : self.y} )
        
        self.infer = ed.KLqp( latent_vars = self.dict_var, data = {self.bayes_y : self.y} )
            
        opti = tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE)
        
        self.infer.initialize( optimizer = opti, n_iter= self.num_iter, n_samples = self.loss_sample_num, \
                              scale={self.y : self.num_batch_per_epoch} )
        #scale={y: N / M}
        
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        
    
    # return loss
    def train_batch(self, auto_train, x_train, y_train, keep_prob):
        
        info_dict = self.infer.update({self.auto : auto_train, self.x : x_train, self.y : y_train, \
                                       self.keep_prob : keep_prob})
        
        '''
        # n_samples: int. Number of samples from variational model for calculating stochastic gradients.
        #inference.run(n_iter = 1, n_samples = 20)
        
        
        #ed.KLqp( )
        #ReparameterizationKLqp
        self.inference = ed.ReparameterizationKLKLqp(latent_vars = self.dict_var, data = {self.bayes_y : y_train,\
                                                       self.auto : v_train,\
                                                       self.x : distr_train, 
                                                       self.y : y_train, 
                                                       self.keep_prob : keep_prob})
            
        opti = tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE)
        self.inference.initialize(optimizer = opti)
        
        #self.inference.initialize(n_samples = 20)
        
        self.inference.run(n_iter = 1000, n_samples = 10)
        
        #{'t': t, 'loss': loss}
        #info_dict = self.inference.update()
        
        # ?
        #var_list = self.dict_var.values()
        #loss, grads_and_vars = inference.build_loss_and_gradients(var_list)
        #loss = -loss
        
        #print '----- training loss: ', info_dict['loss']
        #self.sess.run([loss], feed_dict = {self.auto:v_train, self.x:distr_train, self.y:y_train,\
        #                                   self.keep_prob:keep_prob })
        '''
        
        return 0
        
    def evaluate_ini(self):
        
        tmp_var_post = self.dict_var.values()
        dict_para_post = {}
        
        for item in tmp_var_post:
            dict_para_post.update({item.name: item})
        
        #['sig_distr/b/', 'mean_v/b/', 'gate_distr/w_l/', 'mean_distr/w_l/', 'gate_distr/w_r/', 'sig_v/b/', 'mean_distr/b/',\'mean_v/w/', 'gate_v/w/', 'mean_distr/w_r/', 'sig_v/w/', 'gate_v/b/', 'gate_distr/b/', 'sig_distr/w/']
        
        for i in range(self.eval_sample_num):
            
            auto_w = dict_para_post['mean_v/w/'].sample()
            auto_b = dict_para_post['mean_v/b/'].sample()
            
            d_w_l = dict_para_post['mean_distr/w_l/'].sample()
            d_w_r = dict_para_post['mean_distr/w_r/'].sample()
            d_b = dict_para_post['mean_distr/b/'].sample()
        
            gate_auto_w = dict_para_post['gate_v/w/'].sample()
            gate_auto_b = dict_para_post['gate_v/b/'].sample()
            
            gate_x_w_l = dict_para_post['gate_distr/w_l/'].sample()
            gate_x_w_r = dict_para_post['gate_distr/w_r/'].sample()
            gate_x_b = dict_para_post['gate_distr/b/'].sample()
            
            mean_auto = bayesian_linear(self.auto, 0, '', True, {}, tf.cast(auto_w, tf.float32), tf.cast(auto_b, tf.float32),\
                                        self.infer_type)
            
            mean_x = bayesian_bilinear(self.x, 0, '', True, {}, tf.cast(d_w_l, tf.float32), tf.cast(d_w_r, tf.float32),\
                                       tf.cast(auto_b, tf.float32), self.infer_type)
            
            logit_auto = bayesian_linear(self.auto, 0, '', True, {}, tf.cast(gate_auto_w, tf.float32), \
                                        tf.cast(gate_auto_b, tf.float32), self.infer_type)
            logit_x = bayesian_bilinear(self.x, 0, '', True, {}, tf.cast(gate_x_w_l, tf.float32), 
                                        tf.cast(gate_x_w_r, tf.float32), tf.cast(gate_x_b, tf.float32), self.infer_type)
            
            # [N, 2]
            logits = tf.concat([logit_auto, logit_x], 1)
            gates = tf.nn.softmax(logits)
            
            y_hat_sample = tf.reduce_sum(tf.concat([mean_auto, mean_x],1)*gates, 1, True)
            
            if i == 0:
                y_hat = y_hat_sample
            else:
                y_hat = tf.concat([y_hat, y_hat_sample], 1)
        
        # prediction
        self.y_hat = tf.reduce_sum(y_hat, 1)/self.eval_sample_num
        # ? TO DO 
        self.pre_auto = tf.squeeze(mean_auto) 
        self.pre_x = tf.squeeze(mean_x)
        
        # -- regularization place holder
        self.regu = tf.zeros([1,])
        
        # --- error metric
        # MSE
        self.mse  = tf.losses.mean_squared_error( self.y, self.y_hat )
        self.rmse = tf.sqrt(self.mse)
        # MAE
        self.mae  = tf.reduce_mean( tf.abs(self.y - self.y_hat) )
        # MAPE
        self.mape = tf.reduce_mean( tf.abs((self.y - self.y_hat)/(self.y+1e-10)) )
        
    
    #   infer givn testing data
    def inference(self, auto_test, x_test, y_test, keep_prob):
        
        return self.sess.run([self.rmse, self.mae, self.mape, self.regu], \
                             feed_dict = {self.auto:auto_test, self.x:x_test, self.y:y_test, self.keep_prob:keep_prob})
    
    #   predict givn testing data
    def predict(self, auto_test, x_test, keep_prob):
        
        return self.sess.run([self.y_hat, self.pre_auto, self.pre_x], feed_dict = {self.auto:auto_test, \
                                                       self.x:x_test, self.keep_prob:keep_prob })
    
    #   mixture gates givn testing data
    def predict_gates(self, auto_test, x_test, keep_prob):
        return self.sess.run( self.gates, feed_dict = {self.auto:auto_test, \
                                                        self.x:x_test, self.keep_prob:keep_prob })
    
    def model_reset(self):
        self.init = tf.global_variables_initializer()
        self.sess.run( self.init )
        
    # collect the optimized variable values w.r.t a keyword
    def collect_coeff_values(self, vari_keyword):
        return [ tf_var.name for tf_var in tf.trainable_variables() if (vari_keyword in tf_var.name) ],\
    [ tf_var.eval() for tf_var in tf.trainable_variables() if (vari_keyword in tf_var.name) ]
    
    def test(self, v_test, distr_test, y_test, keep_prob):
        
        return self.sess.run([tf.shape(self.logit)], feed_dict = {self.auto:v_test, \
                                                    self.x:distr_test,  self.y:y_test, self.keep_prob:keep_prob })
    
    
    
# ---- Posterior condition Bayesian variational mixture linear ----

class variational_mixture_linear_condition():
    
    def __init__(self, session, lr, l2, batch_size, steps_auto, dim_x, steps_x, bool_log, bool_bilinear, \
                 loss_type, distr_type, activation_type):
        
        # build the network graph 
        self.LEARNING_RATE = lr
                
        self.N_BATCH = batch_size
        self.L2 = l2
        
        self.MAX_NORM = 0.0
        self.epsilon = 1e-3
        
        self.sess = session
        
        self.bool_log = bool_log
        self.loss_type = loss_type
        self.distr_type = distr_type
        
        # initialize placeholders
        self.auto = tf.placeholder(tf.float32, [None, steps_auto])
        self.y = tf.placeholder(tf.float32, [None, ])
        self.keep_prob = tf.placeholder(tf.float32)
        
        if bool_bilinear == True:
            self.x = tf.placeholder(tf.float32, [None, steps_x, dim_x])
        else:
            self.x = tf.placeholder(tf.float32, [None, dim_x])
        
        
        # --- prediction of individual models

        # models on individual feature groups
        mean_v, regu_v_mean = linear(self.auto, steps_auto, 'mean_v', True)
        
        if bool_bilinear == True:
            mean_x, regu_x_mean = bilinear(self.x, [steps_x, dim_x], 'mean_distr', True)
        else:
            mean_x, regu_x_mean = linear(self.x, dim_x, 'mean_distr', True)

        # activation 
        if activation_type == 'relu':
            mean_x = tf.nn.relu(mean_x)
            mean_v = tf.nn.relu(mean_v)
            
        elif activation_type == 'leaky_relu':
            mean_x = tf.maximum( mean_x, 0.2*mean_x )
            mean_v = tf.maximum( mean_v, 0.2*mean_v )
        
        # concatenate individual means 
        self.mean_stack = tf.stack( [mean_v, mean_x], 1 )
        mean_stack = tf.stack( [mean_v, mean_x], 1 )
        
        # --- variance of individual models
        
        # variance depedent on features 
        varv, regu_v_var = linear(self.auto, steps_auto, 'sig_v', True)
        
        # only linear 
        vardistr, regu_x_var = linear( tf.reshape(self.x, [-1, steps_x*dim_x]), steps_x*dim_x, 'sig_distr', True)
        
        var_v = tf.square(varv)
        var_x = tf.square(vardistr)
        
        # concatenate individual variance 
        var_stack = tf.stack( [var_v, var_x], 1 )
        
        # variance constant for each expert
        #varv =    tf.Variable(tf.random_normal([1,], stddev = math.sqrt(2.0/float(1.0))) )
        #varreq =  tf.Variable(tf.random_normal([1,], stddev = math.sqrt(2.0/float(1.0))) )
        #vardistr= tf.Variable(tf.random_normal([1,], stddev = math.sqrt(2.0/float(1.0))) )
        #regu_sig = tf.nn.l2_loss(varv)+tf.nn.l2_loss(vardistr)
        
        # inverse variance standardize 
        #inv_var_v = tf.square(inv_varv)
        #inv_var_d = tf.square(inv_vardistr)
        
        
        # --- gate weights of individual models
        
        # logstic
        '''
        if bool_bilinear == True:
            self.logit, regu_gate = bilinear(self.x, [steps_x, dim_x], 'gate_x', True)
            
        else:
            self.logit, regu_gate = linear(self.x, dim_x, 'gate_x', True)
        
        self.gates = tf.stack( [1.0 - tf.sigmoid(self.logit), tf.sigmoid(self.logit)] ,1 )
        '''
        
        # softmax
        logit_v, regu_v_gate = linear(self.auto, steps_auto, 'gate_v', True)
        
        if bool_bilinear == True:
            # ?
            logit_x, regu_x_gate = bilinear(self.x, [steps_x, dim_x], 'gate_x', True)
            
        else:
            logit_x, regu_x_gate = linear(self.x, dim_x, 'gate_x', True)
        
        self.logit = tf.squeeze( tf.stack([logit_v, logit_x], 1) )
        self.gates = tf.nn.softmax(self.logit)
        
        
        # --- negative log likelihood 
        
        if distr_type == 'gaussian':
            tmpllk_v = tf.exp(-0.5*tf.square(self.y - mean_v)/(var_v + 1e-5))/(2.0*np.pi*(var_v + 1e-5))**0.5
            tmpllk_x = tf.exp(-0.5*tf.square(self.y - mean_x)/(var_x + 1e-5))/(2.0*np.pi*(var_x + 1e-5))**0.5
            
        elif distr_type == 'log':
            
            tmpllk_v = tf.exp(-0.5*tf.square(tf.log(self.y+1e-10) - mean_v)/(var_v + 1e-10))/(2.0*np.pi*(var_v + 1e-10))**0.5/(self.y+1e-10)
            
            tmpllk_x = tf.exp(-0.5*tf.square(tf.log(self.y+1e-10) - mean_x)/(var_x + 1e-10))/(2.0*np.pi*(var_x + 1e-10))**0.5/(self.y+1e-10)
        
        else:
            print '[ERROR] distribution type'
        
        llk = tf.multiply( (tf.stack([tmpllk_v, tmpllk_x], 1)), self.gates ) 
        self.neg_logllk = tf.reduce_sum( -1.0*tf.log(tf.reduce_sum(llk, 1)+1e-5) )
        
        
        # --- regularization
        
        # temporal coherence, diversity 
        
        # gate smoothness 
        logit_v_diff = logit_v[1:]-logit_v[:-1]
        logit_x_diff = logit_x[1:]-logit_x[:-1]
        
        def tf_var(tsr):
            return tf.reduce_mean(tf.square(tsr - tf.reduce_mean(tsr)))
        
        regu_gate_smooth = tf_var(logit_v_diff) + tf_var(logit_x_diff)  
        
        # gate diversity
        gate_diff = logit_v - logit_x
        regu_gate_diver = tf_var(gate_diff)
        
        # mean diversity
        regu_mean_diver = tf.reduce_mean( tf.maximum(0.0, (mean_v - self.y)*(mean_x - self.y)) )
        
        # mean non-negative  
        regu_mean_pos = tf.reduce_sum( tf.maximum(0.0, -1.0*mean_v) + tf.maximum(0.0, -1.0*mean_x) )
        
        
        if bool_bilinear == True:
            
            if loss_type == 'sq' and distr_type == 'gaussian':
                
                self.regu = 0.01*(regu_v_mean) + 0.001*(regu_x_mean[0]) + 0.001*(regu_x_mean[1])+\
                            0.001*(regu_v_gate) + 0.0001*(regu_x_gate[0] + regu_x_gate[1])\
                            + 0.001*regu_mean_pos\
                            #+ 0.0001*regu_mean_diver
                            
            elif loss_type == 'lk' and distr_type == 'gaussian':
                
                self.regu = 0.001*(regu_v_mean) + 0.001*(regu_x_mean[0]) + 0.001*(regu_x_mean[1])\
                        + 0.0001*(regu_v_gate) + 0.00001*(regu_x_gate[0] + regu_x_gate[1])\
                        + 0.00001*(regu_v_var + regu_x_var)
                        #+ 0.001*regu_mean_diver   
                
                # activation and hinge regularization 
                if activation_type == 'relu':
                    self.regu = self.regu
                
                elif activation_type == 'leaky_relu':
                    self.regu += 0.001*regu_mean_pos
                    
                elif activation_type == '':
                    self.regu += 0.001*regu_mean_pos
                
                elif activation_type == 'none':
                    self.regu = self.regu
                
                
                
            elif loss_type == 'sq' and distr_type == 'log':
                
                self.regu = 0.01*(regu_v_mean) + 0.001*(regu_x_mean[0]) + 0.00001*(regu_x_mean[1])+\
                        0.001*(regu_v_gate) + 0.00001*(regu_x_gate[0] + regu_x_gate[1])
            
            elif loss_type == 'lk' and distr_type == 'log':
                
                self.regu = 0.001*(regu_v_mean) + 0.001*(regu_x_mean[0]) + 0.001*(regu_x_mean[1])+\
                            0.0001*(regu_v_gate) + 0.0001*(regu_x_gate[0] + regu_x_gate[1])\
                            #+ 0.0001*regu_mean_diver
            else:
                print '[ERROR] loss type'
        
        else:
            
            if loss_type == 'sq' and distr_type == 'gaussian':
                
                self.regu = 0.01*regu_v_mean + 0.01*regu_x_mean\
                          + 0.0001*(regu_v_gate + regu_x_gate) - 0.001*regu_gate_diver
            
            elif loss_type == 'lk' and distr_type == 'gaussian':
                
                self.regu = 0.01*regu_v_mean + 0.0001*regu_x_mean + 0.0001*(regu_v_gate + regu_x_gate)
                        
            elif loss_type == 'sq' and distr_type == 'log':
                
                self.regu = 0.01*regu_v_mean + 0.0001*regu_x_mean + 0.0001*(regu_v_gate + regu_x_gate)
            
            elif loss_type == 'lk' and distr_type == 'log':
                
                self.regu = 0.001*regu_v_mean + 0.0001*regu_x_mean + 0.0001*(regu_v_gate + regu_x_gate)
            
            else:
                print '[ERROR] loss type'
        
        
        # --- mixture prediction, mean and variance
        
        if distr_type == 'gaussian':
            
            # mean and prediction
            self.y_hat = tf.reduce_sum(tf.multiply(mean_stack, self.gates), 1)
            
            self.pre_v = mean_v
            self.pre_x = mean_x
            
            # variance 
            sq_mean_stack =  var_stack - tf.square(mean_stack)
            mix_sq_mean = tf.reduce_sum(tf.multiply(sq_mean_stack, self.gates), 1)
            
            self.var_hat = mix_sq_mean - tf.square(self.y_hat)
            
        elif distr_type == 'log':
            
            # mean and prediction
            self.y_hat = tf.reduce_sum(tf.multiply(tf.exp(mean_stack), self.gates), 1)
            
            self.pre_v = tf.exp(mean_v)
            self.pre_x = tf.exp(mean_x)
            
            self.orig = mean_v
            
            # variance 
            # TO DO
            
        else:
            print '[ERROR] distribution type'
        
    
    def evaluate_ini(self):
        
        # --- error metric
        
        # RMSE
        self.mse  = tf.losses.mean_squared_error( self.y, self.y_hat )
        self.rmse = tf.sqrt(self.mse)
        
        # MAE
        self.mae  = tf.reduce_mean( tf.abs(self.y - self.y_hat) )
        
        # MAPE
        mask = tf.greater(tf.abs(self.y), 0.000001)
        
        y_mask = tf.boolean_mask(self.y, mask)
        y_hat_mask = tf.boolean_mask(self.y_hat, mask)
        
        self.mape = tf.reduce_mean( tf.abs((y_mask - y_hat_mask)/(y_mask+1e-10)) )
        
    def model_reset(self):
        self.init = tf.global_variables_initializer()
        self.sess.run( self.init )
        
#   initialize loss and optimization operations for training
    def train_ini(self):
        
        # loss
        if self.loss_type == 'sq':
            self.loss = self.mse + self.regu
            
        elif self.loss_type == 'lk':
            self.loss = self.neg_logllk + self.regu
        
        else:
            print '[ERROR] loss type'
        
        self.train = tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE)
        self.optimizer =  self.train.minimize(self.loss)
        
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        
        # !
        #self.optimizer = tf.train.ProximalAdagradOptimizer(learning_rate = 0.05, \
        #                                                  l2_regularization_strength = 0.0002, \
        #                                                  l1_regularization_strength = 0.005).minimize(self.lk_loss) 
        
        #self.optimizer = tf.train.FtrlOptimizer(learning_rate = 0.03, \
        #                                        l2_regularization_strength = 0.001, \
        #                                        l1_regularization_strength = 1.0).minimize(self.neg_logllk)
        
        
    #   training on batch of data
    def train_batch(self, auto_train, x_train, y_train, keep_prob ):
        
        # !
        _, c = self.sess.run([self.optimizer, self.neg_logllk],\
                             feed_dict={self.auto: auto_train, \
                                        self.x:x_train, self.y:y_train, self.keep_prob:keep_prob })
        return c
    
    #   infer givn testing data
    def inference(self, auto_test, x_test, y_test, keep_prob):
        
        return self.sess.run([self.rmse, self.mae, self.mape, self.regu], feed_dict = {self.auto:auto_test, \
                                                    self.x:x_test,  self.y:y_test, self.keep_prob:keep_prob })
    
    #   prediction
    def predict(self, auto_test, x_test, keep_prob):
        
        return self.sess.run( [self.y_hat, self.pre_v, self.pre_x], feed_dict = {self.auto:auto_test, \
                                                       self.x:x_test,  self.keep_prob:keep_prob })
    
    #   mixture gates
    def predict_gates(self, auto_test, x_test, keep_prob):
        return self.sess.run( self.gates , feed_dict = {self.auto:auto_test, \
                                                        self.x:x_test,  self.keep_prob:keep_prob })
    #   mixture logits
    def predict_logit(self, auto_test, x_test, keep_prob):
        return self.sess.run( self.logit , feed_dict = {self.auto:auto_test, \
                                                        self.x:x_test,  self.keep_prob:keep_prob })
    # collect the optimized variable values
    def collect_coeff_values(self, vari_keyword):
        return [ tf_var.name for tf_var in tf.trainable_variables() if (vari_keyword in tf_var.name) ],\
    [ tf_var.eval() for tf_var in tf.trainable_variables() if (vari_keyword in tf_var.name) ]
        
    def test(self, auto_train, x_train, y_train, keep_prob ):
        
        #return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mean_v')
        # [var.name for var in tf.trainable_variables()],
        return self.sess.run([tf.shape(self.mean_stack)],\
                             feed_dict={self.auto:auto_train, \
                                        self.x:x_train, self.y:y_train, self.keep_prob:keep_prob })

    
'''    
            elif infer_type == 'variational-condi':
                
                vari_w = tf.get_variable("vari_w", [dim_x, dim_x])
                vari_b = tf.get_variable("vari_b", [dim_x])
                
                qW = Normal(name = 'w', \
                            loc = tf.reduce_mean(tf.matmul(x, vari_w),0) + vari_b, \
                            scale = tf.ones([dim_x, 1]) )
                
                #tf.nn.softplus(tf.get_variable("qW/scale", [dim_x, 1])))
                qb = Normal(name = 'b', \
                            loc=tf.get_variable("qb/loc", [1,]), \
                            scale=tf.ones([1,]))
                    #scale=tf.nn.softplus(tf.get_variable("qb/scale", [1,])))
        
                dict_var.update({w:qW})
                dict_var.update({b:qb})


elif infer_type == 'variational-condi':
                
                # conditional on input
                # TO DO 
                
                vari_w_l = tf.get_variable("vari_w_l", [shape_one_x[1], 1])
                vari_b_l = tf.get_variable("vari_b_l", [shape_one_x[0], 1])
                
                vari_w_r = tf.get_variable("vari_w_r", [shape_one_x[0], 1])
                vari_b_r = tf.get_variable("vari_b_r", [shape_one_x[1], 1])
                
                
                qW_l = Normal(name = 'w_l', \
                              loc = tf.reduce_mean(tf.squeeze(tf.tensordot(x, vari_w_l, 1)),0) + vari_b_l, \
                              scale=tf.ones([shape_one_x[0], 1]))
                
                      #scale=tf.nn.softplus(tf.get_variable("qW_t/scale",[shape_one_x[0], 1])))
                qW_r = Normal(name = 'w_r', \
                              loc = tf.reduce_mean(tf.squeeze(tf.tensordot(tf.transpose(x,[0,2,1]),vari_w_r,1)),0)+vari_b_r,\
                              scale=tf.ones([shape_one_x[1], 1]) )
                      #scale=tf.nn.softplus(tf.get_variable("qW_v/scale", [shape_one_x[1], 1])))
                
                qb   = Normal(name = 'b', \
                              loc=tf.get_variable("qb/loc", [1,]),\
                              scale=tf.ones([1,]) )
                      #scale=tf.nn.softplus(tf.get_variable("qb/scale", [1,])))
                    
                
                dict_var.update({w_l : qW_l})
                dict_var.update({w_r : qW_r})
                dict_var.update({b : qb})
'''
# ---- Non-positive regularized Bayesian variational mixture linear ----
