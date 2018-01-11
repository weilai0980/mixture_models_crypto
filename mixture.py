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

# local packages
from utils_libs import *

'''
norm:

log  norm:

regu var: llk negative, err --

'''
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
        
        '''
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
        '''
        
        if bool_bias == True:
            h = tf.matmul(tmph, w_t) + tf.matmul(exter, w_ex) + b 
        else:
            h = tf.matmul(tmph, w_t) + tf.matmul(exter, w_ex) 
            
    # l2, l1 regularization
    
    #tf.nn.l2_loss(w_t) tf.reduce_sum(tf.abs(w_v)
    return tf.squeeze(h), [ tf.nn.l2_loss(w_t) + tf.nn.l2_loss(w_v), tf.nn.l2_loss(w_ex) ] 


# ---- Mixture linear NORMAL likelihood ----

class mixture_linear_lk():

    def __init__(self, session, lr, l2, batch_size, order_v, order_distr, order_steps, bool_log, bool_bilinear, \
                loss_type, distr_type):
        
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
        self.v_auto = tf.placeholder(tf.float32, [None, order_v])
        self.y = tf.placeholder(tf.float32, [None, ])
        self.keep_prob = tf.placeholder(tf.float32)
        
        if bool_bilinear == True:
            self.distr = tf.placeholder(tf.float32, [None, order_steps, order_distr])
        else:
            self.distr = tf.placeholder(tf.float32, [None, order_distr])
        
        tmp_d = tf.reshape(self.distr, [-1, order_steps*order_distr])
        
        
        # --- prediction of individual models

        # models on individual feature groups
        mean_v, regu_v_mean = linear_predict(self.v_auto, order_v, 'mean_v', True)
        
        if bool_bilinear == True:
            mean_distr, regu_d_mean = bilinear(self.distr, [order_steps, order_distr], 'mean_distr', True)
        
        else:
            mean_distr, regu_d_mean = linear_predict(self.distr, order_distr, 'mean_distr', True)

        # concatenate individual means 
        mean_stack = tf.stack( [mean_v, mean_distr], 1 )
        
        
        # --- variance of individual models
        
        # variance depedent on features 
        varv, regu_v_var = linear(self.v_auto, order_v, 'sig_v', True)
        
        if bool_bilinear == True:
            vardistr, regu_d_var = linear(tmp_d, order_steps*order_distr, 'sig_distr', True)
        else:
            vardistr, regu_d_var = linear(self.distr, order_distr, 'sig_distr', True)
            
        var_v = tf.square(varv)
        var_d = tf.square(vardistr)
        
        # concatenate individual variance 
        var_stack = tf.stack( [var_v, var_d], 1 )
        
        # variance constant for each expert
        #varv =    tf.Variable(tf.random_normal([1,], stddev = math.sqrt(2.0/float(1.0))) )
        #varreq =  tf.Variable(tf.random_normal([1,], stddev = math.sqrt(2.0/float(1.0))) )
        #vardistr= tf.Variable(tf.random_normal([1,], stddev = math.sqrt(2.0/float(1.0))) )
        #regu_sig = tf.nn.l2_loss(varv)+tf.nn.l2_loss(vardistr)
        
        # inverse variance standardize 
        #inv_var_v = tf.square(inv_varv)
        #inv_var_d = tf.square(inv_vardistr)
        
        
        # --- gate weights of individual models
        '''
        if bool_bilinear == True:
            
            self.logit, regu_gate = bilinear_with_external( self.distr, [order_steps, order_distr], \
                                                            self.v_auto, order_v, 'gate_distr', True )
            
            #bilinear(self.distr, [order_steps, order_distr], 'gate_distr', True)
            
        else:
            # ?
            self.logit, regu_gate = linear(self.distr, order_distr, 'gate_distr', True)
        
        self.gates = tf.stack( [tf.sigmoid(self.logit), 1.0 - tf.sigmoid(self.logit)] ,1 )
        '''
        
        # gate based on multimodal distribution        
        logit_v, regu_v_gate = linear(self.v_auto, order_v, 'gate_v', True)
        
        if bool_bilinear == True:
            # ?
            logit_d, regu_d_gate = bilinear(self.distr, [order_steps, order_distr], 'gate_distr', True)
            
        else:
            logit_d, regu_d_gate = linear(self.distr, order_distr, 'gate_distr', True)
        
        self.logit = tf.squeeze( tf.stack([logit_v, logit_d], 1) )
        self.gates = tf.nn.softmax(self.logit)
        
        
        # --- negative log likelihood 
        
        if distr_type == 'norm':
            tmpllk_v = tf.exp(-0.5*tf.square(self.y - mean_v)/(var_v+1e-5))/(2.0*np.pi*(var_v+1e-5))**0.5
            tmpllk_d = tf.exp(-0.5*tf.square(self.y - mean_distr)/(var_d+1e-5))/(2.0*np.pi*(var_d+1e-5))**0.5
            
        elif distr_type == 'log':
            
            tmpllk_v = tf.exp(-0.5*tf.square(tf.log(self.y+1e-10) - mean_v)/(var_v+1e-10))/(2.0*np.pi*(var_v+1e-10))**0.5/(self.y+1e-10)
            
            tmpllk_d = tf.exp(-0.5*tf.square(tf.log(self.y+1e-10) - mean_distr)/(var_d+1e-10))/(2.0*np.pi*(var_d+1e-10))**0.5/(self.y+1e-10)
        
        else:
            print '[ERROR] distribution type'
        
        llk = tf.multiply( (tf.stack([tmpllk_v, tmpllk_d], 1)), self.gates ) 
        self.neg_logllk = tf.reduce_sum( -1.0*tf.log(tf.reduce_sum(llk, 1)+1e-5) )
        
        # --- regularization
        
        # smoothness 
        logit_v_diff = logit_v[1:]-logit_v[:-1]
        logit_d_diff = logit_d[1:]-logit_d[:-1]
        
        def tf_var(tsr):
            return tf.reduce_mean(tf.square(tsr - tf.reduce_mean(tsr)))
        
        regu_gate_smooth = tf_var(logit_v_diff) + tf_var(logit_d_diff)  
        
        # gate diversity
        gate_diff = logit_v - logit_d
        regu_gate_diver = tf_var(gate_diff)
        
        # prediction diversity
        mean_diff = mean_v - mean_distr
        regu_pre_diver = tf_var(mean_diff)
        
        # positive mean 
        # TO DO 
        
        
        if bool_bilinear == True:
            
            if loss_type == 'sq' and distr_type == 'norm':
                
                self.regu = 0.01*(regu_v_mean) + 0.001*(regu_d_mean[0]) + 0.001*(regu_d_mean[1])+\
                            0.0001*(regu_v_gate) + 0.0001*(regu_d_gate[0] + regu_d_gate[1])\
                            #- 0.0001*regu_pre_diver\
                            #- 0.0001*regu_gate_diver
            
            elif loss_type == 'lk' and distr_type == 'norm':
                
                # for roll
                self.regu = 0.1*(regu_v_mean) + 0.001*(regu_d_mean[0]) + 0.001*(regu_d_mean[1])+\
                        0.0001*(regu_v_gate) + 0.0001*(regu_d_gate[0] + regu_d_gate[1])\
                        + 0.001*(regu_v_var + regu_d_var)
                
                # for one-shot
                #self.regu = 0.01*(regu_v_pre) + 0.001*(regu_d_pre[0]) + 0.00001*(regu_d_pre[1])+\
                #        0.001*(regu_v_gate) + 0.00001*(regu_d_gate[0] + regu_d_gate[1])\
                #        + 0.001*(regu_v_var + regu_d_var)
                        
                        
            elif loss_type == 'sq' and distr_type == 'log':
                
                self.regu = 0.01*(regu_v_mean) + 0.001*(regu_d_mean[0]) + 0.00001*(regu_d_mean[1])+\
                        0.001*(regu_v_gate) + 0.00001*(regu_d_gate[0] + regu_d_gate[1])
            
            
            elif loss_type == 'lk' and distr_type == 'log':
                
                self.regu = 0.1*(regu_v_mean) + 0.01*(regu_d_mean[0]) + 0.01*(regu_d_mean[1])+\
                        0.0001*(regu_v_gate) + 0.00001*(regu_d_gate[0] + regu_d_gate[1])
                
            
            else:
                print '[ERROR] loss type'
        
        else:
            
            if loss_type == 'sq' and distr_type == 'norm':
                
                self.regu = 0.01*regu_v_mean + 0.01*regu_d_mean\
                          + 0.0001*(regu_v_gate + regu_d_gate)\
                          - 0.0001*regu_pre_diver\
                          - 0.001*regu_gate_diver
                #+ 0.0001*regu_gate_smooth
            
            elif loss_type == 'lk' and distr_type == 'norm':
                
                self.regu = 0.01*regu_v_mean + 0.0001*regu_d_mean + 0.0001*(regu_v_gate + regu_d_gate)
                        
            elif loss_type == 'sq' and distr_type == 'log':
                
                self.regu = 0.01*regu_v_mean + 0.0001*regu_d_mean + 0.0001*(regu_v_gate + regu_d_gate)
            
            elif loss_type == 'lk' and distr_type == 'log':
                
                self.regu = 0.001*regu_v_mean + 0.0001*regu_d_mean + 0.0001*(regu_v_gate + regu_d_gate)
            
            else:
                print '[ERROR] loss type'
        
        
        # --- mixture prediction, mean and variance
        
        if distr_type == 'norm':
            
            # mean and prediction
            self.y_hat = tf.reduce_sum(tf.multiply(mean_stack, self.gates), 1)
            
            self.pre_v = mean_v
            self.pre_d = mean_distr
            
            # variance 
            sq_mean_stack =  var_stack - tf.square(mean_stack)
            mix_sq_mean = tf.reduce_sum(tf.multiply(sq_mean_stack, self.gates), 1)
            
            self.var_hat = mix_sq_mean - tf.square(self.y_hat)
            
        
        elif distr_type == 'log':
            
            # mean and prediction
            self.y_hat = tf.reduce_sum(tf.multiply(tf.exp(mean_stack), self.gates), 1)
            
            self.pre_v = tf.exp(mean_v)
            self.pre_d = tf.exp(mean_distr)
            
            self.orig = mean_v
            
            # variance 
            # TO DO
            
        else:
            print '[ERROR] distribution type'
        
        
        # --- mixture prediction errors
        
        # MSE
        self.err = tf.losses.mean_squared_error( self.y, self.y_hat )
        # MAPE
        #self.mape = tf.reduce_mean( tf.abs( (self.y - self.y_hat)/(self.y+1e-10) ) )
        
        '''
        if bool_log == True:
            # ?
            self.y_hat = tf.reduce_sum(tf.multiply(tf.exp(pre), self.gates), 1)
            #self.y_hat = tf.exp(tf.reduce_sum(tf.multiply(pre, self.gates), 1))
            
            self.pre_v = tf.exp(pre_v)
            self.pre_d = tf.exp(pre_distr)
            
        else:
            self.y_hat = tf.reduce_sum( tf.multiply(pre, self.gates), 1 )
            
            self.pre_v = pre_v
            self.pre_d = pre_distr
         '''   
    
    def model_reset(self):
        self.init = tf.global_variables_initializer()
        self.sess.run( self.init )
        
#   initialize loss and optimization operations for training
    def train_ini(self):
        
        # loss
        if self.loss_type == 'sq':
            self.lk_loss = self.err + self.regu
            
        elif self.loss_type == 'lk':
            self.lk_loss = self.neg_logllk + self.regu
        
        else:
            print '[ERROR] loss type'
        
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.lk_loss)
        
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
    def train_batch(self, v_train, distr_train, y_train, keep_prob ):
        
        # !
        _, c = self.sess.run([self.optimizer, self.neg_logllk],\
                             feed_dict={self.v_auto:v_train, \
                                        self.distr:distr_train, self.y:y_train, self.keep_prob:keep_prob })
        return c
    
    #   infer givn testing data
    def inference(self, v_test, distr_test, y_test, keep_prob):
        
        return self.sess.run([self.err, self.regu], feed_dict = {self.v_auto:v_test, \
                                                    self.distr:distr_test,  self.y:y_test, self.keep_prob:keep_prob })
    
    #   predict givn testing data
    def predict(self, v_test, distr_test, keep_prob):
        
        return self.sess.run( [self.y_hat, self.pre_v, self.pre_d], feed_dict = {self.v_auto:v_test, \
                                                       self.distr:distr_test,  self.keep_prob:keep_prob })
    
    #   mixture gates givn testing data
    def predict_gates(self, v_test, distr_test, keep_prob):
        return self.sess.run( self.gates , feed_dict = {self.v_auto:v_test, \
                                                        self.distr:distr_test,  self.keep_prob:keep_prob })
    
    def predict_logit(self, v_test, distr_test, keep_prob):
        return self.sess.run( self.logit , feed_dict = {self.v_auto:v_test, \
                                                        self.distr:distr_test,  self.keep_prob:keep_prob })
    # collect the optimized variable values
    def collect_coeff_values(self, vari_keyword):
        return [ tf_var.name for tf_var in tf.trainable_variables() if (vari_keyword in tf_var.name) ],\
    [ tf_var.eval() for tf_var in tf.trainable_variables() if (vari_keyword in tf_var.name) ]
        
        
