#!/usr/bin/python

from utils_libs import *

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


# TO DO:
# max norm constraints

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

# ---- utilities functions ----

def linear( x, dim_x, scope ):
    
    with tf.variable_scope(scope):
        w = tf.Variable( tf.random_normal([dim_x, 1], stddev = math.sqrt(2.0/float(dim_x))) )
        b = tf.Variable( tf.zeros([1,]))
                
        h = tf.matmul(x, w) + b
            
        #l2
        regularizer = tf.nn.l2_loss(w)
            
        #l1
        regularizer = 0.5*tf.reduce_sum(tf.abs(w)) + 0.5*tf.nn.l2_loss(w)
    
    return tf.squeeze(h), regularizer

def linear_predict( x, dim_x, scope ):
    
    with tf.variable_scope(scope):
        w = tf.Variable( tf.random_normal([dim_x, 1], stddev = math.sqrt(2.0/float(dim_x))) )
        b = tf.Variable( tf.zeros([1,]))
        
        b1 = tf.Variable( tf.zeros([1,]))
        
        h = tf.matmul(x, w) + b
            
        #l2
        regularizer = tf.nn.l2_loss(w)
            
        #l1
        regularizer = 0.5*tf.reduce_sum(tf.abs(w)) + 0.5*tf.nn.l2_loss(w)
    
    return tf.squeeze(h), regularizer
    

def plain_dense( x, x_dim, dim_layers, scope, dropout_keep_prob):
    
    with tf.variable_scope(scope):
        # initilization
        w = tf.get_variable('w', [x_dim, dim_layers[0]], dtype=tf.float32,\
                                    initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.zeros([dim_layers[0]]))
        h = tf.nn.relu( tf.matmul(x, w) + b )

        regularization = tf.nn.l2_loss(w)
                
        #dropout
        h = tf.nn.dropout(h, dropout_keep_prob)
        
    for i in range(1, len(dim_layers)):
        with tf.variable_scope(scope+str(i)):
            w = tf.get_variable('w', [dim_layers[i-1], dim_layers[i]],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.zeros( dim_layers[i] ))
            h = tf.nn.relu( tf.matmul(h, w) + b )
                
            regularization += tf.nn.l2_loss(w)
        
    return h, regularization

# ---- Mixture Linear ----

class mixture_linear():
    
    def __init__(self, session, lr, l2, batch_size, order_v, order_req, order_distr ):
        
        # build the network graph 
        self.LEARNING_RATE = lr
                
        self.N_BATCH = batch_size
        self.L2 = l2
   
        self.MAX_NORM = 0.0
        self.epsilon = 1e-3
        
        self.sess = session
        
        # initialize placeholders
        self.v_pre = tf.placeholder(tf.float32, [None, order_v])
        self.req   = tf.placeholder(tf.float32, [None, order_req])
        self.distr = tf.placeholder(tf.float32, [None, order_distr])
        self.y     = tf.placeholder(tf.float32, [None, ])
        
        self.keep_prob = tf.placeholder(tf.float32)

        # models on individual feature groups
        pre_v, regular_v     = linear_predict(self.v_pre, order_v, 'v_pre' )
        #pre_req, regular_req = linear(self.req, order_req, 'req' )
        pre_distr, regular_distr = linear_predict(self.distr, order_distr, 'distr')

        # mixing process
        pre = tf.stack( [pre_v, pre_distr], 1 )  
        #pre = tf.squeeze( pre )  
        
        logit_v, regu_v = linear(self.v_pre, order_v,    'gate_v' )
        #logit_req, regu_r = linear(self.req,   order_req,  'gate_req' )
        logit_distr, regu_d = linear(self.distr, order_distr, 'gate_distr')


        
        concat_x = tf.concat([self.v_pre, self.distr], 1)
        logit_c, regu_c = linear( concat_x, order_v + order_distr, 'logit_c' )
        
        #self.tmp_shape = tf.shape(self.v_pre)
        #ins_num = tmp_shape[0]
        #logit_distr = tf.constant(1.0, shape=[ins_num,])
        
        
        
        # depedent on features 
        varv, regu_vv =   linear(self.v_pre, order_v,   'sig_v')
        #varreq, regu_vr = linear(self.req,   order_req, 'sig_req')
        vardistr, regu_vd =linear(self.distr, order_distr,'sig_distr')
        
        
        # constant for each expert
        #varv =    tf.Variable(tf.random_normal([1,], stddev = math.sqrt(2.0/float(1.0))) )
        #varreq =  tf.Variable(tf.random_normal([1,], stddev = math.sqrt(2.0/float(1.0))) )
        #vardistr= tf.Variable(tf.random_normal([1,], stddev = math.sqrt(2.0/float(1.0))) )
        #regu_sig = tf.nn.l2_loss(varv)+tf.nn.l2_loss(vardistr)
        
        var_v = tf.square(varv)
        #var_req = tf.exp(varreq)
        var_distr = tf.square(vardistr)
        
        self.logit = tf.squeeze( tf.stack( [logit_v, logit_distr], 1 ) )
        self.gates = tf.nn.softmax(self.logit)
        
        tmp_gate = tf.sigmoid( tf.squeeze(logit_c) )
        self.gates = tf.stack( [tmp_gate, 1.0-tmp_gate],1 )
        
        #--- boosting
        concat_x = self.v_pre
        logit_c, regu_c = linear( concat_x, order_v , 'logit_c' )
        
        tmp_gate = tf.sigmoid( tf.squeeze(logit_c) )
        self.gates = tf.stack( [tmp_gate, 1.0-tmp_gate],1 )
        #---
        
        #negative log likelihood
        tmpllk_v     = tf.exp(-0.5 * tf.square(self.y- tf.squeeze(pre_v) )/(var_v**2+1e-5))/(2.0*np.pi*(var_v**2+1e-5))**0.5
        #tmpllk_req   = tf.exp(-0.5 * tf.square(self.y- tf.squeeze(pre_req) )/(var_req**2+1e-5))/(2.0*np.pi*(var_req**2+1e-5))**0.5
        tmpllk_distr = tf.exp(-0.5 * tf.square(self.y- tf.squeeze(pre_distr) )/(var_distr**2+1e-5))/(2.0*np.pi*(var_distr**2+1e-5))**0.5
        
        llk = tf.multiply( (tf.stack([tmpllk_v, tmpllk_distr], 1)), self.gates ) 
        
        self.neg_logllk = tf.reduce_sum( -1.0*tf.log( tf.reduce_sum(llk, 1)+1e-5 ) )
        
        
        # mixed prediction
        # mixed prediction
        self.y_hat = pre_v + tmp_gate*pre_distr 
        
        
        # regularization
        self.regu = self.L2*(regular_v + regular_distr  )
         #regu_v + regu_d + regu_vv + regu_vd
    
    def test_func(self, v_train, req_train, distr_train, y_train ):
        
        res = self.sess.run([self.test],\
                             feed_dict={self.v_pre:v_train, self.req:req_train,\
                                        self.distr:distr_train, self.y:y_train })
        return res
    
#   initialize loss and optimization operations for training
    def train_ini(self):
        
        # loss: mixed likelihood 
        self.lk_loss = self.neg_logllk + self.regu
        
        # loss: mixed prediction
        self.err = tf.losses.mean_squared_error( self.y, self.y_hat )        
        self.loss = self.err + self.regu
        
        # !
        self.optimizer = \
        tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.loss)
        
        
        self.init = tf.global_variables_initializer()
        self.sess.run( self.init )
        
        
    #   training on batch of data
    def train_batch(self, v_train, req_train, distr_train, y_train, keep_prob ):
        
        # !
        _,c = self.sess.run([self.optimizer, self.loss],\
                             feed_dict={self.v_pre:v_train, self.req:req_train,\
                                        self.distr:distr_train, self.y:y_train, self.keep_prob:keep_prob })
        return c
    
    #   infer givn testing data
    def inference(self, v_test, req_test, distr_test, y_test, keep_prob):
        return self.sess.run([self.err, self.regu], feed_dict = {self.v_pre:v_test,      self.req:req_test,\
                                                      self.distr:distr_test,  self.y:y_test, self.keep_prob:keep_prob })
    #   predict givn testing data
    def predict(self, v_test, req_test, distr_test, keep_prob):
        return self.sess.run( self.y_hat, feed_dict = {self.v_pre:v_test,      self.req:req_test,\
                                                       self.distr:distr_test,  self.keep_prob:keep_prob })
    
     #   predict givn testing data
    def predict_gates(self, v_test, req_test, distr_test, keep_prob):
        return self.sess.run( self.gates , feed_dict = {self.v_pre:v_test,      self.req:req_test,\
                                                        self.distr:distr_test,  self.keep_prob:keep_prob })
    
    def predict_logit(self, v_test, req_test, distr_test, keep_prob):
        return self.sess.run( self.logit , feed_dict = {self.v_pre:v_test,      self.req:req_test,\
                                                        self.distr:distr_test,  self.keep_prob:keep_prob })
    
    
# ---- Mixture MLP ---- 

class neural_mixture_dense():
    
    def dense_layers(self, dim_layers, x, dim_x, dim_output, scope, dropout_keep_rate):
        
        with tf.variable_scope(scope + str(0)):
                w = tf.Variable( tf.random_normal([dim_x, dim_layers[0]], stddev = math.sqrt(2.0/float(dim_x))) )
                b = tf.Variable( tf.zeros([dim_layers[0]]))
                
                h = tf.nn.relu( tf.matmul(x, w) + b )
                regularizer = tf.nn.l2_loss(w)
        
        # dropout
        h = tf.nn.dropout(h, dropout_keep_rate)

        for i in range(1, len(dim_layers)):
            
            with tf.variable_scope(scope + str(i)):
                
                tmph = h
                
                w = tf.Variable( tf.random_normal([dim_layers[i-1], dim_layers[i]], \
                                                  stddev = math.sqrt(2.0/float(dim_layers[i-1]))) )
                b = tf.Variable( tf.zeros([dim_layers[i]]))
                
                #residual connection
                h = tf.nn.relu( tf.matmul(h, w) + b )
                #h = tmph + h
                
                # L2  
                regularizer += tf.nn.l2_loss(w)    
        
        with tf.variable_scope(scope + "output"):
            w = tf.Variable( tf.random_normal([dim_layers[-1], dim_output ],stddev = math.sqrt(2.0/float(dim_layers[-1]))) )
            b = tf.Variable( tf.zeros([1,]))
            
            output = tf.matmul(h, w) + b
            # L2  
            #regularizer += tf.nn.l2_loss(w) 
        
        return tf.squeeze(output), h, regularizer
    
    
    def __init__(self, session, hidden_dims, lr, l2, batch_size, order_v, order_req, order_distr ):
        
        # build the network graph 
        self.LEARNING_RATE = lr
                
        self.N_BATCH = batch_size
        self.L2 = l2
        
        self.HIDDEN_DIMS = hidden_dims
   
        self.MAX_NORM = 0.0
        self.epsilon = 1e-3
        
        self.sess = session
        
        # initialize placeholders
        self.v_pre = tf.placeholder(tf.float32, [None, order_v])
        self.req   = tf.placeholder(tf.float32, [None, order_req])
        self.distr = tf.placeholder(tf.float32, [None, order_distr])
        self.y     = tf.placeholder(tf.float32, [None, ])
        
        self.keep_prob = tf.placeholder(tf.float32)
        
        # models on individual feature groups
        pre_v, h_v, regular_v = self.dense_layers(self.HIDDEN_DIMS[0], self.v_pre, order_v, 1, 'v_pre', self.keep_prob )
        pre_req, h_req, regular_req = self.dense_layers(self.HIDDEN_DIMS[1], self.req, order_req, 1, 'req',self.keep_prob )
        pre_distr, h_distr, regular_distr = self.dense_layers(self.HIDDEN_DIMS[2], self.distr, order_distr, 1,\
                                                              'distr', self.keep_prob )

        # regularization 
        self.regu = regular_v + regular_req + regular_distr
        
        # concatenate individiual predictive mean of each expert
        pre = tf.stack( [pre_v, pre_distr], 1 )  
        pre = tf.squeeze( pre )  
        
        # explicit feature + hidden feature
        #hf_v = tf.concat([ self.v_pre, h_v ],1 )
        #hf_r = tf.concat([ self.req,   h_req ],1 )
        #hf_d = tf.concat([ self.distr, h_distr ],1 )
        
        hf_v = self.v_pre
        #hf_r = self.req
        hf_d = self.distr
        
        '''
        logit_v, regu_v = linear(self.v_pre, order_v,    'gate_v' )
        logit_req, regu_r = linear(self.req,   order_req,  'gate_req' )
        logit_distr, regu_d = linear(self.distr, order_distr, 'gate_distr')
        
        self.regu += (tf.nn.l2_loss(regu_v)+tf.nn.l2_loss(regu_r)+tf.nn.l2_loss(regu_d)) 
        '''
        
        '''
            #wt1 = tf.Variable( tf.random_normal([self.HIDDEN_DIMS[0][-1]+order_v, tmpdim], \
            #                                  stddev = math.sqrt(2.0/float(self.HIDDEN_DIMS[0][-1]+order_v))) )
            #wt2 = tf.Variable( tf.random_normal([self.HIDDEN_DIMS[1][-1]+order_req, tmpdim], \
            #                                  stddev = math.sqrt(2.0/float(self.HIDDEN_DIMS[1][-1]+order_req))) )
            #wt3 = tf.Variable( tf.random_normal([self.HIDDEN_DIMS[2][-1]+order_distr, tmpdim], \
            #                                  stddev = math.sqrt(2.0/float(self.HIDDEN_DIMS[2][-1]+order_distr))) )
        '''
        
        concat_x = tf.concat( [self.v_pre, self.distr], 1 )
        concat_dim = order_v + order_distr
        
        concat_mix_para, concat_h, regu = self.dense_layers( self.HIDDEN_DIMS[3], concat_x, concat_dim, 4, 'gate',\
                                                             self.keep_prob )
        self.regu += regu
        
        self.logit, varv, vard = tf.split( concat_mix_para, [2,1,1], 1 )
        
        
        '''
        # deriving gate weights
        with tf.variable_scope("gate"):          

            tmpdim = 3
            
            wt1 = tf.Variable( tf.random_normal([order_v, tmpdim], \
                                              stddev = math.sqrt(2.0/float(self.HIDDEN_DIMS[0][-1]+order_v))) )
            wt2 = tf.Variable( tf.random_normal([order_req, tmpdim], \
                                              stddev = math.sqrt(2.0/float(self.HIDDEN_DIMS[1][-1]+order_req))) )
            wt3 = tf.Variable( tf.random_normal([order_distr, tmpdim], \
                                              stddev = math.sqrt(2.0/float(self.HIDDEN_DIMS[2][-1]+order_distr))) )
            
            b1 = tf.Variable( tf.zeros([tmpdim, ]))
            b2 = tf.Variable( tf.zeros([tmpdim, ]))
            b3 = tf.Variable( tf.zeros([tmpdim, ]))
            self.regu += (tf.nn.l2_loss(wt1)+tf.nn.l2_loss(wt2)+tf.nn.l2_loss(wt3))  
            
            
            wa = tf.Variable( tf.random_normal([tmpdim, 1], \
                                              stddev = math.sqrt(2.0/float(tmpdim))) )
            ba = tf.Variable( tf.zeros([1, ]))
            self.regu += tf.nn.l2_loss(wa)
            
            logit_v = tf.squeeze(     tf.matmul( tf.tanh( tf.matmul(hf_v, wt1) + b1 ), wa)+ba )
            logit_req = tf.squeeze(   tf.matmul( tf.tanh( tf.matmul(hf_r, wt2) + b2 ), wa)+ba )
            logit_distr = tf.squeeze( tf.matmul( tf.tanh( tf.matmul(hf_d, wt3) + b3 ), wa)+ba )
            
            self.logit = tf.stack( [logit_v, logit_req, logit_distr], 1 )
        '''
        
        # gates 
        self.gates = tf.nn.softmax(self.logit)
        
        # mixed prediction
        self.y_hat = tf.reduce_sum( tf.multiply(pre, self.gates), 1 ) 
         
        
        #negative log likelihood
        '''
        varv,_,regu = self.dense_layers(self.HIDDEN_DIMS[3], hf_v, self.HIDDEN_DIMS[0][-1]+order_v, 'sig_v', self.keep_prob )
        self.regu += regu
        
        varr,_,regu = self.dense_layers(self.HIDDEN_DIMS[3], hf_r, self.HIDDEN_DIMS[1][-1]+order_req, 'sig_r', self.keep_prob )
        self.regu += regu

        vard,_,regu=self.dense_layers(self.HIDDEN_DIMS[3], hf_d, self.HIDDEN_DIMS[2][-1]+order_distr, 'sig_d', self.keep_prob )
        self.regu += regu
        '''
        # constant for each expert
        #varv = tf.Variable(tf.random_normal([1,], stddev = math.sqrt(2.0/float(1.0))) )
        #varr = tf.Variable(tf.random_normal([1,], stddev = math.sqrt(2.0/float(1.0))) )
        #vard = tf.Variable(tf.random_normal([1,], stddev = math.sqrt(2.0/float(1.0))) )
        #regu_sig = tf.nn.l2_loss(varv)+tf.nn.l2_loss(varr)+tf.nn.l2_loss(vard)
        #self.regu += regu_sig
        '''
        varv,_,regu = self.dense_layers(self.HIDDEN_DIMS[3], hf_v, order_v, 'sig_v', self.keep_prob )
        self.regu += regu
        
        varr,_,regu = self.dense_layers(self.HIDDEN_DIMS[3], hf_r, order_req, 'sig_r', self.keep_prob )
        self.regu += regu

        vard,_,regu=self.dense_layers(self.HIDDEN_DIMS[3], hf_d, order_distr, 'sig_d', self.keep_prob )
        self.regu += regu
        '''
        
        sd_v = tf.squeeze(tf.square(varv))
        #sd_r = tf.squeeze(tf.square(varr))
        sd_d = tf.squeeze(tf.square(vard))
                          
        tmpllk_v = tf.exp(-0.5*tf.square(self.y-tf.squeeze(pre_v))/(sd_v**2+1e-5))/(2.0*np.pi*(sd_v**2+1e-5))**0.5
        #tmpllk_req = tf.exp(-0.5*tf.square(self.y-tf.squeeze(pre_req))/(sd_r**2+1e-5))/(2.0*np.pi*(sd_r**2+1e-5))**0.5
        tmpllk_distr = tf.exp(-0.5*tf.square(self.y-tf.squeeze(pre_distr))/(sd_d**2+1e-5))/(2.0*np.pi*(sd_d**2+1e-5))**0.5
        
        llk = tf.multiply( tf.squeeze(tf.stack([tmpllk_v, tmpllk_distr], 1)), self.gates ) 
        self.neg_logllk = tf.reduce_sum( -1.0*tf.log( tf.reduce_sum(llk, 1)+1e-5 ) )
        
        #test
        self.test1 =  tf.shape( llk )
        self.test2 =  tf.shape( self.neg_logllk )
    
    def test(self, v_train, req_train, distr_train, y_train, keep_prob ):
        res = self.sess.run([self.test1, self.test2 ],\
                             feed_dict={self.v_pre:v_train, self.req:req_train,\
                                        self.distr:distr_train, self.y:y_train, self.keep_prob:keep_prob })
        return res
    
#   initialize loss and optimization operations for training
    def train_ini(self):
        
        # negative log likelihood
        self.loss = self.neg_logllk + self.L2*self.regu
        self.weight_regu = self.L2*self.regu
        
        # squared error
        self.err = tf.losses.mean_squared_error( self.y, self.y_hat )        
        #self.loss = self.err + self.L2*self.regu
        
        # !
        self.optimizer = \
        tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.loss)
#         tf.train.AdadeltaOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)
#         tf.train.RMSPropOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)
#         tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(self.cost)
#         tf.train.AdadeltaOptimizer(learning_rate = self.lr).minimize(self.cost)
#         tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.cost)
#         tf.train.GradientDescentOptimizer(learning_rate = self.lr).minimize(self.cost)
        
        self.init = tf.global_variables_initializer()
        self.sess.run( self.init )
        
        
    #   training on batch of data
    def train_batch(self, v_train, req_train, distr_train, y_train, keep_prob ):
        
        # !
        _,c = self.sess.run([self.optimizer, self.loss],\
                             feed_dict={self.v_pre:v_train, self.req:req_train,\
                                        self.distr:distr_train, self.y:y_train, self.keep_prob:keep_prob })
        return c
    
    
    #   infer givn testing data
    def inference(self, v_test, req_test, distr_test, y_test, keep_prob):
        
        return self.sess.run([self.err, self.weight_regu],feed_dict = {self.v_pre:v_test,      self.req:req_test,\
                                                      self.distr:distr_test,  self.y:y_test, self.keep_prob:keep_prob })
    #   predict givn testing data
    def predict(self, v_test, req_test, distr_test, keep_prob):
        return self.sess.run( self.y_hat, feed_dict = {self.v_pre:v_test,      self.req:req_test,\
    
                                                       self.distr:distr_test,  self.keep_prob:keep_prob })
    #   predict givn testing data
    def predict_gates(self, v_test, req_test, distr_test, keep_prob):
        return self.sess.run( self.gates , feed_dict = {self.v_pre:v_test,      self.req:req_test,\
                                                        self.distr:distr_test,  self.keep_prob:keep_prob })
    
    def predict_logit(self, v_test, req_test, distr_test, keep_prob):
        return self.sess.run( self.logit , feed_dict = {self.v_pre:v_test,      self.req:req_test,\
                                                        self.distr:distr_test,  self.keep_prob:keep_prob })
    
# ---- Plain MLP consuming concatenated features ----    
class neural_plain_mlp():
    
    def dense_layers(self, dim_layers, x, dim_x, scope, dropout_keep_rate):
        
        with tf.variable_scope(scope + str(0)):
                w = tf.Variable( tf.random_normal([dim_x, dim_layers[0]], stddev = math.sqrt(2.0/float(dim_x))) )
                b = tf.Variable( tf.zeros([dim_layers[0]]))
                
                h = tf.nn.relu( tf.matmul(x, w) + b )
                regularizer = 0.5*tf.reduce_sum(tf.abs(w)) + 0.5*tf.nn.l2_loss(w)
        
        # dropout
        h = tf.nn.dropout(h, dropout_keep_rate)

        for i in range(1, len(dim_layers)):
            
            with tf.variable_scope(scope + str(i)):
                
                tmp = h
                
                w = tf.Variable( tf.random_normal([dim_layers[i-1], dim_layers[i]], \
                                                  stddev = math.sqrt(2.0/float(dim_layers[i-1]))) )
                b = tf.Variable( tf.zeros([dim_layers[i]]))

                #residual 
                h = tf.nn.relu( tf.matmul(h, w) + b ) + tmp
                # L2  
                regularizer += (0.5*tf.reduce_sum(tf.abs(w)) + 0.5*tf.nn.l2_loss(w))    
        
        with tf.variable_scope(scope + "output"):
            w = tf.Variable( tf.random_normal([dim_layers[-1], 1 ],stddev = math.sqrt(2.0/float(dim_layers[-1]))) )
            b = tf.Variable( tf.zeros([1,]))
            
            output = tf.matmul(h, w) + b
            
            # L2  
            regularizer += tf.nn.l2_loss(w)
        
        return output, h, regularizer
    
    
    def __init__(self, session, hidden_dims, lr, l2, batch_size, dim_x ):
        
        # build the network graph 
        self.LEARNING_RATE = lr
                
        self.N_BATCH = batch_size
        self.L2 = l2
        
        self.HIDDEN_DIMS = hidden_dims
   
        self.MAX_NORM = 0.0
        self.epsilon = 1e-3
        
        self.sess = session
        
        # initialize placeholders
        self.x = tf.placeholder(tf.float32, [None, dim_x])
        self.y = tf.placeholder(tf.float32, [None, ])
        
        self.keep_prob = tf.placeholder(tf.float32)
        
        # models on individual feature groups
        y_hat, h, regular = self.dense_layers(self.HIDDEN_DIMS, self.x, dim_x, 'mlp', self.keep_prob )
        
        #  prediction and regularization 
        self.y_hat = tf.squeeze(y_hat) 
        self.regu  = regular
        
        
#   initialize loss and optimization operations for training
    def train_ini(self):
        
        self.err = tf.losses.mean_squared_error( self.y, self.y_hat)        
        self.loss = self.err + self.L2*self.regu
            
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.loss)
        
        self.init = tf.global_variables_initializer()
        self.sess.run( self.init )
        
    #   training on batch of data
    def train_batch(self, x_train, y_train, keep_prob ):
       
        _,c = self.sess.run([self.optimizer, self.loss],\
                             feed_dict={self.x:x_train, self.y:y_train, self.keep_prob:keep_prob })
        return c
    
    #   infer givn testing data
    def inference(self, x_test, y_test, keep_prob):
        return self.sess.run([self.err], feed_dict = {self.x:x_test, self.y:y_test, self.keep_prob:keep_prob })
    
    #   predict givn testing data
    def predict(self, x_test, keep_prob):
        return self.sess.run( self.y_hat, feed_dict = {self.x:x_test, self.keep_prob:keep_prob })
    
    
# ---- Mixture LSTM ----

def _linear(args,
            output_size,
            bias,
            bias_initializer=None,
            kernel_initializer=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
        dtype=dtype,
        initializer=kernel_initializer)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      if bias_initializer is None:
        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
      biases = vs.get_variable(
          _BIAS_VARIABLE_NAME, [output_size],
          dtype=dtype,
          initializer=bias_initializer)
    return nn_ops.bias_add(res, biases)

def ln(tensor, scope = None, epsilon = 1e-5):
    """ Layer normalizes a 2D tensor along its second axis """
    assert(len(tensor.get_shape()) == 2)
    m, v = tf.nn.moments(tensor, [1], keep_dims=True)
    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'layer_norm'):
        scale = tf.get_variable('scale',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    LN_initial = (tensor - m) / tf.sqrt(v + epsilon)

    return LN_initial * scale + shift

class LayerNormalizedLSTMCell(tf.nn.rnn_cell.RNNCell):
    """
    Adapted from TF's BasicLSTMCell to use Layer Normalization.
    Note that state_is_tuple is always True.
    """

    def __init__(self, num_units, forget_bias=1.0, activation=tf.nn.tanh):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            # change bias argument to False since LN will add bias via shift
            concat = _linear([inputs, h], 4 * self._num_units, False)

            i, j, f, o = tf.split(concat, 4, 1 )
            

            # add layer normalization to each gate
            i = ln(i, scope = 'i/')
            j = ln(j, scope = 'j/')
            f = ln(f, scope = 'f/')
            o = ln(o, scope = 'o/')

            new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) *
                   self._activation(j))

            # add layer_normalization in calculation of new hidden state
            new_h = self._activation(ln(new_c, scope = 'new_h/')) * tf.nn.sigmoid(o)
            new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

            return new_h, new_state
        
#layer normalization
#https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
def ln_lstm_stacked( x, dim_layers, scope ):
    
    with tf.variable_scope(scope):
        #Deep lstm: residual or highway connections 
        #block: residual connections
        
        #tf.nn.rnn_cell.LSTMCell
        lstm_cell = LayerNormalizedLSTMCell(dim_layers[0])
                                                #initializer= tf.contrib.keras.initializers.glorot_normal())
        hiddens, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = x, dtype = tf.float32)
            
    for i in range(1,len(dim_layers)):
        with tf.variable_scope(scope+str(i)):
            
            lstm_cell = LayerNormalizedLSTMCell(dim_layers[i])
                                                   # initializer= tf.contrib.keras.initializers.glorot_normal())
            hiddens, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = hiddens, dtype = tf.float32)
    
    # return hidden states on each time step
    return hiddens 


def plain_lstm_split( x, dim_x, dim_layers, scope):
    
    indivi_ts = tf.split(x, num_or_size_splits = dim_x, axis = 2)
    concat_h  = []
    
    for i in range( dim_x ):
        
        current_x = indivi_ts[i]
        h, _  = plain_lstm_stacked( current_x, dim_layers, scope+str(i))
        # obtain the last hidden state    
        tmp_hiddens = tf.transpose( h, [1,0,2]  )
        h = tmp_hiddens[-1]
            
        concat_h.append(h)
         
    # hidden space merge
    h = tf.concat(concat_h, 1)
    
    return h 


'''
if weight_type=="tanh":
            lower_bound=-np.sqrt(6. / (in_dim + out_dim));
            upper_bound=np.sqrt(6. / (in_dim + out_dim));
elif weight_type=="sigmoid":
            lower_bound=-4*np.sqrt(6. / (in_dim + out_dim));
            upper_bound=4*np.sqrt(6. / (in_dim + out_dim));
elif weight_type=="none":
            lower_bound=0;
            upper_bound=1./(in_dim+out_dim);
'''

def dense_layers_with_output(dim_layers, x, dim_x, dim_output, scope, dropout_keep_rate):
        
        with tf.variable_scope(scope + str(0)):
                w = tf.Variable( tf.random_normal([dim_x, dim_layers[0]], stddev = math.sqrt(2.0/float(dim_x))) )
                b = tf.Variable( tf.zeros([dim_layers[0]]))
                
                h = tf.nn.relu( tf.matmul(x, w) + b )
                regularizer = tf.nn.l2_loss(w)
        
        # dropout
        h = tf.nn.dropout(h, dropout_keep_rate)

        for i in range(1, len(dim_layers)):
            
            with tf.variable_scope(scope + str(i)):
                
                tmph = h
                
                w = tf.Variable( tf.random_normal([dim_layers[i-1], dim_layers[i]], \
                                                  stddev = math.sqrt(2.0/float(dim_layers[i-1]))) )
                b = tf.Variable( tf.zeros([dim_layers[i]]))
                
                #residual connection
                h = tf.nn.relu( tf.matmul(h, w) + b )
                #h = tmph + h
                
                # L2  
                regularizer += tf.nn.l2_loss(w)    
        
        with tf.variable_scope(scope + "output"):
            w = tf.Variable( tf.random_normal([dim_layers[-1], dim_output ],stddev = math.sqrt(2.0/float(dim_layers[-1]))) )
            b = tf.Variable( tf.zeros([1,]))
            
            output = tf.matmul(h, w) + b
            # L2  
            #regularizer += tf.nn.l2_loss(w) 
        
        return tf.squeeze(output), h, regularizer
    
# batch normalization
def bn_dense_layers_with_output(dim_layers, x, dim_x, dim_output, scope, dropout_keep_rate):
        
        h = x
        dim_layers.insert(0, dim_x)
        
        # dropout
        h = tf.nn.dropout(h, dropout_keep_rate)
        
        regularizer = 0.0
        for i in range(1, len(dim_layers)):
            with tf.variable_scope(scope + str(i)):
                
                w_BN = tf.Variable( tf.random_normal([dim_layers[i-1], dim_layers[i]],stddev = \
                                                     math.sqrt(2.0/float(dim_layers[i-1]))) )
                h_BN = tf.matmul(h, w_BN)
                batch_mean, batch_var = tf.nn.moments(h_BN, [0])
        
                scale = tf.Variable(tf.ones([dim_layers[i]]))
                beta  = tf.Variable(tf.zeros([dim_layers[i]]))
                
                h = tf.nn.relu(tf.nn.batch_normalization(h_BN, batch_mean, batch_var, beta, scale, 1e-5))
                
                # L2  
                regularizer += tf.nn.l2_loss(w_BN)    
        
        with tf.variable_scope(scope + "output"):
            w = tf.Variable( tf.random_normal([dim_layers[-1], dim_output ],stddev = math.sqrt(2.0/float(dim_layers[-1]))) )
            b = tf.Variable( tf.zeros([1,]))
            
            output = tf.matmul(h, w) + b
            # L2  
            #regularizer += tf.nn.l2_loss(w) 
        
        return tf.squeeze(output), h, regularizer
    
class neural_mixture_lstm():
    
    def context_from_hiddens_lstm(self, h, bool_attention):
        
        if bool_attention == True:
            return 0
        else:
            tmp_hiddens = tf.transpose( h, [1,0,2]  )
            h = tmp_hiddens[-1]
            
            return h
    
    def __init__(self, session, dense_dims, lstm_dims, lr, l2, batch_size, steps, dims ):
        
        # build the network graph 
        self.LEARNING_RATE = lr
                
        self.N_BATCH = batch_size
        self.L2 = l2
        
        self.MAX_NORM = 0.0
        self.epsilon = 1e-3
        
        self.sess = session
        
        # initialize placeholders
        self.v_pre = tf.placeholder(tf.float32, [None, steps[0], dims[0]])
        self.req   = tf.placeholder(tf.float32, [None, steps[1], dims[1]])
        self.distr = tf.placeholder(tf.float32, [None, steps[2], dims[2]])
        self.y     = tf.placeholder(tf.float32, [None, ])
        self.keep_prob = tf.placeholder(tf.float32)
        
        # models on individual feature groups
        h_v = ln_lstm_stacked( self.v_pre, lstm_dims[0], 'v_pre' )
        #h_r = ln_lstm_stacked( self.req,   lstm_dims[1],   'req' )
        h_d = ln_lstm_stacked( self.distr, lstm_dims[1], 'distr' )
        
        # extract context hidden states
        cont_h_v = self.context_from_hiddens_lstm(h_v, False)
        #cont_h_r = self.context_from_hiddens_lstm(h_r, False)
        cont_h_d = self.context_from_hiddens_lstm(h_d, False)
        
        concat_h = tf.concat( [cont_h_v, cont_h_d], 1 )
        concat_h_dim = sum( [i[-1]  for i in lstm_dims] )
        
        
        # individual prediction
        y_hat_v,_,regu = dense_layers_with_output(dense_dims[0], cont_h_v, lstm_dims[0][-1], 1, 'y_hat_v',\
                                                          self.keep_prob)
        self.regu = regu
        #y_hat_r,_,regu = dense_layers_with_output(dense_dims[1], cont_h_r, lstm_dims[1][-1], 1, 'y_hat_r',\
                                                       #   self.keep_prob)
        #self.regu += regu
        y_hat_d,_,regu = dense_layers_with_output(dense_dims[1], cont_h_d, lstm_dims[1][-1], 1, 'y_hat_d',\
                                                          self.keep_prob)
        self.regu += regu
        
        # concatenate individiual predictive mean of each expert
        y_hat_concat = tf.squeeze( tf.stack( [y_hat_v, y_hat_d], 1 ) )          
        
        # obtain gating parameters
        concat_mix_para, _ , regu = bn_dense_layers_with_output( dense_dims[2], concat_h, concat_h_dim, 4, 'gate', \
                                                                  self.keep_prob )
        self.regu += regu
        
        # logits and variance of each expert
        self.logit, varv, vard = tf.split( concat_mix_para, [2,1,1], 1 )
        
        
        # gate probability 
        self.gates = tf.nn.softmax(self.logit)
        
        #negative log likelihood
        sd_v = tf.squeeze(tf.square(varv))
        #sd_r = tf.squeeze(tf.square(varr))
        sd_d = tf.squeeze(tf.square(vard))
                          
        tmpllk_v = tf.exp(-0.5*tf.square(self.y-tf.squeeze(y_hat_v))/(sd_v**2+1e-5))/(2.0*np.pi*(sd_v**2+1e-5))**0.5
        #tmpllk_req = tf.exp(-0.5*tf.square(self.y-tf.squeeze(y_hat_r))/(sd_r**2+1e-5))/(2.0*np.pi*(sd_r**2+1e-5))**0.5
        tmpllk_distr = tf.exp(-0.5*tf.square(self.y-tf.squeeze(y_hat_d))/(sd_d**2+1e-5))/(2.0*np.pi*(sd_d**2+1e-5))**0.5
        
        llk = tf.multiply( tf.squeeze(tf.stack([tmpllk_v, tmpllk_distr], 1)), self.gates ) 
        self.neg_logllk = tf.reduce_sum( -1.0*tf.log( tf.reduce_sum(llk, 1)+1e-5 ) )
               
        # mixed prediction
        self.y_hat = tf.reduce_sum( tf.multiply(y_hat_concat, self.gates), 1 ) 
                     
        #test
        self.test1 =  tf.shape( concat_h )
        self.test2 =  tf.shape( llk )
        
    
    def test(self, v_train, req_train, distr_train, y_train, keep_prob ):
        res = self.sess.run([self.test1, self.test2 ],\
                             feed_dict={self.v_pre:v_train, self.req:req_train,\
                                        self.distr:distr_train, self.y:y_train, self.keep_prob:keep_prob })
        return res
    
#   initialize loss and optimization operations for training
    def train_ini(self):
        
        # negative log likelihood
        self.loss = self.neg_logllk + self.L2*self.regu
        self.weight_regu = self.L2*self.regu
        
        # squared error
        self.err = tf.losses.mean_squared_error( self.y, self.y_hat )        
        #self.loss = self.err + self.L2*self.regu
        
        # !
        self.optimizer = \
        tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.loss)
#         tf.train.AdadeltaOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)
#         tf.train.RMSPropOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)
#         tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(self.cost)
#         tf.train.AdadeltaOptimizer(learning_rate = self.lr).minimize(self.cost)
#         tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.cost)
#         tf.train.GradientDescentOptimizer(learning_rate = self.lr).minimize(self.cost)
        
        self.init = tf.global_variables_initializer()
        self.sess.run( self.init )
        
        
    #   training on batch of data
    def train_batch(self, v_train, req_train, distr_train, y_train, keep_prob ):
        
        # !
        _,c = self.sess.run([self.optimizer, self.loss],\
                             feed_dict={self.v_pre:v_train, self.req:req_train,\
                                        self.distr:distr_train, self.y:y_train, self.keep_prob:keep_prob })
        return c
    
    
    #   infer givn testing data
    def inference(self, v_test, req_test, distr_test, y_test, keep_prob):
        
        return self.sess.run([self.err, self.weight_regu],feed_dict = {self.v_pre:v_test,      self.req:req_test,\
                                                      self.distr:distr_test,  self.y:y_test, self.keep_prob:keep_prob })
    #   predict givn testing data
    def predict(self, v_test, req_test, distr_test, keep_prob):
        return self.sess.run( self.y_hat, feed_dict = {self.v_pre:v_test,      self.req:req_test,\
    
                                                       self.distr:distr_test,  self.keep_prob:keep_prob })
    #   predict givn testing data
    def predict_gates(self, v_test, req_test, distr_test, keep_prob):
        return self.sess.run( self.gates , feed_dict = {self.v_pre:v_test,      self.req:req_test,\
                                                        self.distr:distr_test,  self.keep_prob:keep_prob })
    
    def predict_logit(self, v_test, req_test, distr_test, keep_prob):
        return self.sess.run( self.logit , feed_dict = {self.v_pre:v_test,      self.req:req_test,\
                                                        self.distr:distr_test,  self.keep_prob:keep_prob })