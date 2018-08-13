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

# local packages
from utils_libs import *
from utils_rnn_basics import *


_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"
    
    
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
        self.v_auto = tf.placeholder(tf.float32, [None, order_v])
        self.req   = tf.placeholder(tf.float32, [None, order_req])
        self.distr = tf.placeholder(tf.float32, [None, order_distr])
        self.y     = tf.placeholder(tf.float32, [None, ])
        
        self.keep_prob = tf.placeholder(tf.float32)
        
        # models on individual feature groups
        pre_v, h_v, regular_v = self.dense_layers(self.HIDDEN_DIMS[0], self.v_auto, order_v, 1, 'v_auto', self.keep_prob )
        pre_req, h_req, regular_req = self.dense_layers(self.HIDDEN_DIMS[1], self.req, order_req, 1, 'req',self.keep_prob )
        pre_distr, h_distr, regular_distr = self.dense_layers(self.HIDDEN_DIMS[2], self.distr, order_distr, 1,\
                                                              'distr', self.keep_prob )

        # regularization 
        self.regu = regular_v + regular_req + regular_distr
        
        # concatenate individiual predictive mean of each expert
        pre = tf.stack( [pre_v, pre_distr], 1 )  
        pre = tf.squeeze( pre )  
        
        # explicit feature + hidden feature
        #hf_v = tf.concat([ self.v_auto, h_v ],1 )
        #hf_r = tf.concat([ self.req,   h_req ],1 )
        #hf_d = tf.concat([ self.distr, h_distr ],1 )
        
        hf_v = self.v_auto
        #hf_r = self.req
        hf_d = self.distr
        
        
        concat_x = tf.concat( [self.v_auto, self.distr], 1 )
        concat_dim = order_v + order_distr
        
        concat_mix_para, concat_h, regu = self.dense_layers( self.HIDDEN_DIMS[3], concat_x, concat_dim, 4, 'gate',\
                                                             self.keep_prob )
        self.regu += regu
        
        self.logit, varv, vard = tf.split( concat_mix_para, [2,1,1], 1 )
        
        
        # gates 
        self.gates = tf.nn.softmax(self.logit)
        
        # mixed prediction
        self.y_hat = tf.reduce_sum( tf.multiply(pre, self.gates), 1 ) 
         
        
        #negative log likelihood
       
        # constant for each expert
        #varv = tf.Variable(tf.random_normal([1,], stddev = math.sqrt(2.0/float(1.0))) )
        #varr = tf.Variable(tf.random_normal([1,], stddev = math.sqrt(2.0/float(1.0))) )
        #vard = tf.Variable(tf.random_normal([1,], stddev = math.sqrt(2.0/float(1.0))) )
        #regu_sig = tf.nn.l2_loss(varv)+tf.nn.l2_loss(varr)+tf.nn.l2_loss(vard)
        #self.regu += regu_sig
        
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
                             feed_dict={self.v_auto:v_train, self.req:req_train,\
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
        
        self.init = tf.global_variables_initializer()
        self.sess.run( self.init )
        
        
    #   training on batch of data
    def train_batch(self, v_train, req_train, distr_train, y_train, keep_prob ):
        
        # !
        _,c = self.sess.run([self.optimizer, self.loss],\
                             feed_dict={self.v_auto:v_train, self.req:req_train,\
                                        self.distr:distr_train, self.y:y_train, self.keep_prob:keep_prob })
        return c
    
    
    #   infer givn testing data
    def inference(self, v_test, req_test, distr_test, y_test, keep_prob):
        
        return self.sess.run([self.err, self.weight_regu],feed_dict = {self.v_auto:v_test,      self.req:req_test,\
                                                      self.distr:distr_test,  self.y:y_test, self.keep_prob:keep_prob })
    #   predict givn testing data
    def predict(self, v_test, req_test, distr_test, keep_prob):
        return self.sess.run( self.y_hat, feed_dict = {self.v_auto:v_test,      self.req:req_test,\
    
                                                       self.distr:distr_test,  self.keep_prob:keep_prob })
    #   predict givn testing data
    def predict_gates(self, v_test, req_test, distr_test, keep_prob):
        return self.sess.run( self.gates , feed_dict = {self.v_auto:v_test,      self.req:req_test,\
                                                        self.distr:distr_test,  self.keep_prob:keep_prob })
    
    def predict_logit(self, v_test, req_test, distr_test, keep_prob):
        return self.sess.run( self.logit , feed_dict = {self.v_auto:v_test,      self.req:req_test,\
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
    
    
# ---- Utility functions for LSTM ----

def plain_dense( x, x_dim, dim_layers, scope, dropout_keep_prob):
    
    with tf.variable_scope(scope):
        # initilization
        w = tf.get_variable('w', [x_dim, dim_layers[0]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.zeros([dim_layers[0]]))
        h = tf.nn.relu( tf.matmul(x, w) + b )

        regularization = tf.nn.l2_loss(w)
                
        #dropout
        h = tf.nn.dropout(h, dropout_keep_prob)
        
    for i in range(1, len(dim_layers)):
        with tf.variable_scope(scope+str(i)):
            w = tf.get_variable('w', [dim_layers[i-1], dim_layers[i]], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.zeros( dim_layers[i]))
            h = tf.nn.relu( tf.matmul(h, w) + b )
                
            regularization += tf.nn.l2_loss(w)
        
    return h, regularization

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

# layer normalization for LSTM
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

def lstm_stacked( x, dim_layers, scope ):
    
    with tf.variable_scope(scope):
        #Deep lstm: residual or highway connections 
        #block: residual connections
        
        #tf.nn.rnn_cell.LSTMCell
        lstm_cell = tf.nn.rnn_cell.LSTMCell(dim_layers[0])
                                                #initializer= tf.contrib.keras.initializers.glorot_normal())
        hiddens, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = x, dtype = tf.float32)
            
    for i in range(1,len(dim_layers)):
        with tf.variable_scope(scope+str(i)):
            
            lstm_cell = tf.nn.rnn_cell.LSTMCell (dim_layers[i])
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
        
        # dropout
        h = tf.nn.dropout(x, dropout_keep_rate)
        
        with tf.variable_scope(scope + str(0)):
                w = tf.Variable( tf.random_normal([dim_x, dim_layers[0]], stddev = math.sqrt(2.0/float(dim_x))) )
                b = tf.Variable( tf.zeros([dim_layers[0]]))
                h = tf.nn.relu( tf.matmul(x, w) + b )
                # L2
                regularizer = tf.nn.l2_loss(w)

        for i in range(1, len(dim_layers)):
            
            with tf.variable_scope(scope + str(i)):
                
                tmph = h
                
                w = tf.Variable( tf.random_normal([dim_layers[i-1], dim_layers[i]], \
                                                  stddev = math.sqrt(2.0/float(dim_layers[i-1]))) )
                b = tf.Variable( tf.zeros([dim_layers[i]]))
                h = tf.nn.relu( tf.matmul(h, w) + b )

                #residual connection
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
    

def context_from_hiddens_lstm(h, bool_attention):
    
    if bool_attention == True:
        return 0
    else:
        tmp_hiddens = tf.transpose( h, [1,0,2]  )
        h = tmp_hiddens[-1]
            
        return h


# used for outputing prediction or logits of classification
def one_dense(x, x_dim, scope, out_dim, activation):
    
    with tf.variable_scope(scope):
        
        w = tf.get_variable('w', shape=[x_dim, out_dim], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.zeros([1]))
        
        if activation == 'relu':
            
            return tf.squeeze(tf.nn.relu(tf.matmul(x, w) + b)), tf.nn.l2_loss(w)
        
        elif activation == 'leaky_relu':
            
            return tf.maximum( tf.matmul(x, w) + b, 0.2*(tf.matmul(x, w) + b) ), tf.nn.l2_loss(w)
        
        elif activation == 'linear':
            
            return tf.squeeze( tf.matmul(x, w) + b ), tf.nn.l2_loss(w)
        
        else:
            print ' ------------ [ERROR] activiation type'

    
    
# ---- LSTM mixture ----

class lstm_mixture():
    
    def context_from_hiddens_lstm(self, h, bool_attention):
        
        if bool_attention == True:
            
            return 0
        
        else:
            tmp_hiddens = tf.transpose( h, [1,0,2]  )
            h = tmp_hiddens[-1]
            
            return h
    
    def __init__(self, session, lr, l2, steps_auto, dim_x, steps_x, num_dense, max_norm, lstm_size_layers,\
                 loss_type, activation_type, pos_regu, gate_type):
        
        # build the network graph 
        self.LEARNING_RATE = lr
        
        # [ l2_mean, l2_var, l2_gate, l2_pos ]
        # TO DO 
        
        self.sess = session
        
        self.loss_type = loss_type
        
        # initialize placeholders
        self.auto = tf.placeholder(tf.float32, [None, steps_auto, 1])
        self.x = tf.placeholder(tf.float32, [None, steps_x, dim_x])
        self.y = tf.placeholder(tf.float32, [None, ])
        self.keep_prob = tf.placeholder(tf.float32, [None, ])
        
        # --- individual LSTM ---

        h_auto, _ = plain_lstm( self.auto, lstm_size_layers, 'lstm_auto', self.keep_prob )
        
        h_x, _ = plain_lstm( self.x, lstm_size_layers, 'lstm_x', self.keep_prob )
        
        # obtain the last hidden state
        tmp_h_auto  = tf.transpose( h_auto, [1,0,2] )[-1]
        tmp_h_x = tf.transpose( h_x, [1,0,2] )[-1]

        # component
        h_auto, regu_dense_auto, out_dim = multi_dense(tmp_h_auto, lstm_size_layers[-1], num_dense, 'dense_auto',\
                                                       tf.gather(self.keep_prob, 0), max_norm)
        
        mean_auto, regu_mean_auto = one_dense(h_auto, out_dim, 'mean_auto', 1, activation_type)
        var_auto, regu_var_auto = one_dense(h_auto, out_dim, 'var_auto', 1, 'relu')
        sd_auto = var_auto
        
        regu_mean_auto = regu_dense_auto + regu_mean_auto
        
        # component
        h_x, regu_dense_x, out_dim = multi_dense(tmp_h_x, lstm_size_layers[-1], num_dense, 'dense_x', \
                                                 tf.gather(self.keep_prob, 0), max_norm)
        
        mean_x, regu_mean_x = one_dense(h_x, out_dim, 'mean_x', 1, activation_type)
        var_x, regu_var_x = one_dense(h_x, out_dim, 'var_x', 1, 'relu')
        sd_x = var_x
        
        regu_mean_x = regu_dense_x + regu_mean_x
            
        # mean concatenate of each expert
        mean_concat = tf.squeeze( tf.stack( [mean_auto, mean_x], 1 ) )          
        
        # --- gate ---
        
        # TO DO 
        # adaptive moving average logit 
        
        
        
        
        if gate_type == 'softmax':
            # softmax
            # [N 2D]
            h_concat = tf.concat( [tmp_h_auto, tmp_h_x], 1 )
            self.logit, regu_gate = one_dense(h_concat, lstm_size_layers[-1]+lstm_size_layers[-1], 'gate', 2, 'relu')
            self.gates = tf.nn.softmax(self.logit)
        
        elif gate_type == 'logistic':
            # logistic
            tmp_logit, regu_gate = one_dense(tmp_h_x, lstm_size_layers[-1], 'logit_x', 1, 'relu')
            self.logit = tf.stack( [tf.ones(tf.shape(tmp_logit)[0]), tmp_logit], 1 ) 
            self.gates = tf.stack( [1.0 - tf.sigmoid(tmp_logit), tf.sigmoid(tmp_logit)] ,1 )
        
        
        else:
            print ' ----- [ERROR] gate type'
            
        # --- regularization ---
        # ?
        
        # mean non-negative  
        regu_mean_pos = tf.reduce_sum( tf.maximum(0.0, -1.0*mean_auto) + tf.maximum(0.0, -1.0*mean_x) )
        
        self.regu = l2*(regu_mean_auto+regu_mean_x) + \
                    l2*(regu_var_auto+regu_var_x) + \
                    l2*(regu_gate)
        
        if pos_regu == True:
            self.regu += l2*regu_mean_pos
            
        
        # --- loss ---
        
        # loss: negative log likelihood - normal 
        tmpllk_auto_norm = tf.exp(-0.5*tf.square(self.y - mean_auto)/(sd_auto**2+1e-5))/(2.0*np.pi*(sd_auto**2+1e-5))**0.5
        tmpllk_x_norm = tf.exp(-0.5*tf.square(self.y - mean_x)/(sd_x**2+1e-5))/(2.0*np.pi*(sd_x**2+1e-5))**0.5
        
        llk_norm = tf.multiply( tf.squeeze(tf.stack([tmpllk_auto_norm, tmpllk_x_norm], 1)), self.gates ) 
        self.neg_logllk_norm = tf.reduce_sum( -1.0*tf.log( tf.reduce_sum(llk_norm, 1)+1e-5 ) )
        
        self.y_hat_norm = tf.reduce_sum( tf.multiply(mean_concat, self.gates), 1 )
        
        # loss: negative log likelihood - lognormal
        tmpllk_auto_log = tf.exp(-0.5*tf.square(tf.log(self.y+1e-5)-mean_auto)/(sd_auto**2+1e-5))/(2.0*np.pi*(sd_auto**2+1e-5))**0.5/(self.y+1e-5)
        tmpllk_x_log = tf.exp(-0.5*tf.square(tf.log(self.y+1e-5)-mean_x)/(sd_x**2+1e-5))/(2.0*np.pi*(sd_x**2+1e-5))**0.5/(self.y+1e-5)
        
        llk_log = tf.multiply( (tf.stack([tmpllk_auto_log, tmpllk_x_log], 1)), self.gates ) 
        self.neg_logllk_log = tf.reduce_sum( -1.0*tf.log(tf.reduce_sum(llk_log, 1)+1e-5) ) 
        
        self.y_hat_log = tf.reduce_sum( tf.multiply(tf.exp(mean_concat), self.gates), 1 )
               
        # loss: squared 
        self.y_hat_sq = tf.reduce_sum( tf.multiply(mean_concat, self.gates), 1 ) 
        self.sq = tf.losses.mean_squared_error( self.y, self.y_hat_sq )
        
        # loss type
        if self.loss_type == 'gaussian':
            self.y_hat = self.y_hat_norm
            self.loss = self.neg_logllk_norm + self.regu
        
        elif self.loss_type == 'lognorm':
            self.y_hat = self.y_hat_log
            self.loss = self.neg_logllk_log + self.regu
        
        elif self.loss_type == 'sq':
            self.y_hat = self.y_hat_sq
            self.loss = self.sq + self.regu
            
            
        # --- errors metric ---
        
        # RMSE
        self.rmse = tf.sqrt( tf.losses.mean_squared_error(self.y, self.y_hat) )
        # MAPE
        self.mape = tf.reduce_mean( tf.abs((self.y - self.y_hat)/(self.y+1e-10)) )
        # MAE
        self.mae = tf.reduce_mean( tf.abs(self.y - self.y_hat) )
        
        
    def model_reset(self):
        self.init = tf.global_variables_initializer()
        self.sess.run( self.init )
        
#   initialize loss and optimization operations for training
    def train_ini(self):
        
        # !
        self.optimizer = \
        tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.loss)
        
        self.init = tf.global_variables_initializer()
        self.sess.run( self.init )
        
        
    #   training on batch of data
    def train_batch(self, v_train, distr_train, y_train, keep_prob ):
        
        # !
        _,c = self.sess.run([self.optimizer, self.loss],\
                             feed_dict={self.auto:v_train, self.x:distr_train, self.y:y_train, self.keep_prob:keep_prob})
        return c
    
    
    #   infer givn testing data
    def inference(self, auto_test, x_test, y_test, keep_prob):
        
        return self.sess.run([self.rmse, self.mae, self.mape], \
                             feed_dict = {self.auto:auto_test, self.x:x_test, self.y:y_test, self.keep_prob:keep_prob })
    
    #   predict givn testing data
    def predict(self, auto_test, x_test, keep_prob):
        return self.sess.run( self.y_hat, feed_dict = {self.auto:auto_test,\
                                                       self.x:x_test, self.keep_prob:keep_prob })
    #   predict givn testing data
    def predict_gates(self, auto_test, x_test, keep_prob):
        return self.sess.run( self.gates, feed_dict = {self.auto:auto_test,\
                                                        self.x:x_test, self.keep_prob:keep_prob })
    
    #def predict_logit(self, auto_test, x_test, keep_prob):
    #    return self.sess.run( self.logit, feed_dict = {self.auto:auto_test,\
    #                                                   self.x:x_test, self.keep_prob:keep_prob })


# ---- LSTM feature concatenation ----

class lstm_concat():
    

    def __init__(self, session, lr, l2, steps_auto, dim_x, steps_x, num_dense, max_norm, size_layers_lstm):
        
        # build the network graph 
        self.LEARNING_RATE = lr
                
        self.epsilon = 1e-3
        
        self.sess = session
        
        # initialize placeholders
        self.auto = tf.placeholder(tf.float32, [None, steps_auto, 1])
        self.y = tf.placeholder(tf.float32, [None, ])
        self.keep_prob = tf.placeholder(tf.float32, [None])
        
        self.x = tf.placeholder(tf.float32, [None, steps_x, dim_x])
        
        
        # --- individual LSTM ---

        h_auto, _ = plain_lstm( self.auto, size_layers_lstm, 'lstm-v', self.keep_prob )
        
        h_x, _ = plain_lstm( self.x, size_layers_lstm, 'lstm-ob', self.keep_prob )
        
        # obtain the last hidden state
        tmp_h_auto  = tf.transpose( h_auto, [1,0,2] )
        tmp_h_x = tf.transpose( h_x, [1,0,2] )
        
        
        # --- concatenation to output prediction ---
        
        # [N 2D]
        h = tf.concat( [tmp_h_auto[-1], tmp_h_x[-1]], 1 )
            
        # dropout
        h, regu_dense, out_dim = multi_dense( h, 2*size_layers_lstm[-1], num_dense, \
                                              'dense', tf.gather(self.keep_prob, 0), max_norm)
        
        with tf.variable_scope("output"):
            
            w = tf.get_variable('w', shape=[out_dim, 1],\
                                     initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.zeros([1]))
            
            self.y_hat = tf.squeeze( tf.nn.relu(tf.matmul(h, w) + b) )
            
            # regularization
            # ?
            self.regu = l2*tf.nn.l2_loss(w)
            
        # regularization
        self.regu += l2*regu_dense
        
        # --- errors metric ---
        
        # RMSE
        self.mse = tf.losses.mean_squared_error(self.y, self.y_hat)
        self.rmse = tf.sqrt( self.mse )
        # MAPE
        self.mape = tf.reduce_mean( tf.abs((self.y - self.y_hat)/(self.y+1e-10)) )
        # MAE
        self.mae = tf.reduce_mean( tf.abs(self.y - self.y_hat) )
        
        # --- loss ---
        self.loss = self.mse + self.regu
        
    
    def model_reset(self):
        self.init = tf.global_variables_initializer()
        self.sess.run( self.init )
        
#   initialize loss and optimization operations for training
    def train_ini(self):
            
        self.train = tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE)
        self.optimizer =  self.train.minimize(self.loss)
        
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        
        
    #   training on batch of data
    def train_batch(self, auto_train, x_train, y_train, keep_prob ):
        
        # !
        _, c = self.sess.run([self.optimizer, self.loss],\
                             feed_dict={self.auto:auto_train, \
                                        self.x:x_train, self.y:y_train, self.keep_prob:keep_prob })
        return c
    
    #   infer givn testing data
    def inference(self, auto_test, x_test, y_test, keep_prob):
        
        return self.sess.run([self.rmse, self.mae, self.mape], feed_dict = {self.auto:auto_test, \
                                                                 self.x:x_test, self.y:y_test, self.keep_prob:keep_prob })
    
    #   predict givn testing data
    def predict(self, auto_test, x_test, keep_prob):
        
        return self.sess.run( [self.y_hat], feed_dict = {self.auto:auto_test, \
                                                         self.x:x_test, self.keep_prob:keep_prob })
    
