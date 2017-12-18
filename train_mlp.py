#!/usr/bin/python

#from utils_libs import *
from ml_models import *
from neural_mixture import *
from utils_data_prep import *


# --- Load pre-processed training and testing data ---
file_postfix = "mle_norm_reg"

xtrain = np.load("../dataset/bk/xtrain_"+file_postfix+".dat")
xtest  = np.load("../dataset/bk/xtest_" +file_postfix+".dat")
ytrain = np.load("../dataset/bk/ytrain_"+file_postfix+".dat")
ytest  = np.load("../dataset/bk/ytest_" +file_postfix+".dat")

# !! IMPORTANT: feature normalization
xtest  = conti_normalization_test_dta( xtest, xtrain )
xtrain = conti_normalization_train_dta( xtrain )

print np.shape(xtrain), np.shape(ytrain), np.shape(xtest), np.shape(ytest)

# --- parameters ---

# txt file to record errors in training process 
res_log    = "res/neu_mlp.txt"
model_file = 'model/neu_mlp_mle.ckpt'
bool_train = True

# representation ability
para_n_hidden_list = [ 64, 64, 64 ] 
para_lr = 0.001
para_n_epoch = 500
para_batch_size = 128

# regularization
para_keep_prob = 1.0
para_l2 = 0.07

# fixed parameters
para_dim = len(xtrain[0])

# validation parameters
para_eval_byepoch = 10

# initialize the log
#with open(res_log, "w") as text_file:
#    text_file.close()

    
# --- main process ---   

if bool_train == True:
    
    with tf.Session() as sess:
        
        clf = neural_plain_mlp( sess, para_n_hidden_list, para_lr, para_l2, para_batch_size, para_dim )
    
        # initialize the network                          
        clf.train_ini()
        
        total_cnt   = np.shape(xtrain)[0]
        total_batch = int(total_cnt/para_batch_size)
        total_idx   = range(total_cnt)
    
        #   begin training epochs
        for epoch in range(para_n_epoch):
            tmpc=0.0
            
            # shuffle traning instances each epoch
            np.random.shuffle(total_idx)
            
            #  Loop over all batches
            for i in range(total_batch):
                batch_idx = total_idx[ i*para_batch_size: (i+1)*para_batch_size ] 
            
                batch_x = xtrain[ batch_idx ]
                batch_y = ytrain[ batch_idx ]
            
                tmpc += clf.train_batch( batch_x, batch_y, para_keep_prob )
                
            tmp_test_acc  = clf.inference(xtest, ytest,  para_keep_prob) 
            tmp_train_acc = clf.inference(xtrain, ytrain, para_keep_prob) 
            print "loss on epoch ", epoch, " : ", 1.0*tmpc/total_batch, sqrt(tmp_train_acc), sqrt(tmp_test_acc)
        
            with open(res_log, "a") as text_file:
                text_file.write( "Epoch %d : %f, %f, %f,  \n"%(epoch, 1.0*tmpc/total_batch, sqrt(tmp_train_acc[0]), sqrt(tmp_test_acc[0]))) 
        
        print "Optimization Finished!"
        
        # for test
        pytest = clf.predict(xtest, para_keep_prob)
        np.savetxt("res/pytest.txt", zip(pytest, ytest), delimiter=',')
        
        
else:
    
    # --- train the network under the best parameter set-up ---
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        
        clf = plain_mlp( sess, para_n_hidden_list, para_lr, para_l2, para_batch_size, para_dim )
    
        # initialize the network
        clf.train_ini()
        
        total_cnt   = np.shape(xtrain)[0]
        total_batch = int(total_cnt/para_batch_size)
        total_idx   = range(total_cnt)
        
        # begin training cycles
        for epoch in range(para_n_epoch):
            
            #  shuffle traning instances each epoch
            np.random.shuffle(total_idx)
            
            #  Loop over all batches
            for i in range(total_batch):
                batch_idx = total_idx[ i*para_batch_size: (i+1)*para_batch_size ] 
                
                batch_x =  xtrain[ batch_idx ]
                batch_y = ytrain[ batch_idx ]
            
                clf.train_batch( batch_x, batch_y, para_keep_prob )
        
        # save the model
        save_path = saver.save(sess, model_file)
        print("Model saved in file: %s" %save_path)




