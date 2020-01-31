from __future__ import print_function

from keras.layers import Input, Dropout, Concatenate, Dense
#from sklearn.ensemble import RandomForestClassifier
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as Ks

import numpy as np
import random
import time

MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
#NB_EPOCH =  400
PATIENCE = 100  # early stopping patience
GCN_SIZE = 16
Merge_SIZE = 256
FNN_SIZE = 16
ACTIVATION =  "relu" # "tanh"  #,"sigmoid"


def load_data_sets(FilePath="data/",DataFile="breastA",nFiles=8):
   
    inputs_list = list()
    for i in range(0, nFiles):
        features = np.genfromtxt("{}{}/input/X{}".format(FilePath, DataFile, i), dtype=np.float32)
        inputs_list.append(features)
    
    y = np.genfromtxt("{}{}/input/labels".format(FilePath, DataFile), dtype=np.int)    
    
    return inputs_list, y

def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)+ pow(10.0, -9)))


def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def evaluate_preds(preds, labels, indices):

    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc

def k_fold_cross_validation(items, k, randomize=False):

    if randomize:
        items = list(items)
        shuffle(items)

    slices = [items[i::k] for i in range(k)]

    for i in range(k):
        validation = slices[i]
        training = [item
                    for s in slices if s is not validation
                    for item in s]
        yield training, validation
    
    #return training, validation

def evaluate_preds_new(preds, labels):

    split_loss = categorical_crossentropy(preds, labels)
    split_acc = accuracy(preds, labels)

    return split_loss, split_acc


def Get_features_weight(weight_list, n_features, n_graphs):
     
    w_features =  np.zeros((n_graphs,n_features))    
    w_graphs = np.zeros(n_graphs) 
    
    group_weights = abs(weight_list[2*n_graphs])
    for i in range(0,n_graphs):
        idx = i*2
        w_features[i,:] = np.sum(abs(weight_list[idx]), axis=1)
        
        start = i*16
        w_graphs[i] = np.sum(group_weights[range(start,start+16),:])
        #w_graphs[i] = np.sum(w_features[i,:])
    w_features_vector = np.sum(w_features,axis=0)
   
    return w_features_vector, w_graphs


def MDGCCN_1_model(inputs, y, DROUPOUT_SIZE, LEARNING_RATE, TUNNING_SIZE):
    
    members = list()
  
    for i in range(0,len(inputs) ):
    
        X_in = Input(shape=(inputs[i].shape[1],))
        H_i = Dropout(DROUPOUT_SIZE)(X_in)        
        names = 'first_' + str(i+1) + '_layer' 
        X_out = Dense(GCN_SIZE, activation=ACTIVATION  #, kernel_initializer='normal'
                      , kernel_regularizer=l2(TUNNING_SIZE),name=names)(H_i)
        model = Model(inputs=X_in, outputs=X_out)
        members.append(model)    
    
    # define multi-headed input
    ensemble_visible = [model.input for model in members]
  
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    
    H = Concatenate(axis=-1)(ensemble_outputs)
    H = Dense(FNN_SIZE, activation=ACTIVATION #, kernel_initializer='normal'
              , kernel_regularizer=l2(TUNNING_SIZE),
                name='ALLX')(H)

    z = Dense(y.shape[1], activation="softmax")(H)
   
    model = Model(inputs=ensemble_visible, outputs=z)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LEARNING_RATE))
    
    return model

def MDGCCN_2_model(inputs, y, DROUPOUT_SIZE, LEARNING_RATE, TUNNING_SIZE):
    
    members = list()
    
    X_0 = Input(shape=(inputs[0].shape[1],))
       
  
    for i in range(1,len(inputs) ):
    
        X_in = Input(shape=(inputs[i].shape[1],))
        #H_i = Dropout(DROUPOUT_SIZE)(X_in)        
        names = 'first_' + str(i+1) + '_layer' 
        H_i = Dense(GCN_SIZE, activation=ACTIVATION #, kernel_initializer='normal'
                      , kernel_regularizer=l2(TUNNING_SIZE),name=names)(X_in)
        #H_i = Dropout(DROUPOUT_SIZE)(H_i)        
        names = 'second_' + str(i+1) + '_layer' 
        X_out = Dense(GCN_SIZE, activation=ACTIVATION #, kernel_initializer='normal'
                      , kernel_regularizer=l2(TUNNING_SIZE),name=names)(H_i)
        model = Model(inputs=X_in, outputs=X_out)
        members.append(model)    
    
    # define multi-headed input
    ensemble_visible = [X_0] + [model.input for model in members]
  
    # concatenate merge output from each model
    ensemble_outputs = [X_0] + [model.output for model in members]
    
    
    H = Concatenate(axis=-1)(ensemble_outputs)
    H = Dense(FNN_SIZE, activation=ACTIVATION#, kernel_initializer='normal'
              , kernel_regularizer=l2(TUNNING_SIZE),
                name='dnn1')(H)

    z = Dense(y.shape[1], activation="softmax")(H)   
    model = Model(inputs=ensemble_visible, outputs=z)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LEARNING_RATE))
    
    return model

def FNN_model(X, y, DROUPOUT_SIZE, LEARNING_RATE, TUNNING_SIZE, LAYERS_NUM):

    X0 = Input(shape=(X.shape[1],))
    
    H0 = Dropout(DROUPOUT_SIZE)(X0)
    H0 = Dense(FNN_SIZE, activation=ACTIVATION #, kernel_initializer='normal'
               , kernel_regularizer=l2(TUNNING_SIZE),
               name='originalX0')(H0)
    
    for i in range(1,LAYERS_NUM):
        H0 = Dropout(DROUPOUT_SIZE)(H0)
        H0 = Dense(FNN_SIZE, activation=ACTIVATION #, kernel_initializer='normal'
                   , kernel_regularizer=l2(TUNNING_SIZE),
                name='added000')(H0)
    
    #H0 = Dropout(DROUPOUT_SIZE)(H0)
    z = Dense(y.shape[1], activation="softmax")(H0)

    model = Model(inputs=[X0], outputs=z)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LEARNING_RATE))    
    return model

def get_model(inputs, y, DROUPOUT_SIZE, LEARNING_RATE, TUNNING_SIZE, MODEL_TYPE='DNN2'):
    
    if MODEL_TYPE=='DNN2' :
        LAYERS_NUM = 2
        model = FNN_model(inputs, y, DROUPOUT_SIZE, LEARNING_RATE, TUNNING_SIZE,LAYERS_NUM)
        return model
    
    if MODEL_TYPE=='DNN3' :      
        LAYERS_NUM = 3
        model = FNN_model(inputs, y, DROUPOUT_SIZE, LEARNING_RATE, TUNNING_SIZE,LAYERS_NUM)
        return model
    
    if MODEL_TYPE=='MDGCNN1' :
        model = MDGCCN_1_model(inputs, y, DROUPOUT_SIZE, LEARNING_RATE, TUNNING_SIZE)
        return model
    
    model = MDGCCN_2_model(inputs, y, DROUPOUT_SIZE, LEARNING_RATE, TUNNING_SIZE)
    return model


#def MDGCCN(inputs, y, training_mask, validatoin_mask, test_mask
#                 ,DROUPOUT_SIZE, LEARNING_RATE, TUNNING_SIZE,NB_EPOCH, MODEL_TYPE='DNN2'):
def MDGCCN(inputs, y, training_mask,test_mask
             ,DROUPOUT_SIZE, LEARNING_RATE, TUNNING_SIZE,NB_EPOCH, MODEL_TYPE='DNN2'):
    
    model = get_model(inputs, y, DROUPOUT_SIZE, LEARNING_RATE, TUNNING_SIZE, MODEL_TYPE)    

    train_val_loss = 0.0
    train_val_acc = 0.0
    wait = 0
    preds = None
    best_val_loss = 99999 
    
    for epoch in range(1, NB_EPOCH + 1):
        # Log wall-clock time
        t = time.time()
        
        model.fit(inputs, y, sample_weight=training_mask,
                  batch_size=y.shape[0], epochs=1, shuffle=True, verbose=0)

        # Predict on full dataset
        preds = model.predict(inputs, batch_size=y.shape[0])
         # Train / validation scores
        #train_val_loss = categorical_crossentropy(preds[training_mask | validatoin_mask], y[training_mask | validatoin_mask])
        #train_val_acc = accuracy(preds[training_mask | validatoin_mask], y[training_mask | validatoin_mask])
        train_val_loss = categorical_crossentropy(preds[training_mask ], y[training_mask ])
        train_val_acc = accuracy(preds[training_mask ], y[training_mask ])

        if train_val_loss < best_val_loss:
            best_val_loss = train_val_loss
            wait = 0
        else:
            if wait >= PATIENCE:
                print('Epoch {}: early stopping'.format(epoch))
                break
            wait += 1

    # Testing
    test_loss = categorical_crossentropy(preds[test_mask], y[test_mask])
    test_acc = accuracy(preds[test_mask], y[test_mask])    
    
    names  = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()
  
    Ks.clear_session()
    del model

    return test_loss,test_acc,epoch,names,weights

def MDGCCN_CV(inputs, y, training_mask, test_mask
                 ,DROUPOUT_SIZE, LEARNING_RATE, TUNNING_SIZE, NB_EPOCH, MODEL_TYPE='DNN2'):
       
    which_training = np.where(training_mask)
    which_training = which_training[0]
    
    random.shuffle(which_training)
    cv_num = 5
    
    #kf = KFold(n_splits=cv_num)
    cvscores = list()
    which_training_index = np.array(range(0,len(which_training)))

    for train_index, test_index in k_fold_cross_validation(which_training_index,cv_num):
              
        #model = get_model(inputs, y, DROUPOUT_SIZE, LEARNING_RATE, TUNNING_SIZE, MODEL_TYPE)    
        training_mask_cv = np.zeros( len(training_mask), dtype=bool )
        testing_mask_cv = np.zeros( len(training_mask), dtype=bool )
        
        #print("TRAIN:", train_index, "TEST:", test_index)
        training_mask_cv[which_training[train_index]] =  np.ones( len(train_index), dtype=bool )
        testing_mask_cv[which_training[test_index]] =  np.ones( len(test_index), dtype=bool )
         
        #model.fit(inputs, y, sample_weight=training_mask_cv,
        #          batch_size=y.shape[0], epochs=NB_EPOCH+1, shuffle=True, verbose=0)
        
        #preds = model.predict(inputs, batch_size=y.shape[0])
        test_loss,test_acc,epoch,names,weights= MDGCCN(inputs, y, training_mask_cv,testing_mask_cv
             ,DROUPOUT_SIZE, LEARNING_RATE, TUNNING_SIZE,NB_EPOCH, MODEL_TYPE)
    
        #scores = accuracy(preds[testing_mask_cv], y[testing_mask_cv])        	     
        cvscores.append(test_acc)  
        
    return  np.mean(cvscores)


def MDGCCN_Exp(inputs, y, training_mask, test_mask
                 ,DROUPOUT_SIZE, LEARNING_RATE_LIST, TUNNING_SIZE_LIST, NB_EPOCH_LIST, MODEL_TYPE='DNN2'):
    
    max_cvsscores = 0
    opt_NB_EPOCH = 0
    opt_TUNNING = 0
    opt_LEARNING = 0
    
    for i in range(0,len(LEARNING_RATE_LIST)):
        for j in range(0,len(TUNNING_SIZE_LIST)):
            for k in range(0,len(NB_EPOCH_LIST)):
                cvsscores = MDGCCN_CV(inputs, y, training_mask, test_mask
                    ,DROUPOUT_SIZE, LEARNING_RATE_LIST[i], TUNNING_SIZE_LIST[j], NB_EPOCH_LIST[k], MODEL_TYPE=MODEL_TYPE)
                if cvsscores > max_cvsscores:
                    max_cvsscores = cvsscores
                    opt_NB_EPOCH = NB_EPOCH_LIST[k]
                    opt_TUNNING =  TUNNING_SIZE_LIST[j]
                    opt_LEARNING = LEARNING_RATE_LIST[i]
    
    test_loss, acc, epoch, names, weights =  MDGCCN(inputs, y, training_mask, test_mask, DROUPOUT_SIZE,opt_TUNNING,opt_LEARNING,opt_NB_EPOCH,MODEL_TYPE)
                   
    return acc, max_cvsscores, opt_LEARNING, opt_TUNNING, opt_NB_EPOCH, test_loss #, w_graphs, w_features
           
def MDGCCN_Exp_weight(inputs, y, training_mask, test_mask
                 ,DROUPOUT_SIZE, LEARNING_RATE_LIST, TUNNING_SIZE_LIST, NB_EPOCH_LIST):
    
    max_cvsscores = 0
    opt_NB_EPOCH = 0
    opt_TUNNING = 0
    opt_LEARNING = 0
    MODEL_TYPE='MDGCNN1'
    
    for i in range(0,len(LEARNING_RATE_LIST)):
        for j in range(0,len(TUNNING_SIZE_LIST)):
            for k in range(0,len(NB_EPOCH_LIST)):
                cvsscores = MDGCCN_CV(inputs, y, training_mask, test_mask
                    ,DROUPOUT_SIZE, LEARNING_RATE_LIST[i], TUNNING_SIZE_LIST[j], NB_EPOCH_LIST[k], MODEL_TYPE=MODEL_TYPE)
                if cvsscores > max_cvsscores:
                    max_cvsscores = cvsscores
                    opt_NB_EPOCH = NB_EPOCH_LIST[k]
                    opt_TUNNING =  TUNNING_SIZE_LIST[j]
                    opt_LEARNING = LEARNING_RATE_LIST[i]
    
    test_loss, acc, epoch, names, weights =  MDGCCN(inputs, y, training_mask, test_mask, DROUPOUT_SIZE,opt_TUNNING,opt_LEARNING,opt_NB_EPOCH,MODEL_TYPE)
               
    return weights,names
           



