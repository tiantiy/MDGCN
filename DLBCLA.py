# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:50:35 2019

@author: tiger
"""

from __future__ import print_function

import numpy as np
import random
#from utils import *
from mgcn import *

# 'breastA', breastB, DLBCLA, DLBCLB, DLBCLC, DLBCLD, MultiA, MultiB
# 'breastB', 'DLBCLA', 'DLBCLB', 'DLBCLC',
DATASET = 'DLBCLA' #,'breastB'] #, 'DLBCLA', 'DLBCLB', 'DLBCLC', 'DLBCLD']
nFiles = 6
FilePath="data/"

DROUPOUT = 0.2
'''
TUNNING_LIST = [0.001, 0.002, 0.005] # [0.0001] #,0.0001,0.00001] #,0.0001,0.00001] #,0.00005]
LEARNING_RATE_LIST = [0.01, 0.001, 0.0005] #  [0.001,0.0005,0.0001] #[0.0005, 0.0001] 
NB_EPOCH_LIST = [100, 200, 300]
'''

'''
2019-9-12 settings
TUNNING_LIST = [0.001,0.002,0.005]
LEARNING_RATE_LIST = [0.01,0.001,0.0005] 
NB_EPOCH_LIST =  [200,300,400]
'''

#2019-9-16 settings
TUNNING_LIST = [0.001]#,0.002,0.005]
LEARNING_RATE_LIST = [0.001,0.0005] 
NB_EPOCH_LIST =  [80,120,160,200,300]
DATE = "2019-9-16-7"

OUTPATH = "{}{}/".format(FilePath, DATASET)
#OUTPATH_WEIGHT = "{}{}{}".format("H:/networks/weights/", DATASET, "/")

inputs_list,y =  load_data_sets(FilePath,DATASET,nFiles)

MDGCNN1_outlist = list()    
DNN1_outlist = list()
   
for i in range(1,500):
    
    random.seed(i)
    
    randomset = np.random.rand(len(y))
    training_mask = randomset < 0.65
    test_mask = (randomset >= 0.65)
    
    out_array = np.zeros(6)
    MGCN_array = np.zeros(nFiles)
    DNN_array = np.zeros(nFiles)
    
    ###################################
    MODEL_TYPE = 'MDGCNN1'
    print(y.shape)
    print(inputs_list[0].shape)
    print(len(inputs_list))
    out_array = MDGCCN_Exp(inputs_list, y, training_mask, test_mask
                 ,DROUPOUT, LEARNING_RATE_LIST, TUNNING_LIST, NB_EPOCH_LIST, MODEL_TYPE=MODEL_TYPE)
    MGCN_array[0] = out_array[0]
	
    for j in range(1,nFiles):
        out_array = MDGCCN_Exp([inputs_list[0],inputs_list[j]], y, training_mask, test_mask
                               ,DROUPOUT, LEARNING_RATE_LIST, TUNNING_LIST, NB_EPOCH_LIST, MODEL_TYPE=MODEL_TYPE)
        MGCN_array[j] = out_array[0]
        
    MDGCNN1_outlist.append(MGCN_array)
    
    #######################################################
    MODEL_TYPE = 'DNN2'
    out_array = MDGCCN_Exp(inputs_list[0], y, training_mask, test_mask
                           ,DROUPOUT, LEARNING_RATE_LIST, TUNNING_LIST, NB_EPOCH_LIST, MODEL_TYPE=MODEL_TYPE)
    DNN_array[0] = out_array[0]
    
    for j in range(1,nFiles):
        out_array = MDGCCN_Exp(inputs_list[j], y, training_mask, test_mask
                               ,DROUPOUT, LEARNING_RATE_LIST, TUNNING_LIST, NB_EPOCH_LIST, MODEL_TYPE=MODEL_TYPE)
        DNN_array[j] = out_array[0]
    DNN1_outlist.append(DNN_array)
    ###########################################
    
    np.savetxt( "{}{}-MDGCNN.csv".format(OUTPATH,DATE), MDGCNN1_outlist, delimiter=',')  # X is an array
    np.savetxt( "{}{}-DNN.csv".format(OUTPATH,DATE), DNN1_outlist, delimiter=',')  # X is an array   