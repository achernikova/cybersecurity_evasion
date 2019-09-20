import pandas as pd
import tensorflow as tf
import numpy as np
np.set_printoptions(precision = 5)

import sys
import os
import time
import math

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras import optimizers
from keras import metrics

from keras import backend as K
K.set_learning_phase(1)

def read_features_from_dirs(dirs):

    LABEL_COLUMN = 'label'
    TIME_BIN_COLUMN = 'window_timestamp'
    SRC_IP_COLUMN = 'internal_ip'

    features_dict= {}  
    labels_dict= {}
    
    files_dict = []
    
    for dir1 in dirs:
        df = pd.read_csv(dir1)
        
        for index, row in df.iterrows():
            key = str(row[TIME_BIN_COLUMN])+ row[SRC_IP_COLUMN]
            label = row[LABEL_COLUMN]

            if label == 'BOTNET':
                label = 1
            else:
                label = 0
            
            if key in files_dict:
                print('Error, duplicate key!')
                exit(0)
            else:
                files_dict.append(key)
            
            if key not in features_dict:
                features_dict[key] = []
                labels_dict[key] = label
    
            numerical_features = [float(x) for x in row[:-3]]
            #print(numerical_features)
            features_dict[key].extend(numerical_features)
    
            if labels_dict[key] != label:
                print('Error, labels should be the same!')
                exit(0)
        
    combined_x = []
    combined_y = []
    
    for key in features_dict:
        combined_x.append(features_dict[key])
        combined_y.append(labels_dict[key])
    
    return combined_x, combined_y

def read_attack_features_from_dirs(dirs):

    LABEL_COLUMN = 'label'
    TIME_BIN_COLUMN = 'window_timestamp'
    SRC_IP_COLUMN = 'internal_ip'

    features_dict= {}  
    labels_dict= {}
    
    files_dict = []
    
    for dir1 in dirs:
        df = pd.read_csv(dir1)
        
        for index, row in df.iterrows():
            key = str(row[TIME_BIN_COLUMN])+ row[SRC_IP_COLUMN]
            label = row[LABEL_COLUMN]

            if label == 'BOTNET':
                label = 1
            
                if key in files_dict:
                    print('Error, duplicate key!')
                    exit(0)
                else:
                    files_dict.append(key)
                
                if key not in features_dict:
                    features_dict[key] = []
                    labels_dict[key] = label
        
                numerical_features = [float(x) for x in row[:-3]]
                #print(numerical_features)
                features_dict[key].extend(numerical_features)
        
                if labels_dict[key] != label:
                    print('Error, labels should be the same!')
                    exit(0)
        
    combined_x = []
    combined_y = []
    
    for key in features_dict:
        combined_x.append(features_dict[key])
        combined_y.append(labels_dict[key])
    
    return combined_x, combined_y

def read_roc_features_from_dirs(dirs):

    LABEL_COLUMN = 'label'
    TIME_BIN_COLUMN = 'window_timestamp'
    SRC_IP_COLUMN = 'internal_ip'

    features_dict= {}  
    labels_dict= {}
    
    files_dict = []

    tmp = 0
    
    for dir1 in dirs:
        df = pd.read_csv(dir1)
        
        for index, row in df.iterrows():
            key = str(row[TIME_BIN_COLUMN])+ row[SRC_IP_COLUMN]
            label = row[LABEL_COLUMN]

            if label == 'BACKGROUND' and tmp < 407:

                tmp += 1
                label = 0
            
                if key in files_dict:
                    print('Error, duplicate key!')
                    exit(0)

                else:
                    files_dict.append(key)
                
                if key not in features_dict:
                    features_dict[key] = []
                    labels_dict[key] = label
        
                numerical_features = [float(x) for x in row[:-3]]
                #print(numerical_features)
                features_dict[key].extend(numerical_features)
        
                if labels_dict[key] != label:
                    print('Error, labels should be the same!')
                    exit(0)
        for index, row in df.iterrows():

            key = str(row[TIME_BIN_COLUMN])+ row[SRC_IP_COLUMN]
            label = row[LABEL_COLUMN]

            if label == 'BOTNET':
                
                label = 1
            
                if key in files_dict:
                    print('Error, duplicate key!')
                    exit(0)
                    
                else:
                    files_dict.append(key)
                
                if key not in features_dict:
                    features_dict[key] = []
                    labels_dict[key] = label
        
                numerical_features = [float(x) for x in row[:-3]]
                #print(numerical_features)
                features_dict[key].extend(numerical_features)
        
                if labels_dict[key] != label:
                    print('Error, labels should be the same!')
                    exit(0)
        
    combined_x = []
    combined_y = []
    
    for key in features_dict:
        combined_x.append(features_dict[key])
        combined_y.append(labels_dict[key])
    
    return combined_x, combined_y
        
class MDModel:
    def __init__(self, restore):
        
        network = Sequential()
        network.add(Dense(units = 256, activation = 'relu', input_dim = 756))
        network.add(Dense(units = 128, activation = 'relu'))
        network.add(Dense(units = 64, activation = 'relu'))
        network.add(Dense(units = 1))   
        
        network.load_weights(restore)

        self.model = network

    def predict(self, data):
        return self.model(data)

def generate_data(data, samples, start = 0):
    
    inputs = []
    targets = []
    
    for i in range(samples):

        inputs.append(data.test_data[start + i])
        targets.append(1 - data.test_labels[start + i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets

def get_raw_delta(attack, delta, delta_sign, scaler, num_feature, attack_shape):

    raw_attack = scaler.inverse_transform(attack)
    adv = np.zeros(attack_shape)
    adv[0, num_feature] = delta * delta_sign
    adv += attack
    new_adv = scaler.inverse_transform(adv)
    new_delta = new_adv[0, num_feature] - raw_attack[0, num_feature]
    return new_delta

def get_scaled_delta(attack, delta, scaler, num_feature, attack_shape):

    raw_attack = scaler.inverse_transform(attack)
    adv = np.zeros(attack_shape)
    adv[0, num_feature] = delta
    adv +=raw_attack
    new_adv = scaler.transform(adv)
    new_delta = new_adv[0, num_feature] - attack[0, num_feature]
    return new_delta

def sigmoid(x):

    if x < -100:
        return 0 
    else:
        return 1 / (1 + math.exp(-x))

class AttackData:

    def __init__(self, dirs, scaler):

        x, y = read_attack_features_from_dirs(dirs)

        x = scaler.transform(x)

        self.test_data = x
        self.test_labels = y

class ROCData:

    def __init__(self, dirs, scaler):

        x_roc, y_roc = read_roc_features_from_dirs(dirs)
        x_roc = scaler.transform(x_roc)

        self.test_data = x_roc
        self.test_labels = y_roc

def read_min_max(min_file, max_file):

    with open(min_file, 'r') as f:
        mins = f.read()

    mins = mins.strip()
    mins = mins.replace(' ', '')
    mins_str_list = mins.split(',')
    min_features = [float(i) for i in mins_str_list]


    with open(max_file, 'r') as f:
        maxs = f.read()

    
    maxs = maxs.strip()
    maxs = maxs.replace(' ', '')
    maxs_str_list = maxs.split(',')
    max_features = [float(i) for i in maxs_str_list]


    return min_features, max_features







