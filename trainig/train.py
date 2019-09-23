
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
K.set_learning_phase(1)

from sklearn.preprocessing import StandardScaler

def create_DNN(units, input_dim_param, lr_param):

    network = Sequential()
    network.add(Dense(units = units[0], activation = 'relu', input_dim = input_dim_param))
    network.add(Dropout(0.1))
    network.add(Dense(units = units[1], activation = 'relu'))
    network.add(Dropout(0.1))
    network.add(Dense(units = units[2], activation = 'relu'))
    network.add(Dense(units = 1, activation = 'sigmoid'))

    sgd = Adam(lr = lr_param)
    network.compile(loss = 'binary_crossentropy', optimizer = sgd)
    
    return network

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

train_data = read_features_from_dirs(['../data/features_stat_scenario1.csv','../data/features_stat_scenario9.csv'])
test_data = read_features_from_dirs(['../data/features_stat_scenario2.csv'])

x_train, y_train = train_data
x_test, y_test = test_data

y_train_array = np.array(y_train)
x_train_array = np.array(x_train)

y_test_array = np.array(y_test)
x_test_array = np.array(x_test)

scaler = StandardScaler(copy = True, with_mean = True, with_std = True)
x_train_array = scaler.fit_transform(x_train_array)
x_test_array = scaler.transform(x_test_array)

LAYERS = [256, 128, 64]
INPUT_DIM = 756
LR = [0.0002592943797404667]

for lrate in LR:
 
  nn =  create_DNN(units = LAYERS, input_dim_param = INPUT_DIM, lr_param = lrate)

  nn.fit(x_train_array, y_train_array, verbose = 0, epochs = 50, batch_size = 64,
                              shuffle = True)
                              
  probas = nn.predict_proba(x_test_array)
  predictions = (probas >= 0.5).astype(int)  

  print('F1',f1_score(y_test_array, predictions))
  fpr, tpr, thresholds = roc_curve(y_test_array, probas)
  roc_auc = auc(fpr, tpr)
  print('ROC AUC', roc_auc)

  nn.save('model_whole_scenarios19')
