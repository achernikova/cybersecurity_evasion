import numpy as np
import math
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from keras.models import load_model

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

import imageio as im
#import skimage.transform as st

from os import path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import roc_curve,auc
from scipy import interp
from itertools import cycle 
import joblib

from neris_model_data_utilities import MDModel, ROCData, generate_data, sigmoid

num_data = 814

inds = open('botnets.txt', 'r')
botnets = []
for line in inds:
    botnets = line.split(',')
botnets.remove('')

botnets_int = [8,9,10,11,12,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405]
adv_scaler = joblib.load('scaler_scenarios19.pkl')

#results without attack
model = MDModel("model_whole_scenarios19")
roc_data = ROCData(['features_stat_scenario2.csv'], adv_scaler)

predicted_labels = np.empty(num_data)
true_labels = roc_data.test_labels

for i in range (num_data):

    input_vector, target_label = generate_data(roc_data, samples = 1, start = i)
    predicted_labels[i] = sigmoid(model.model.predict(input_vector))

predicted_labels_4 = np.copy(predicted_labels)
predicted_labels_10 = np.copy(predicted_labels)
predicted_labels_16 = np.copy(predicted_labels)


dist1 = 8
dist2 = 10
dist3 = 12

b = 0
for j in range(407):

    if j in botnets_int:

        f_4 = open('prob_distance_' + str(dist1) + '_adv#' + str(botnets[b]) + '.txt', 'r')
        for line in f_4:
            predicted_labels_4[j + 407] = float(line) 

        f_10 = open('prob_distance_' + str(dist2) + '_adv#' +str(botnets[b]) +'.txt', 'r')
        for line in f_10:
            predicted_labels_10[j + 407] = float(line) 

        f_16 = open('prob_distance_' + str(dist3) + '_adv#' +str(botnets[b]) +'.txt', 'r')
        for line in f_16:
            predicted_labels_16[j + 407] = float(line) 

        

        b = b + 1

fpr_4, tpr_4, _ = roc_curve(true_labels, predicted_labels_4)
roc_auc_4 = auc(fpr_4, tpr_4)

fpr_10, tpr_10, _=roc_curve(true_labels,predicted_labels_10)
roc_auc_10 = auc(fpr_10, tpr_10)

fpr_16, tpr_16, _=roc_curve(true_labels,predicted_labels_16)
roc_auc_16 = auc(fpr_16, tpr_16)

fpr, tpr,_ = roc_curve(true_labels, predicted_labels)
roc_auc = auc(fpr, tpr)

plt.figure()

##Adding the ROC
plt.plot(fpr, tpr, lw=2.5, label='No attack (AUC = {0:0.2f})'''.format(roc_auc))
plt.plot(fpr_4, tpr_4,lw=2.5, label='Attack, d_max = 8 (AUC = {0:0.2f})'''.format(roc_auc_4))
plt.plot(fpr_10, tpr_10,lw=2.5, label='Attack, d_max = 10 (AUC = {0:0.2f})'''.format(roc_auc_10))
plt.plot(fpr_16, tpr_16, lw=2.5, label='Attack, d_max = 12 (AUC = {0:0.2f})'''.format(roc_auc_16))


##Random FPR and TPR
plt.plot([0, 1], [0, 1],  lw=2, linestyle='--')

##Title and label
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.legend(loc = 0, fontsize = 16)
ax = plt.gca()
plt.setp(ax.get_xticklabels(), fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=18)
plt.tight_layout()

plt.savefig('ROCs_8_10_12.png', dpi = 1200)
