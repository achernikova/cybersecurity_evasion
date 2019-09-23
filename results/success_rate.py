import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

NUM_BOTNETS = 407
NUM_DISTANCES = 10
NUM_BENIGN_BY_MODEL = 20

success = np.zeros(NUM_DISTANCES)
distances = np.zeros(NUM_DISTANCES)

# get malicious uindicies
inds = open('botnets.txt', 'r')

botnets = []

for line in inds:
    botnets = line.split(',')

botnets.remove('')

for  i in range(NUM_DISTANCES):

    distances[i] = (i + 1) * 2
    success_distance  = 0

    for k in botnets:

        prob = open('prob_distance_' + str(int(distances[i]))+ '_adv#' + k + '.txt', 'r')

        for line in prob:

            if float(line) < 0.5:
                success_distance += 1
        
        success[i] = success_distance

success = (success + NUM_BENIGN_BY_MODEL)/ NUM_BOTNETS

plt.xlabel("Maximum perturbation", fontsize = 18)
plt.ylabel("Attack success rate", fontsize = 18)
plt.plot(distances, success,label='Neris Attack',linewidth = 2.5, markersize = 6,marker='o')
plt.legend(loc = 0,fontsize = 16)
ax = plt.gca()
plt.setp(ax.get_xticklabels(), fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=18)
plt.tight_layout()
plt.savefig('success_rate.png', dpi = 1200)
