from  neris_model_data_utilities import MDModel, generate_data, sigmoid, AttackData, read_min_max, get_raw_delta, get_scaled_delta
                                       
import numpy as np
import sys
import tensorflow as tf
import os
import time
import math

import pandas as pd
import random
import matplotlib.pyplot as plt

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

import joblib

np.set_printoptions(precision = 5)
np.set_printoptions(suppress = True)

class Neris_attack:

    def __init__(self):

        #features that can be updated
        self.UPDATE = np.array([3, 4, 5, 12, 13, 14, 9, 10, 11, 15, 16, 17, 24,25,26,33,34,35,30,31,32,36,37,38, 45,46,47,54,55,56,51,52,53,57,58,59,
        66,67,68,75,76,77,72,73,74,78,79,80,87,88,89,96,97,98,93,94,95,99,100,101,108,109,110,117,118,119,114,115,116,120,121,122,
        129,130,131,138,139,140,135,136,137,141,142,143,150,151,152,159,160,161,156,157,158,162,163,164,171,172,173,180,181,182,177,178,179,183,184,185,
        192,193,194,201,202,203,198,199,200,204,205,206,213,214,215,222,223,224,219,220,221,225,226,227,234,235,236,243,244,245,240,241,242,246,247,248,
        255,256,257,264,265,266,261,262,263,267,268,269,276,277,278,285,286,287,282,283,284,288,289,290,297,298,299,306,307,308,303,304,305,309,310,311,
        318,319,320,327,328,329,324,325,326,330,331,332,339,340,341,348,349,350,345,346,347,351,352,353,360,361,362,369,370,371,366,367,368,372,373,374,
        381,382,383,390,391,392,387,388,389,393,394,395,402,403,404,411,412,413,408,409,410,414,415,416,423,424,425,432,433,434,429,430,431,435,436,437,
        444,445,446,453,454,455,450,451,452,456,457,458,465,466,467,474,475,476,471,472,473,477,478,479,486,487,488,495,496,497,492,493,494,498,499,500,
        507,508,509,516,517,518,513,514,515,519,520,521,528,529,530,537,538,539,534,535,536,540,541,542,549,550,551,558,559,560,555,556,557,561,562,563,
        570,571,572,579,580,581,576,577,578,582,583,584,591,592,593,600,601,602,597,598,599,603,604,605,612,613,614,621,622,623,618,619,620,624,625,626,
        633,634,635,642,643,644,639,640,641,645,646,647,654,655,656,663,664,665,660,661,662,666,667,668,675,676,677,684,685,686,681,682,683,687,688,689,
        696,697,698,705,706,707,702,703,704,708,709,710,717,718,719,726,727,728,723,724,725,729,730,731,738,739,740,747,748,749,744,745,746,750,751,752])

        #Families: Total bytes - Min bytes - Max bytes - Total duration- Min duration - Max duration - Total packets - Min packets - Max packets
        self.FAMILIES = np.array([[3, 4, 5, 12, 13, 14, 9, 10, 11, 15, 16, 17],[24,25,26,33,34,35,30,31,32,36,37,38],[45,46,47,54,55,56,51,52,53,57,58,59],
        [66,67,68,75,76,77,72,73,74,78,79,80],[87,88,89,96,97,98,93,94,95,99,100,101],[108,109,110,117,118,119,114,115,116,120,121,122],
        [129,130,131,138,139,140,135,136,137,141,142,143],[150,151,152,159,160,161,156,157,158,162,163,164],[171,172,173,180,181,182,177,178,179,183,184,185],
        [192,193,194,201,202,203,198,199,200,204,205,206],[213,214,215,222,223,224,219,220,221,225,226,227],[234,235,236,243,244,245,240,241,242,246,247,248],
        [255,256,257,264,265,266,261,262,263,267,268,269],[276,277,278,285,286,287,282,283,284,288,289,290],[297,298,299,306,307,308,303,304,305,309,310,311],
        [318,319,320,327,328,329,324,325,326,330,331,332],[339,340,341,348,349,350,345,346,347,351,352,353],[360,361,362,369,370,371,366,367,368,372,373,374],
        [381,382,383,390,391,392,387,388,389,393,394,395],[402,403,404,411,412,413,408,409,410,414,415,416],[423,424,425,432,433,434,429,430,431,435,436,437],
        [444,445,446,453,454,455,450,451,452,456,457,458],[465,466,467,474,475,476,471,472,473,477,478,479],[486,487,488,495,496,497,492,493,494,498,499,500],
        [507,508,509,516,517,518,513,514,515,519,520,521],[528,529,530,537,538,539,534,535,536,540,541,542],[549,550,551,558,559,560,555,556,557,561,562,563],
        [570,571,572,579,580,581,576,577,578,582,583,584],[591,592,593,600,601,602,597,598,599,603,604,605],[612,613,614,621,622,623,618,619,620,624,625,626],
        [633,634,635,642,643,644,639,640,641,645,646,647],[654,655,656,663,664,665,660,661,662,666,667,668],[675,676,677,684,685,686,681,682,683,687,688,689],
        [696,697,698,705,706,707,702,703,704,708,709,710],[717,718,719,726,727,728,723,724,725,729,730,731],[738,739,740,747,748,749,744,745,746,750,751,752]])

        #Integer features
        self.integers = np.array([3, 4, 5, 9, 10, 11, 15, 16, 17,24,25,26,30,31,32,36,37,38,45,46,47,51,52,53,57,58,59,
        66,67,68,72,73,74,78,79,80,87,88,89,93,94,95,99,100,101,108,109,110,114,115,116,120,121,122,
        129,130,131,135,136,137,141,142,143,150,151,152,156,157,158,162,163,164,171,172,173,177,178,179,183,184,185,
        192,193,194,198,199,200,204,205,206,213,214,215,219,220,221,225,226,227,234,235,236,240,241,242,246,247,248,
        255,256,257,261,262,263,267,268,269,276,277,278,282,283,284,288,289,290,297,298,299,303,304,305,309,310,311,
        318,319,320,324,325,326,330,331,332,339,340,341,345,346,347,351,352,353,360,361,362,366,367,368,372,373,374,
        381,382,383,387,388,389,393,394,395,402,403,404,408,409,410,414,415,416,423,424,425,429,430,431,435,436,437,
        444,445,446,450,451,452,456,457,458,465,466,467,471,472,473,477,478,479,486,487,488,492,493,494,498,499,500,
        507,508,509,513,514,515,519,520,521,528,529,530,534,535,536,540,541,542,549,550,551,555,556,557,561,562,563,
        570,571,572,576,577,578,582,583,584,591,592,593,597,598,599,603,604,605,612,613,614,618,619,620,624,625,626,
        633,634,635,639,640,641,645,646,647,654,655,656,660,661,662,666,667,668,675,676,677,681,682,683,687,688,689,
        696,697,698,702,703,704,708,709,710,717,718,719,723,724,725,729,730,731,738,739,740,744,745,746,750,751,752])

        #Number of attack iterations
        self.NUM_ITERATIONS = 100

        #Ports that can not have TCP or UDP connections
        self.PORTS_NO_UDP = [0, 4, 5, 6, 8, 9, 14, 16]
        self.PORTS_NO_TCP = [10, 12, 13]

        SCALER_PATH = 'scaler_scenarios19.pkl'
        ATTACK_PATH = '../data/features_stat_scenario2.csv'
        self.MODEL_PATH = 'model_whole_scenarios19'

        MINIMUM_PATH = 'minimum.txt'
        MAXIMUM_PATH = 'maximum.txt'

        self.PACKETS_ID = 6
        self.BYTES_ID = 0
        self.DURATION_ID = 3
        self.TCP_ID = 9
        self.UDP_ID = 10
        self.ICMP_ID = 11

        self.MAX_PACKET_TCP_CONN = 87
        self.MAX_PACKET_UDP_CONN = 49

        self.MIN_DURATION_PACKET_TCP = 0.0001
        self.MIN_DURATION_PACKET_UDP = 0.0001
        self.MAX_DURATION_PACKET_TCP = 20.83
        self.MAX_DURATION_PACKET_UDP = 24.82

        self.MAX_BYTES_PACKET_TCP = 1292
        self.MAX_BYTES_PACKET_UDP = 1036
        self.MIN_BYTES_PACKET_TCP = 20
        self.MIN_BYTES_PACKET_UDP = 20

        self.num_features = 756
        self.num_port_families = 36

        self.adv_scaler = joblib.load(SCALER_PATH)

        self.model =  MDModel(self.MODEL_PATH) 
        attack_file = [ATTACK_PATH]
        self.data = AttackData(attack_file, self.adv_scaler)

        self.min_features, self.max_features = read_min_max(MINIMUM_PATH, MAXIMUM_PATH)

    def attack_round(self, distance, sess):

        #size of adversary
        shape = (1, self.num_features)
        #value of border features(there is no need to update them after they become 'border')
        #so we set them to some very small value, so the attack ignores them aftewards
        MAX_BORDERS = -1000000
        #maximum distance between adversary and input vector
        d_max = distance
        #number of successful attacks
        success = 0     
        #placeholder for the adversary
        attack =  tf.placeholder(tf.float32, shape = shape, name = "attack")
        #optimization function for attack
        loss_logit =  self.model.predict(attack)
        #gradient of optimization function
        gradient = tf.gradients(loss_logit, [attack])[0]   
        #code to prevent tensorflow from assigning random values to the model's weights(tensorflow - keras specificy)
        self.model.model.load_weights(self.MODEL_PATH)
        
        for i in range(len(self.data.test_data)):

            print('----------------attack ', i)
            input_vector, target_label = generate_data(self.data, samples = 1, start = i)

            #if the predicion for the input vector is 'malicious'(means that we can start running the attack)
            if sigmoid(self.model.model.predict(input_vector)) >= 0.5 :    

                #hyperparameters : number of connections that we add
                connections_ports = np.ones((2, self.num_port_families))
                #features, that already were argmaxes
                updated = []
                borders = []
                
                adversary = np.copy(input_vector)

                #number of added packets for both types of connections
                total_packets_udp = np.zeros(len(self.FAMILIES))
                total_packets_tcp  = np.zeros(len(self.FAMILIES))

                for j in range(self.NUM_ITERATIONS):

                    raw_adversary = self.adv_scaler.inverse_transform(adversary)
                    res_grad = sess.run(gradient, feed_dict={attack: adversary})
                    abs_grad = abs(res_grad)
                    abs_grad_update = np.copy(abs_grad)

                    abs_grad[:,borders] = MAX_BORDERS
                    arg_max = np.argmax(abs_grad)
                    borders.append(arg_max)
       
                    if arg_max in self.UPDATE:
                        if arg_max not in updated:
                            updated.append(arg_max)

                            #get indicies of the family we want to update
                            for k in range(len(self.FAMILIES)):
                                if arg_max in self.FAMILIES[k]:

                                    inds = self.FAMILIES[k]
                                    deltas_update = abs_grad_update[0, inds]
                                    delta_signs = np.zeros((deltas_update.shape))
                                    for l in range(len(inds)):
                                        if(res_grad[0, inds[l]] < 0):
                                            delta_signs[l] = -1 
                                        else:
                                            delta_signs[l] = 1

                            port_id = int(np.floor((inds[self.BYTES_ID] - 3)/21))
                            input_vector_raw = self.adv_scaler.inverse_transform(input_vector)

                            current_scaled_adversary = np.copy(adversary)
                            current_raw_adversary = self.adv_scaler.inverse_transform(current_scaled_adversary)

                            #update connections by the value from connection_ports(hyperparameter)
                            for k in [self.TCP_ID, self.UDP_ID]:   
                                if k == self.TCP_ID and port_id not in self.PORTS_NO_TCP:                           
                                    if delta_signs[k] < 0:
                                        new_conn_value = self.update_from_conn_up(current_raw_adversary, inds, delta_signs[k] * connections_ports[k - 9, int(port_id)], k)
                                        current_raw_adversary[0, inds[k]] = new_conn_value
                                        connections_ports[k - 9, int(port_id)] += 10

                                elif k == self.UDP_ID and port_id not in self.PORTS_NO_UDP:
                                    if delta_signs[k] < 0:
                                        new_conn_value = self.update_from_conn_up(current_raw_adversary, inds, delta_signs[k] * connections_ports[k - 9, int(port_id)], k)
                                        current_raw_adversary[0, inds[k]] = new_conn_value
                                        connections_ports[k - 9, int(port_id)] += 10

                            #total number of added connections at this iteraaion of particulat attack round  
                            iter_conns_tcp = current_raw_adversary[0, inds[self.TCP_ID]] - raw_adversary[0, inds[self.TCP_ID]] 
                            iter_conns_udp = current_raw_adversary[0, inds[self.UDP_ID]] - raw_adversary[0, inds[self.UDP_ID]]

                            #total number of added connections
                            total_conns_tcp = current_raw_adversary[0, inds[self.TCP_ID]] - input_vector_raw[0, inds[self.TCP_ID]]
                            total_conns_udp = current_raw_adversary[0, inds[self.UDP_ID]] - input_vector_raw[0, inds[self.UDP_ID]]

                            #update scaled feature vector after connections update
                            current_scaled_adversary = self.adv_scaler.transform(current_raw_adversary)
                            distance_tmp = np.linalg.norm(current_scaled_adversary - input_vector)

                            if iter_conns_tcp + iter_conns_udp  > 0 and distance_tmp <= d_max:

                                pbd_scaled_adversary = np.copy(current_scaled_adversary)
                                pbd_raw_adversary = np.copy(current_raw_adversary)

                                f_id = self.PACKETS_ID

                                res_grad = sess.run(gradient, feed_dict={attack: pbd_scaled_adversary})
                                abs_grad = abs(res_grad)
                                delta_packets = abs_grad[0, inds[f_id]]
                                if(res_grad[0, inds[f_id]] < 0):
                                    delta_sign = -1 
                                else:
                                    delta_sign = 1

                                raw_delta = np.nan_to_num(get_raw_delta(pbd_scaled_adversary, delta_packets, delta_sign, self.adv_scaler, inds[f_id], shape))
                                if inds[f_id] in self.integers: 
                                    if delta_sign < 0:
                                        raw_delta = math.ceil(raw_delta)
                                    else:
                                        raw_delta = math.floor(raw_delta)
                                raw_delta_packets = abs(raw_delta)

                                current_raw = np.copy(pbd_raw_adversary)
                                    
                                #corner case: delta_packets is 0, then we add minimum possible number of packets per UDP.TCP connection, which in our case is 2
                                if raw_delta_packets == 0:

                                    raw_delta =(iter_conns_tcp + iter_conns_udp) * -2

                                    iter_packets_udp = iter_conns_tcp * 2                            
                                    iter_packets_tcp = iter_conns_udp * 2

                                    total_packets_udp[port_id] +=iter_packets_udp
                                    total_packets_tcp[port_id] +=iter_packets_tcp

                                    #increase the total value of packets plus project mathematical dependencies(min/ max packets per UDP/TCP connection) if needed
                                    pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]]  = self.update_from_total_up(pbd_raw_adversary, input_vector_raw, inds,
                                                                                                        raw_delta,  f_id,total_conns_udp, total_conns_tcp, total_packets_udp[port_id], total_packets_tcp[port_id])

                                    pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)
                                    distance = np.linalg.norm(pbd_scaled_adversary - input_vector)

                                    #check resulting distance, and if we are inside L2 norm ball proceed to updating bytes and duration
                                    if distance < d_max:

                                        pbd_raw_adversary = self.update_bytes(pbd_scaled_adversary, pbd_raw_adversary,input_vector_raw, input_vector, inds, shape,                                                                
                                                                        iter_conns_tcp, iter_conns_udp, iter_packets_udp, iter_packets_tcp,  total_packets_udp[port_id], total_packets_tcp[port_id],
                                                                        d_max, sess, gradient, attack)

                                        pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)
                                        distance = np.linalg.norm(pbd_scaled_adversary - input_vector)
                                        
                                        #check resulting distance, and if we are outside L2 norm ball reverse changes
                                        if distance > d_max:

                                            pbd_scaled_adversary = adversary
                                            pbd_raw_adversary = self.adv_scaler.inverse_transform(pbd_scaled_adversary)
                                            total_packets_udp[port_id] -=iter_packets_udp
                                            total_packets_tcp[port_id] -=iter_packets_tcp

                                    #check resulting distance, and if we are outside L2 norm ball reverse changes
                                    else:

                                        pbd_scaled_adversary = adversary
                                        pbd_raw_adversary = self.adv_scaler.inverse_transform(pbd_scaled_adversary)
                                        total_packets_udp[port_id] -=iter_packets_udp
                                        total_packets_tcp[port_id] -=iter_packets_tcp

                                #normal case, conducting binary search on representative features, which is 'total_packets_sent' in our case
                                else:

                                    while raw_delta_packets != 0:
                                        
                                        new_packets = current_raw[0, inds[f_id]] - raw_delta_packets * delta_sign

                                        if new_packets < input_vector_raw[0, inds[f_id]] :

                                            new_packets = input_vector_raw[0, inds[f_id]] 
                                            raw_delta_packets =  current_raw[0, inds[f_id]] - new_packets

                                        #resulting number of added packets after possible update
                                        total_packets = new_packets - input_vector_raw[0, inds[f_id]] 
                                    
                                        #overall number of added tcp/udp connections
                                        total_conns_tcp = pbd_raw_adversary[0, inds[self.TCP_ID]] - input_vector_raw[0, inds[self.TCP_ID]]
                                        total_conns_udp = pbd_raw_adversary[0, inds[self.UDP_ID]] - input_vector_raw[0, inds[self.UDP_ID]]

                                        total_conns = total_conns_tcp + total_conns_udp
                                    
                                        #Projection on lower bound of packets per connection(physical constraint)
                                        if total_packets  < total_conns * 2:
                                            
                                            raw_delta = total_conns * -2

                                            iter_packets_udp = total_conns_udp * 2
                                            iter_packets_tcp = total_conns_tcp * 2

                                            #udpate total value of sent packets and adjust mathematical dependencies (min/max sent packets per connections)if needed
                                            pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]]  = self.update_from_total_up(input_vector_raw, input_vector_raw,
                                                                                                                inds, raw_delta, f_id, total_conns_udp, total_conns_tcp,iter_packets_udp, iter_packets_tcp)

                                            pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)
                                            distance = np.linalg.norm(pbd_scaled_adversary - input_vector)
                                            
                                            #check resulting distance, if feature vector is inside L2 norm ball, proceed to update bytes and packets
                                            if distance < d_max:

                                                pbd_raw_adversary = self.update_bytes(pbd_scaled_adversary, pbd_raw_adversary, input_vector_raw, input_vector, inds, shape,
                                                                                iter_conns_tcp,iter_conns_udp, iter_packets_udp, iter_packets_tcp, iter_packets_udp, iter_packets_tcp,
                                                                                d_max, sess, gradient, attack)

                                                pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)
                                                distance = np.linalg.norm(pbd_scaled_adversary - input_vector)
                                                #check resulting distance, if outside L2 norm ball, reverse changes
                                                if distance > d_max:

                                                    pbd_scaled_adversary = adversary
                                                    pbd_raw_adversary = self.adv_scaler.inverse_transform(pbd_scaled_adversary)

                                                else:

                                                    total_packets_udp[port_id] = iter_packets_udp
                                                    total_packets_tcp[port_id] = iter_packets_tcp

                                            #check resulting distance, if outside L2 norm ball, reverse changes
                                            else:

                                                pbd_scaled_adversary = adversary
                                                pbd_raw_adversary = self.adv_scaler.inverse_transform(pbd_scaled_adversary)
                                        
                                            raw_delta_packets = 0
                                        
                                        #Projection on upper bound with binary search
                                        elif total_packets >  self.MAX_PACKET_TCP_CONN * total_conns_tcp +  self.MAX_PACKET_UDP_CONN * total_conns_udp:

                                            raw_delta = (self.MAX_PACKET_TCP_CONN * total_conns_tcp +  self.MAX_PACKET_UDP_CONN * total_conns_udp) * -1

                                            iter_packets_udp = self.MAX_PACKET_UDP_CONN * total_conns_udp
                                            iter_packets_tcp = self.MAX_PACKET_TCP_CONN * total_conns_tcp 

                                            #udpate total number of sent packets and adjust mathematical dependencies
                                            pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]]  = self.update_from_total_up(input_vector_raw, input_vector_raw,
                                                                                                            inds,raw_delta,  f_id, total_conns_udp, total_conns_tcp, iter_packets_udp, iter_packets_tcp)

                                            pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)

                                            #update bytes and duration
                                            pbd_raw_adversary = self.update_bytes(pbd_scaled_adversary, pbd_raw_adversary, input_vector_raw, input_vector,   inds, shape,                                                                
                                                                        iter_conns_tcp, iter_conns_udp,  iter_packets_udp, iter_packets_tcp,iter_packets_udp, iter_packets_tcp,
                                                                        d_max, sess, gradient, attack)

                                            pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)
                                            distance_tmp = np.linalg.norm(pbd_scaled_adversary - input_vector)
                                            current_prob = sigmoid(self.model.model.predict(pbd_scaled_adversary))
                                            #perform binary search if we are outside L2 norm ball or probability is > 0.5
                                            if distance_tmp > d_max or current_prob > 0.5:

                                                raw_delta_packets  = math.floor(raw_delta_packets/2)

                                                if raw_delta_packets == 0 :                                                
                                                    pbd_raw_adversary = raw_adversary
                                                    pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)

                                            else:
                                                raw_delta_packets = 0
                                                total_packets_udp[port_id] = iter_packets_udp
                                                total_packets_tcp[port_id] = iter_packets_tcp

                                        #Feasible update with binary search over delta for 'total packets sent' which is representative feauture
                                        else:                                    
                                            if delta_sign < 0:
                                                
                                                #if we have TCP and UDP types of connections then we need to spread tha packets between connections                                                
                                                if iter_conns_udp != 0 and iter_conns_tcp != 0:

                                                    tcp_min_packets_before = (total_conns_tcp - iter_conns_tcp)* 2
                                                    tcp_max_packets_before = (total_conns_tcp - iter_conns_tcp) * self.MAX_PACKET_TCP_CONN
                                                    udp_min_packets_before = (total_conns_udp - iter_conns_udp) * 2
                                                    udp_max_packets_before = (total_conns_udp - iter_conns_udp) * self.MAX_PACKET_UDP_CONN
                                                
                                                    tcp_packets_before = total_packets_tcp[port_id]
                                                    udp_packets_before = total_packets_udp[port_id]
                                                    
                                                    tcp_min_packets = iter_conns_tcp * 2  +   tcp_min_packets_before -  tcp_packets_before 
                                                    tcp_max_packets = iter_conns_tcp * self.MAX_PACKET_TCP_CONN + tcp_max_packets_before - tcp_packets_before 
                                                    udp_min_packets = iter_conns_udp * 2 +  udp_min_packets_before - udp_packets_before
                                                    udp_max_packets = iter_conns_udp * self.MAX_PACKET_UDP_CONN + udp_max_packets_before - udp_packets_before                                            
                                                                                                    
                                                    lower_rand_udp = round(max(udp_min_packets, raw_delta_packets - tcp_max_packets))
                                                    upper_rand_udp = round(min(udp_max_packets, raw_delta_packets - tcp_min_packets))

                                                    iter_packets_udp = random.randint(lower_rand_udp, upper_rand_udp + 1)
                                                    iter_packets_tcp = raw_delta_packets - iter_packets_udp

                                                #if we only have tcp connections
                                                elif iter_conns_tcp !=0:

                                                    iter_packets_udp = 0
                                                    iter_packets_tcp = raw_delta_packets
                                                #if we only have udp connections
                                                elif iter_conns_udp != 0:

                                                    iter_packets_tcp = 0
                                                    iter_packets_udp = raw_delta_packets

                                                total_packets_udp[port_id] += iter_packets_udp
                                                total_packets_tcp[port_id] += iter_packets_tcp                                       

                                                #udpate total number of sent packets and adjust mathematical depedencies(min/msx packets sent per connection)
                                                pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]]  = self.update_from_total_up(current_raw, input_vector_raw,
                                                                                                                    inds, raw_delta_packets * delta_sign,  f_id, total_conns_udp, total_conns_tcp, 
                                                                                                                    total_packets_udp[port_id], total_packets_tcp[port_id])
                                   
                                                pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)
                                                #update bytes and duration
                                                pbd_raw_adversary = self.update_bytes( pbd_scaled_adversary, pbd_raw_adversary, input_vector_raw, input_vector,inds, shape,
                                                                            iter_conns_tcp, iter_conns_udp,  iter_packets_udp, iter_packets_tcp, total_packets_udp[port_id], total_packets_tcp[port_id],
                                                                            d_max, sess, gradient, attack)

                                            elif  pbd_raw_adversary[0, inds[f_id]] != input_vector_raw[0, inds[f_id]] and delta_sign > 0:

                                                delta_to_increase = new_packets - input_vector_raw[0, inds[f_id]] 
                                                #if we have TCP and UDP types of connections then we need to spread tha packets between connections
                                                if iter_conns_tcp != 0 and iter_conns_udp != 0:

                                                    tcp_min_packets = total_conns_tcp * 2
                                                    tcp_max_packets = total_conns_tcp * self.MAX_PACKET_TCP_CONN
                                                    udp_min_packets = total_conns_udp * 2
                                                    udp_max_packets = total_conns_udp * self.MAX_PACKET_UDP_CONN
                                                                                                    
                                                    lower_rand_udp = max(udp_min_packets,delta_to_increase - tcp_max_packets)
                                                    upper_rand_udp = min(udp_max_packets,delta_to_increase - tcp_min_packets)

                                                    iter_packets_udp = random.randint(lower_rand_up, upper_rand_udp + 1)
                                                    iter_packets_tcp = delta_to_increase - iter_packets_udp

                                                #if we have only tcp connections
                                                elif iter_conns_tcp != 0:
                                                    iter_packets_udp = 0
                                                    iter_packets_tcp = delta_to_increase

                                                 #if we have only udp connections   
                                                elif iter_conns_udp != 0 :
                                                    iter_packets_tcp = 0
                                                    iter_packets_udp = delta_to_increase

                                                #update total number of sent packets and adjust mathematical dependencies(min/ max packets per connection)
                                                pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]]  = self.update_from_total_up(input_vector_raw, input_vector_raw, inds, delta_to_increase * -1, 
                                                                                                                        f_id, total_conns_udp, total_conns_tcp,iter_packets_udp, iter_packets_tcp)
                                                 
                       
                                                pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)
                                                #update bytes and duration
                                                pbd_raw_adversary = self.update_bytes( pbd_scaled_adversary, pbd_raw_adversary, input_vector_raw, input_vector,
                                                                            inds, shape, iter_conns_tcp, iter_conns_udp,  iter_packets_udp, iter_packets_tcp,
                                                                            iter_packets_udp, iter_packets_tcp,d_max, sess, gradient, attack)

                                            pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)
                                            distance_tmp = np.linalg.norm(pbd_scaled_adversary - input_vector)
                                            current_prob = sigmoid(self.model.model.predict(pbd_scaled_adversary))
                                            #perform binary search if we are outside L2 norma ball pt resulting probability is > 0.5
                                            if distance_tmp > d_max or current_prob > 0.5:

                                                raw_delta_packets  = math.floor(raw_delta_packets/2)

                                                if delta_sign < 0:

                                                    total_packets_udp[port_id] -= iter_packets_udp
                                                    total_packets_tcp[port_id] -= iter_packets_tcp

                                                if raw_delta_packets == 0 :

                                                    pbd_raw_adversary = raw_adversary
                                                    pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)

                                            else:
                                                raw_delta_packets = 0

                                                if delta_sign > 0:

                                                    total_packets_udp[port_id] = iter_packets_udp
                                                    total_packets_tcp[port_id] = iter_packets_tcp

                            else:
                                
                                pbd_raw_adversary = raw_adversary
                                pbd_scaled_adversary = adversary

                            adversary = self.adv_scaler.transform(pbd_raw_adversary)

                    after_pred = sigmoid(self.model.model.predict(adversary))
                                
                    fdist = open('../results/res_distance_distance_' + str(d_max) + '_adv#' + str(i) + '.txt', 'w')
                    fprob = open('../results/prob_distance_' + str(d_max) + '_adv#' + str(i) + '.txt', 'w')
                    fdelta = open('../results/delta_distance'+str(d_max) + '_adv#' + str(i)+'.txt', 'w')

                    raw_write_input = self.adv_scaler.inverse_transform(input_vector)
                    raw_write_adv = self.adv_scaler.inverse_transform(adversary)
                    raw_delta = raw_write_adv - raw_write_input

                    for d in range(len(raw_delta[0])):
                    
                        if (raw_delta[0, d]) !=0:
                            fdelta.write(str(d) + ': '+ str(raw_write_input[0, d]) +  ' ' +str(raw_delta[0, d]) + ' ' + str(raw_write_adv[0, d]) + '\n')
                                
                    fdist.write(str(np.linalg.norm(adversary - input_vector)))
                    fprob.write(str(after_pred))

                    fdelta.close()
                    fdist.close()
                    fprob.close()

                    if after_pred < 0.5:
                        success += 1
                        break  


    def update_from_conn_up(self, adversary, inds, raw_delta, conn_id):

        conn_f =  adversary[0, inds[conn_id]]
        max_conn_f = self.max_features[inds[conn_id]]
        new_conn_f = conn_f

        new_conn_f = conn_f - raw_delta

        if new_conn_f > max_conn_f :
            new_conn_f = max_conn_f 

        return new_conn_f

    def update_from_conn_down(self, adversary, input_vector, inds, raw_delta, conn_id):

        conn_f = adversary[0, inds[conn_id]]
        new_conn_f = conn_f

        initial_conn_f = input_vector[0, inds[conn_id]]

        new_conn_f = conn_f - raw_delta

        if new_conn_f <  initial_conn_f:
            new_conn_f = initial_conn_f

        adversary[0, inds[conn_id]] = new_conn_f

        return new_conn_f

    def update_bytes(self, pbd_scaled_adversary, pbd_raw_adversary, input_vector_raw, input_vector,inds, shape,
                    iter_conns_tcp, iter_conns_udp, iter_packets_udp, iter_packets_tcp, total_packets_udp, total_packets_tcp, d_max, sess, gradient, attack):

        f_id = self.BYTES_ID
        res_grad = sess.run(gradient, feed_dict={attack: pbd_scaled_adversary})
        abs_grad = abs(res_grad)
        delta_bytes = abs_grad[0, inds[f_id]]

        if(res_grad[0, inds[f_id]] < 0):
            delta_sign = -1 
        else:
            delta_sign = 1

        raw_delta = np.nan_to_num(get_raw_delta(pbd_scaled_adversary, delta_bytes, delta_sign, self.adv_scaler, inds[f_id], shape))

        if inds[f_id] in self.integers: 
            if delta_sign < 0:
                raw_delta = math.ceil(raw_delta)
            else:
                raw_delta = math.floor(raw_delta)

        delta_bytes = abs(raw_delta)

        new_bytes = pbd_raw_adversary[0, inds[f_id]] - raw_delta

        if new_bytes < input_vector_raw[0, inds[f_id]] :
            new_bytes = input_vector_raw[0, inds[f_id]] 
            delta_bytes = pbd_raw_adversary[0, inds[f_id]] - new_bytes 
        
        total_bytes = new_bytes - input_vector_raw[0, inds[f_id]] 
        total_packets = total_packets_udp + total_packets_tcp

        total_conns_udp = pbd_raw_adversary[0, inds[self.UDP_ID]] - input_vector_raw[0, inds[self.UDP_ID]]
        total_conns_tcp = pbd_raw_adversary[0, inds[self.TCP_ID]] - input_vector_raw[0, inds[self.TCP_ID]]

        #upper boundary
        if total_bytes > total_packets_tcp * self.MAX_BYTES_PACKET_TCP + total_packets_udp * self.MAX_BYTES_PACKET_UDP:
            delta_bytes  = (total_packets_tcp * self.MAX_BYTES_PACKET_TCP + total_packets_udp * self.MAX_BYTES_PACKET_UDP ) * -1

            pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]] = self.update_from_total_up(input_vector_raw,
                                                                                input_vector_raw, inds, delta_bytes, f_id, total_conns_udp, total_conns_tcp,
                                                                                total_packets_udp, total_packets_tcp)

            pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)

            pbd_raw_adversary = self.update_duration( pbd_raw_adversary, pbd_scaled_adversary, input_vector_raw, input_vector, shape, inds, iter_conns_tcp,
                                                     iter_conns_udp,  iter_packets_udp, iter_packets_tcp,
                                                    total_packets_udp, total_packets_tcp, sess, gradient, attack, d_max)
        #lower boundary
        elif total_bytes < total_packets * self.MIN_BYTES_PACKET_TCP:
            delta_bytes = total_packets * self.MIN_BYTES_PACKET_TCP * -1

            pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]] = self.update_from_total_up(input_vector_raw, 
                                                                                input_vector_raw, inds, delta_bytes,
                                                                                 f_id,total_conns_udp,  total_conns_tcp,total_packets_udp, total_packets_tcp)
        
            pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)

            pbd_raw_adversary = self.update_duration(pbd_raw_adversary, pbd_scaled_adversary,input_vector_raw, input_vector, 
                                                    shape, inds, iter_conns_tcp, iter_conns_udp,
                                                    iter_packets_tcp, iter_packets_tcp,total_packets_udp, total_packets_tcp, sess, gradient, attack, d_max)
        #feasible update
        else:
            if delta_sign < 0:
        
                pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]]= self.update_from_total_up(pbd_raw_adversary,
                                                                                     input_vector_raw, inds, delta_bytes * delta_sign,
                                                                                    f_id, total_conns_udp, total_conns_tcp, total_packets_udp, total_packets_tcp)
       
            elif  pbd_raw_adversary[0, inds[f_id]] != input_vector_raw[0, inds[f_id]] and delta_sign > 0:

                pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]] = self.update_from_total_down(pbd_raw_adversary,
                                                                                         input_vector_raw, inds, delta_bytes * delta_sign,
                                                                                       f_id, total_conns_udp, total_conns_tcp, total_packets_udp, total_packets_tcp)
   
            pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)

            pbd_raw_adversary = self.update_duration( pbd_raw_adversary, pbd_scaled_adversary, input_vector_raw, input_vector, shape, inds, iter_conns_tcp, iter_conns_udp,
                                                    iter_packets_udp, iter_packets_tcp, total_packets_udp, iter_packets_tcp, sess, gradient, attack, d_max)

        return pbd_raw_adversary

    def update_duration(self, pbd_raw_adversary, pbd_scaled_adversary,input_raw_vector, input_vector, shape, inds,
                        iter_conns_tcp, iter_conns_udp, iter_packets_udp, iter_packets_tcp,
                        total_packets_udp, total_packets_tcp, sess, gradient, attack,d_max):
       
        res_grad = sess.run(gradient, feed_dict={attack: pbd_scaled_adversary})
        f_id = self.DURATION_ID
        abs_grad = abs(res_grad)
        delta_duration = abs_grad[0, inds[f_id]]

        if(res_grad[0, inds[f_id]] < 0):
            delta_sign = -1 
        else:
            delta_sign = 1
        raw_delta = np.nan_to_num(get_raw_delta(pbd_scaled_adversary, delta_duration, delta_sign, self.adv_scaler, inds[f_id], shape))

        if inds[f_id] in self.integers: 
            if delta_sign < 0:
                raw_delta = math.ceil(raw_delta)
            else:
                raw_delta = math.floor(raw_delta)

        delta_duration = abs(raw_delta)

        new_duration = pbd_raw_adversary[0, inds[f_id]] - raw_delta

        if new_duration < input_raw_vector[0, inds[f_id]] :
            new_duration = input_raw_vector[0, inds[f_id]] 
            delta_duration = pbd_raw_adversary[0, inds[f_id]] - new_duration
        
        total_conns_udp = pbd_raw_adversary[0, inds[self.UDP_ID]] - input_raw_vector[0, inds[self.UDP_ID]]
        total_conns_tcp = pbd_raw_adversary[0, inds[self.TCP_ID]] - input_raw_vector[0, inds[self.TCP_ID]]

        total_duration = new_duration - input_raw_vector[0, inds[f_id]] 
        total_packets = total_packets_udp + total_packets_tcp

        #lower boundary
        if total_duration < total_packets* self.MIN_DURATION_PACKET_TCP:

            delta_duration = total_packets * self.MIN_DURATION_PACKET_TCP * -1

            #udpate dependencies        
            pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]] = self.update_from_total_up(input_raw_vector,
                                                                                 input_raw_vector, inds, delta_duration, f_id, total_conns_udp,  total_conns_tcp,
                                                                                  total_packets_udp, total_packets_tcp)
        
        
        #upper boundary
        elif total_duration > total_packets_tcp * self.MAX_DURATION_PACKET_TCP + total_packets_udp * self.MAX_DURATION_PACKET_UDP:
            delta_duration = (total_packets_tcp * self.MAX_DURATION_PACKET_TCP + total_packets_udp * self.MAX_DURATION_PACKET_UDP) * -1

            #udpate dependencies
            pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]]  = self.update_from_total_up(input_raw_vector, 
                                                                            input_raw_vector, inds, delta_duration, f_id,total_conns_udp,  total_conns_tcp,
                                                                             total_packets_udp, total_packets_tcp)
        #feasible update
        else:
            if delta_sign < 0:
                pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]]  = self.update_from_total_up(pbd_raw_adversary, input_raw_vector, inds, delta_duration * delta_sign,
                                                                                    f_id,total_conns_udp,  total_conns_tcp, total_packets_udp, total_packets_tcp)
                
        
            elif  pbd_raw_adversary[0, inds[f_id]] != input_raw_vector[0, inds[f_id]] and delta_sign > 0:
                pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]] = self.update_from_total_down(pbd_raw_adversary, input_raw_vector, inds, delta_duration * delta_sign,
                                                                                      f_id, total_conns_udp,  total_conns_tcp, total_packets_udp, total_packets_tcp)

        return pbd_raw_adversary

    def update_from_total_up(self, adversary, input_raw, inds, raw_delta, total_id, total_conns_udp, total_conns_tcp, total_packets_udp,  total_packets_tcp):

        np.random.seed(1)
        
        total_f = adversary[0, inds[total_id]]
        max_total_f = self.max_features[inds[total_id]]
        new_total_f = total_f

        max_f = input_raw[0, inds[total_id + 2]]
        max_max_f = self.max_features[inds[total_id + 2]]
        new_max_f = max_f

        min_f = input_raw[0, inds[total_id + 1]]
        if min_f < 0:
            min_f = 0
        min_min_f = self.min_features[inds[total_id + 1]]
        new_min_f = min_f

        #update total - increase
        new_total_f = total_f - raw_delta

        ###########################################Min/Max

        #get total number of udp and tcp connections OVERALL
        total_conns_tcp = round(total_conns_tcp)
        total_conns_udp = round(total_conns_udp)

        #get total amount of feature added OVERALL
        total_added = new_total_f - input_raw[0, inds[total_id]]

        if total_id == self.PACKETS_ID:
            new_max_f, new_min_f = self.update_max_min_packets(total_added, total_conns_tcp, total_conns_udp, total_packets_tcp, total_packets_udp, new_max_f, new_min_f)
           
        if total_id == self.BYTES_ID:
            new_max_f, new_min_f = self.update_max_min_bytes(total_added, total_conns_tcp, total_conns_udp, total_packets_tcp, total_packets_udp,new_max_f, new_min_f, raw_delta)

        if total_id == self.DURATION_ID:
            new_max_f, new_min_f = self.update_max_min_duration(total_added, total_conns_tcp, total_conns_udp, total_packets_tcp, total_packets_udp, new_max_f, new_min_f, raw_delta)

        return new_total_f, new_min_f, new_max_f

    def update_from_total_down(self, adversary_raw, input_raw, inds, raw_delta, total_id, total_conns_udp, total_conns_tcp, total_packets_udp, total_packets_tcp):

        np.random.seed(1)

        input_total = input_raw[0, inds[total_id]]
        total_f = adversary_raw[0, inds[total_id]]
        new_total_f = total_f

        max_f = input_raw[0, inds[total_id + 2]]
        max_max_f = self.max_features[inds[total_id + 2]]
        new_max_f = max_f

        min_f = input_raw[0, inds[total_id + 1]]
        if min_f < 0:
            min_f = 0
        min_min_f = self.min_features[inds[total_id + 1]]
        new_min_f = min_f

        #decrease total
        new_total_f = total_f - raw_delta

        if total_id == self.PACKETS_ID:
            # check if greater than initial + 20 bytes(min bytes per packet)
            if new_total_f < input_total + 2:
                new_total_f = input_total + 2

        elif total_id == self.BYTES_ID:
            # check if greater than initial + 20 bytes(min bytes per packet)
            if new_total_f < input_total + 20:
                new_total_f = input_total + 20

        elif total_id == self.DURATION_ID:
            # check if greater than initial + min duration per packet 
            if new_total_f < input_total + 0.0001:
                new_total_f = input_total + 0.0001

        ###########################################Min/Max
        total_conns_tcp = round(total_conns_tcp)
        total_conns_udp = round(total_conns_udp)

        #get total amount of feature added
        total_added = new_total_f - input_raw[0, inds[total_id]]

        if total_id == self.PACKETS_ID:
            new_max_f, new_min_f = self.update_max_min_packets(total_added, total_conns_tcp, total_conns_udp, total_packets_tcp, total_packets_udp, new_max_f, new_min_f)
       
        if total_id == self.BYTES_ID:
            new_max_f, new_min_f = self.update_max_min_bytes(total_added, total_conns_tcp, total_conns_udp, total_packets_tcp, total_packets_udp, new_max_f, new_min_f, raw_delta)

        if total_id == self.DURATION_ID:
            new_max_f, new_min_f = self.update_max_min_duration(total_added, total_conns_tcp, total_conns_udp, total_packets_tcp, total_packets_udp,new_max_f, new_min_f, raw_delta)

        return new_total_f, new_min_f, new_max_f

    def update_max_min_packets(self, total_added, total_packets_tcp, total_packets_udp, total_conns_tcp, total_conns_udp,new_max_f, new_min_f):

        total_added = round(total_added)
        
        if total_conns_tcp != 0:

            if total_conns_tcp != 1:
                min_packets_per_connection = math.floor(total_packets_tcp/total_conns_tcp)
                max_packets_per_connection = total_packets_tcp - (total_conns_tcp - 1) * min_packets_per_connection

            else:
                min_packets_per_connection = total_packets_tcp
                max_packets_per_connection = total_packets_tcp

            if max_packets_per_connection > new_max_f:
                new_max_f = max_packets_per_connection

            elif min_packets_per_connection < new_min_f:
                new_min_f = min_packets_per_connection
        
        if total_conns_udp != 0:

            if total_conns_udp != 1:
                min_packets_per_connection = math.floor(total_packets_udp/total_conns_udp)
                max_packets_per_connection = total_packets_udp - (total_conns_udp - 1) * min_packets_per_connection
        
            else:
                min_packets_per_connection = total_packets_udp
                max_packets_per_connection = total_packets_udp 
        
            if max_packets_per_connection > new_max_f:
                new_max_f = max_packets_per_connection
        
            elif min_packets_per_connection < new_min_f:
                new_min_f = min_packets_per_connection

        return new_max_f, new_min_f

    def update_max_min_bytes(self, total_added, total_conns_tcp, total_conns_udp, total_packets_tcp, total_packets_udp,new_max_f, new_min_f, raw_delta):

        total_added = round(total_added)

        #both tpes of connection were added
        if total_conns_tcp !=0 and total_conns_udp !=0 :

            #get udp and tcp packets
            total_udp_added_packets = total_packets_udp
            total_tcp_added_packets = total_packets_tcp

            #lower bound
            if abs(raw_delta)  == total_udp_added_packets *  self.MIN_BYTES_PACKET_UDP  + total_tcp_added_packets * self.MIN_BYTES_PACKET_TCP  :
    
                if total_conns_tcp != 1:
                    min_tcp_per_connection = math.floor(total_tcp_added_packets * self.MIN_BYTES_PACKET_TCP/ total_conns_tcp)
                    max_tcp_per_connection = total_tcp_added_packets * self.MIN_BYTES_PACKET_TCP - (total_conns_tcp - 1) * min_tcp_per_connection

                else:
                    min_tcp_per_connection = total_tcp_added_packets * self.MIN_BYTES_PACKET_TCP
                    max_tcp_per_connection = total_tcp_added_packets * self.MIN_BYTES_PACKET_TCP

                if total_conns_udp != 1:
                    min_udp_per_connection = math.floor(total_udp_added_packets * self.MIN_BYTES_PACKET_UDP/ total_conns_udp)
                    max_udp_per_connection = total_udp_added_packets * self.MIN_BYTES_PACKET_UDP - (total_conns_udp - 1) * min_udp_per_connection

                else:
                    min_udp_per_connection = total_udp_added_packets * self.MIN_BYTES_PACKET_UDP
                    max_udp_per_connection = total_udp_added_packets * self.MIN_BYTES_PACKET_UDP

            #upper bound
            elif abs(raw_delta) == total_udp_added_packets *  self.MAX_BYTES_PACKET_UDP + total_tcp_added_packets * self.MAX_BYTES_PACKET_TCP :
        
                if total_conns_tcp != 1:
                    min_tcp_per_connection = math.floor(total_tcp_added_packets * self.MAX_BYTES_PACKET_TCP/ total_conns_tcp)
                    max_tcp_per_connection = total_tcp_added_packets * self.MAX_BYTES_PACKET_TCP - (total_conns_tcp - 1) * min_tcp_per_connection

                else:
                    min_tcp_per_connection = total_tcp_added_packets * self.MAX_BYTES_PACKET_TCP
                    max_tcp_per_connection = total_tcp_added_packets * self.MAX_BYTES_PACKET_TCP


                if total_conns_udp != 1:
                    min_udp_per_connection = math.floor(total_udp_added_packets * self.MAX_BYTES_PACKET_UDP/ total_conns_udp)
                    max_udp_per_connection = total_udp_added_packets * self.MAX_BYTES_PACKET_UDP - (total_conns_udp - 1) * min_udp_per_connection

                else:
                    min_udp_per_connection = total_udp_added_packets * self.MAX_BYTES_PACKET_UDP
                    max_udp_per_connection = total_udp_added_packets * self.MAX_BYTES_PACKET_UDP

            else:                          
                udp_min_bytes = self.MIN_BYTES_PACKET_UDP * total_udp_added_packets
                udp_max_bytes = self.MAX_BYTES_PACKET_UDP * total_udp_added_packets
                tcp_min_bytes = self.MIN_BYTES_PACKET_TCP * total_tcp_added_packets
                tcp_max_bytes = self.MAX_BYTES_PACKET_TCP * total_tcp_added_packets

                lower_rand_udp = max(udp_min_bytes, total_added - tcp_max_bytes)
                upper_rand_udp = min(udp_max_bytes, total_added - tcp_min_bytes)

                bytes_udp = random.randint(lower_rand_udp, upper_rand_udp + 1)
                bytes_tcp = total_added - bytes_udp

                if total_conns_udp != 1:
                    min_udp_per_connection = math.floor(bytes_udp/ total_conns_udp)
                    max_udp_per_connection = bytes_udp - (total_conns_udp - 1) * min_udp_per_connection                    

                else:
                    min_udp_per_connection = bytes_udp
                    max_udp_per_connection = bytes_udp

                if total_conns_tcp != 1:

                    min_tcp_per_connection = math.floor(bytes_tcp/ total_conns_tcp)
                    max_tcp_per_connection = bytes_tcp - (total_conns_tcp - 1) * min_tcp_per_connection

                else:
                    min_tcp_per_connection = bytes_tcp
                    max_tcp_per_connection = bytes_tcp

            minimum = min_udp_per_connection
            if min_tcp_per_connection < min_udp_per_connection:
                minimum = min_tcp_per_connection

            if minimum < new_min_f :
                new_min_f = minimum
            
            maximum = max_udp_per_connection
            if max_tcp_per_connection > max_udp_per_connection:
                maximum = max_tcp_per_connection

            if maximum > new_max_f :
                new_max_f = maximum

        #only tcp were added
        elif total_conns_tcp != 0:

            if total_conns_tcp != 1:
                min_tcp_per_connection = math.floor(total_added/ total_conns_tcp)
                max_tcp_per_connection = total_added - (total_conns_tcp - 1) * min_tcp_per_connection

            else:
                min_tcp_per_connection = total_added
                max_tcp_per_connection = total_added

            if min_tcp_per_connection < new_min_f:
                new_min_f = min_tcp_per_connection

            if max_tcp_per_connection > new_max_f:
                new_max_f = max_tcp_per_connection

        #only udp were added
        elif total_conns_udp != 0:

            if total_conns_udp != 1:
                min_udp_per_connection = math.floor(total_added/ total_conns_udp)
                max_udp_per_connection = total_added - (total_conns_udp - 1) * min_udp_per_connection
            else:
                min_udp_per_connection = total_added
                max_udp_per_connection = total_added

            if min_udp_per_connection < new_min_f:
                new_min_f = min_udp_per_connection

            if max_udp_per_connection > new_max_f:
                new_max_f = max_udp_per_connection

        return new_max_f, new_min_f

    def update_max_min_duration(self, total_added, total_conns_tcp, total_conns_udp, total_packets_tcp, total_packets_udp,new_max_f, new_min_f, raw_delta):
        #both types of connections were added
        if total_conns_tcp != 0 and total_conns_udp !=0:

            total_udp_added_packets = total_packets_udp
            total_tcp_added_packets = total_packets_tcp
            #lower bound
            if abs(raw_delta)  == total_udp_added_packets *  self.MIN_DURATION_PACKET_UDP  + total_tcp_added_packets * self.MIN_DURATION_PACKET_TCP  :
            
                if total_conns_tcp != 1:
                    min_tcp_per_connection = math.floor(total_tcp_added_packets * self.MIN_DURATION_PACKET_TCP/ total_conns_tcp)
                    max_tcp_per_connection = total_tcp_added_packets * self.MIN_DURATION_PACKET_TCP - (total_conns_tcp - 1) * min_tcp_per_connection

                else:
                    min_tcp_per_connection = total_tcp_added_packets * self.MIN_DURATION_PACKET_TCP
                    max_tcp_per_connection = total_tcp_added_packets * self.MIN_DURATION_PACKET_TCP 


                if total_conns_udp != 1:
                    min_udp_per_connection = math.floor(total_udp_added_packets * self.MIN_DURATION_PACKET_UDP/ total_conns_udp)
                    max_udp_per_connection = total_udp_added_packets * self.MIN_DURATION_PACKET_UDP - (total_conns_udp - 1) * min_udp_per_connection

                else:
                    min_udp_per_connection = total_udp_added_packets * self.MIN_DURATION_PACKET_UDP
                    max_udp_per_connection = total_udp_added_packets * self.MIN_DURATION_PACKET_UDP
            #upper bound
            elif abs(raw_delta) == total_udp_added_packets *  self.MAX_DURATION_PACKET_UDP + total_tcp_added_packets * self.MAX_DURATION_PACKET_TCP :

                if total_conns_tcp != 1:
                    min_tcp_per_connection = math.floor(total_tcp_added_packets * self.MAX_DURATION_PACKET_TCP/ total_conns_tcp)
                    max_tcp_per_connection = total_tcp_added_packets * self.MAX_DURATION_PACKET_TCP - (total_conns_tcp - 1) * min_tcp_per_connection
                else:
                    min_tcp_per_connection = total_tcp_added_packets * self.MAX_DURATION_PACKET_TCP
                    max_tcp_per_connection = total_tcp_added_packets * self.MAX_DURATION_PACKET_TCP 
                
                if total_conns_udp != 1:
                    min_udp_per_connection = math.floor(total_udp_added_packets * self.MAX_DURATION_PACKET_UDP/ total_conns_udp)
                    max_udp_per_connection = total_udp_added_packets * self.MAX_DURATION_PACKET_UDP - (total_conns_udp - 1) * min_udp_per_connection
                else:
                    min_udp_per_connection = total_udp_added_packets * self.MAX_DURATION_PACKET_UDP
                    max_udp_per_connection = total_udp_added_packets * self.MAX_DURATION_PACKET_UDP 

            else:

                udp_min_duration = self.MIN_DURATION_PACKET_UDP * total_udp_added_packets
                udp_max_duration = self.MAX_DURATION_PACKET_UDP * total_udp_added_packets
                tcp_min_duration = self.MIN_DURATION_PACKET_TCP * total_tcp_added_packets
                tcp_max_duration = self.MAX_DURATION_PACKET_TCP * total_tcp_added_packets

                lower_rand_udp = max(udp_min_duration, total_added - tcp_max_duration)
                upper_rand_udp = min(udp_max_duration, total_added - tcp_min_duration)

                duration_udp = random.uniform(lower_rand_udp, upper_rand_udp)
                duration_tcp = total_added - duration_udp

                if total_conns_udp != 1:
                    min_udp_per_connection = math.floor(duration_udp/ total_conns_udp)
                    max_udp_per_connection = duration_udp - (total_conns_udp - 1) * min_udp_per_connection                    

                else:
                    min_udp_per_connection = duration_udp
                    max_udp_per_connection = duration_udp

                if total_conns_tcp != 1:

                    min_tcp_per_connection = math.floor(duration_tcp/ total_conns_tcp)
                    max_tcp_per_connection = duration_tcp - (total_conns_tcp - 1) * min_tcp_per_connection

                else:
                    min_tcp_per_connection = duration_tcp
                    max_tcp_per_connection = duration_tcp


            minimum = min_udp_per_connection
            if min_tcp_per_connection < min_udp_per_connection:
                minimum = min_tcp_per_connection

            if minimum < new_min_f :
                new_min_f = minimum
            
            maximum = max_udp_per_connection
            if max_tcp_per_connection > max_udp_per_connection:
                maximum = max_tcp_per_connection

            if maximum > new_max_f :
                new_max_f = maximum
        #only tcp were added
        elif total_conns_tcp != 0:

            if total_conns_tcp != 1:
                min_tcp_per_connection = math.floor(total_added/ total_conns_tcp)
                max_tcp_per_connection = total_added - (total_conns_tcp - 1) * min_tcp_per_connection

            else:
                min_tcp_per_connection = total_added
                max_tcp_per_connection = total_added 

            if min_tcp_per_connection < new_min_f:
                new_min_f = min_tcp_per_connection

            if max_tcp_per_connection > new_max_f:
                new_max_f = max_tcp_per_connection

        #only udp were added
        elif total_conns_udp != 0:

            if total_conns_udp != 1:
                min_udp_per_connection = math.floor(total_added/ total_conns_udp)
                max_udp_per_connection = total_added - (total_conns_udp - 1) * min_udp_per_connection

            else:
                min_udp_per_connection = total_added
                max_udp_per_connection = total_added                 

            if min_udp_per_connection < new_min_f:
                new_min_f = min_udp_per_connection

            if max_udp_per_connection > new_max_f:
                new_max_f = max_udp_per_connection
    
        return new_max_f, new_min_f

    def run_attack(self):

        with tf.Session() as sess:         
            for i in range(1,11):
                self.attack_round(distance = i * 2 , sess = sess)

            sess.close()


aaa = Neris_attack()
aaa.run_attack()
        
