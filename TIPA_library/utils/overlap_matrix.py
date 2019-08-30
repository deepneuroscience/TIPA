## Contributors
## Dr. Youngjun Cho (Assistant Professor, UCL Computer Science), Chang Liu (MSc in Machine Learning, UCL Computer Science)

import numpy as np

''' 
Signal processing toolkits for Respiration Variability Spectrogram

[Reference] 
Cho, Y., Bianchi-Berthouze, N. and Julier, S.J., 2017. 
DeepBreath: Deep learning of breathing patterns for automatic stress recognition using low-cost thermal imaging in unconstrained settings. In 2017 Seventh International Conference on Affective Computing and Intelligent Interaction (ACII) (pp. 456-463). IEEE. https://doi.org/10.1109/ACII.2017.8273639
'''

def overlap_matrix(data,overlap_rate1,window_size):
    matrix_list = []
    start = 0
    end = window_size
    remain = (np.shape(data)[1])
    while remain>=window_size:
        matrix_list.append(data[:,int(round(start+0.01)):int(round(end+0.01))])
        start += overlap_rate1*window_size#start and end overlap 1
        end += overlap_rate1*window_size
        remain -= overlap_rate1*window_size 
    return matrix_list
     