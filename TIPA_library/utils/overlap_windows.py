## Contributors
## Dr. Youngjun Cho (Assistant Professor, UCL Computer Science), Chang Liu (MSc in Machine Learning, UCL Computer Science)


''' 
Signal processing toolkits for Respiration Variability Spectrogram

[Reference] 
Cho, Y., Bianchi-Berthouze, N. and Julier, S.J., 2017. 
DeepBreath: Deep learning of breathing patterns for automatic stress recognition using low-cost thermal imaging in unconstrained settings. In 2017 Seventh International Conference on Affective Computing and Intelligent Interaction (ACII) (pp. 456-463). IEEE. https://doi.org/10.1109/ACII.2017.8273639
'''

def overlap_windows(data,overlap_rate,window_size):
    window_list = []
    start = 0
    end = window_size
    remain_length = len(data)
    
    while remain_length>=window_size:
        window_list.append(data[int(round(start+0.01)):int(round(end+0.01))])
        start += overlap_rate*window_size#start and end overlap 1
        end += overlap_rate*window_size
        remain_length -= overlap_rate*window_size 
    return window_list
