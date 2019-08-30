## Contributors
## Dr. Youngjun Cho (Assistant Professor, UCL Computer Science), Chang Liu (MSc in Machine Learning, UCL Computer Science)
### currently, this codes needs to be debugged.


import numpy as np
from TIPA_library.utils import overlap_windows, overlap_matrix, gausswin, compute_frequency_grid
from scipy import signal

''' 
Signal processing toolkits for Respiration Variability Spectrogram

[Reference] 
Cho, Y., Bianchi-Berthouze, N. and Julier, S.J., 2017. 
DeepBreath: Deep learning of breathing patterns for automatic stress recognition using low-cost thermal imaging in unconstrained settings. In 2017 Seventh International Conference on Affective Computing and Intelligent Interaction (ACII) (pp. 456-463). IEEE. https://doi.org/10.1109/ACII.2017.8273639
'''
# This code does have an issue with the limited resolution of outputs from scipy.signal.periodogram. (needs to improve)

def rvs(Fss,x0):#t0:
    Tmax = 20
    step_length = Fss*1
    lag = Tmax*Fss
    window_list = overlap_windows.overlap_windows(x0,0.05,lag)
    #time_list = overlap_windows(t0,0.05,lag)
    s_PSD_1 = []
    for i in range(len(window_list)):
        data_use = window_list[i]
        data_info = data_use.copy()
        #time_info = time_list[i]
        max_data = max(data_info)
        min_data = min(data_info)
        for j in range(len(data_info)):
            data_info[j] = (data_info[j]-min_data)/(max_data-min_data)
        
        filterN = 3
        Wn1=0.1
        Wn2=0.85
        Fn=Fss/2
        filter_b, filter_a = scipy.signal.ellip(filterN,1,2,[Wn1/Fn,Wn2/Fn],btype='bandpass')
        filtered_featurescaled_data = scipy.signal.lfilter(filter_b,filter_a,data_info,axis=0)
        w = gausswin.gausswin(5120)
        gaussian_final_window = np.array(filtered_featurescaled_data*(w.T),dtype='float64')
#######
        freq_index,freq_amplitude = scipy.signal.periodogram(gaussian_final_window,Fss)
#         freqs = compute_frequency_grid(oversampling=10)
#         ang_freqs = 2 * np.pi * freqs
#         #t = list(t)
#         freq_amplitude = scipy.signal.lombscargle(time_info,gaussian_final_window,ang_freqs)
        
        #STACK PSD IN SLIDING WNNDOW
        s_PSD_1.append(freq_amplitude)
    s_PSD_1 = np.array(s_PSD_1).T
#     ss_PSD_1=s_PSD_1[1:19,:]
    #return ss_PSD_1
#     seperate_data = overlap_matrix.overlap_matrix(ss_PSD_1,1/120,120)
    return s_PSD_1