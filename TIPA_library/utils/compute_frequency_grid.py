## Contributors
## Dr. Youngjun Cho (Assistant Professor, UCL Computer Science), Chang Liu (MSc in Machine Learning, UCL Computer Science)

import numpy as np

''' 
Signal processing toolkits for Respiration Variability Spectrogram

[Reference] 
Cho, Y., Bianchi-Berthouze, N. and Julier, S.J., 2017. 
DeepBreath: Deep learning of breathing patterns for automatic stress recognition using low-cost thermal imaging in unconstrained settings. In 2017 Seventh International Conference on Affective Computing and Intelligent Interaction (ACII) (pp. 456-463). IEEE. https://doi.org/10.1109/ACII.2017.8273639
'''

def compute_frequency_grid(oversampling=50):
    T=20 #T = t.max() - t.min()#20
    N=5120#N = len(t)#5120
    ts = T/(N-1)
    df = 1 / (oversampling*ts*N)
    fmax = 1/ (2 * ts)
    return np.arange(df,fmax,df)
