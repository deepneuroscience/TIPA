## Contributors
## Dr. Youngjun Cho (Assistant Professor, UCL Computer Science), Chang Liu (MSc in Machine Learning, UCL Computer Science)

import numpy as np

''' 
Signal processing toolkits for Respiration Variability Spectrogram

[Reference] 
Cho, Y., Bianchi-Berthouze, N. and Julier, S.J., 2017. 
DeepBreath: Deep learning of breathing patterns for automatic stress recognition using low-cost thermal imaging in unconstrained settings. In 2017 Seventh International Conference on Affective Computing and Intelligent Interaction (ACII) (pp. 456-463). IEEE. https://doi.org/10.1109/ACII.2017.8273639
'''

        
def gausswin(L, alpha=2.5):
    N = L - 1
    n = np.arange(0,N+1)-N/2
    w = np.exp(-(1/2)*(alpha*n /(N /2))**2)
    return w
