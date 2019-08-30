# TIPA (Thermal Imaging-based Physiological and Affective computing) open-source project

## Author(s): Dr. Youngjun Cho*(Assistant Professor, UCL Computer Science)
## * http://youngjuncho.com

import numpy as np
import cv2

''' 
Simple Dynamic-Range Quantisation.

[Reference] 
Cho, Y., Bianchi-Berthouze, N., Marquardt, N. and Julier, S.J., 2018. Deep Thermal Imaging: Proximate Material Type Recognition in the Wild through Deep Learning of Spatial Surface Temperature Patterns. In Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems, ACM. https://doi.org/10.1145/3173574.3173576
'''
def simpleDRQ(t2d_data, desired_height, desired_width):
    
    (height, width) = t2d_data.shape

    x_start = int((height - width * 0.925) / 2 + 1)
    x_end = int((height + width * 0.925) / 2)
    y_start = int((width - width * 0.925) / 2 + 1)
    y_end = int((width + width * 0.925) / 2)
    t2d_data = t2d_data[x_start:x_end, y_start:y_end]
    
    [adjusted_h, adjusted_w] = t2d_data.shape

    data = np.zeros((leng, 60, 60))


    temp_A = t2d_data
    mmin = temp_A.min()
    mmax = temp_A.max()
    for cho1 in range(0, adjusted_w):
        for cho2 in range(0, adjusted_h):
            temp_A[cho2, cho1] = (temp_A[cho2, cho1] - mmin) / (mmax - mmin)

    return cv2.resize(temp_A, (desired_height, desired_width))
