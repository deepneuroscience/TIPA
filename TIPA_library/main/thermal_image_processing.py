# TIPA (Thermal Imaging-based Physiological and Affective computing) open-source project

## Author(s): Dr. Youngjun Cho*(Assistant Professor, UCL Computer Science)
## * http://youngjuncho.com

import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2
from packaging import version
    
''' Optimal Quantization: Optimal Quantization technique adaptively constructs a color mapping of absolute temperature to improve segmentation, classification and tracking

Reference: Cho, Y., Julier, S. J., Marquardt, N., & Bianchi-Berthouze, N. (2017). Robust tracking of respiratory rate in high-dynamic range scenes using mobile thermal imaging. Biomedical optics express, 8(10), 4480-4503.
'''
# output = optimal_quantization(mat, True)  
def optimal_quantization(t2d_data_original, print_mode = False):

    t2d_data= np.copy(t2d_data_original)
    
    vector_data= np.reshape(t2d_data, t2d_data.shape[0]*t2d_data.shape[1])
    min_T=np.percentile(vector_data, 2.5)  #2.5 percentile point
    max_T=np.percentile(vector_data, 97.5) #97.5 percentile point

    t2d_data[np.where(t2d_data > max_T)]=max_T
    t2d_data[np.where(t2d_data < min_T)]=min_T

    opt_T=min_T
    count=0
    while True:
        count+=1
        mean_back=np.mean(t2d_data[np.where(t2d_data <= opt_T)])
        mean_obj=np.mean(t2d_data[np.where(t2d_data > opt_T)])
        if np.abs(opt_T - (mean_back+mean_obj)/2 )<0.005:
            break;
        else:
            opt_T=(mean_back+mean_obj)/2

    min_T=opt_T

    if print_mode:
        print("optimal thermal range is [%f, %f]"%(min_T,max_T))

    t2d_data[np.where(t2d_data < min_T)]=min_T

    quantized_t_img=255*(t2d_data-min_T)/(max_T-min_T)
    quantized_t_img = quantized_t_img.astype(np.uint8)

    return quantized_t_img                  
                         
def optimal_quantization_range(t2d_data):
    
    vector_data= np.reshape(t2d_data, t2d_data.shape[0]*t2d_data.shape[1])
    min_T=np.percentile(vector_data, 2.5)  #2.5 percentile point
    max_T=np.percentile(vector_data, 97.5) #97.5 percentile point

    t2d_data[np.where(t2d_data > max_T)]=max_T
    t2d_data[np.where(t2d_data < min_T)]=min_T

    opt_T=min_T
    count=0
    while True:
        count+=1
        mean_back=np.mean(t2d_data[np.where(t2d_data <= opt_T)])
        mean_obj=np.mean(t2d_data[np.where(t2d_data > opt_T)])
        if np.abs(opt_T - (mean_back+mean_obj)/2 )<0.005:
            break;
        else:
            opt_T=(mean_back+mean_obj)/2

    min_T=opt_T

    return min_T,max_T     
    
'''
 This is the classical quantization method using fixed points (temperature range of interest).
'''
# output = nonoptimal_quantization(mat, T1,Tk, True)  
def nonoptimal_quantization(t2d_data, min_T, max_T, print_mode = False):
    if print_mode:
        print("your fixed thermal range of interest is [%f, %f]"%(min_T,max_T))

    quantized_t_img= np.copy(t2d_data)
    
    quantized_t_img[np.where(quantized_t_img > max_T)]=max_T
    quantized_t_img[np.where(quantized_t_img < min_T)]=min_T

    quantized_t_img=255*(quantized_t_img-min_T)/(max_T-min_T)
    quantized_t_img = quantized_t_img.astype(np.uint8)

    return quantized_t_img    



    
''' Quantization-enabled thermal image tracking

Reference [1]: Cho, Y., Julier, S. J., Marquardt, N., & Bianchi-Berthouze, N. (2017). Robust tracking of respiratory rate in high-dynamic range scenes using mobile thermal imaging. Biomedical optics express, 8(10), 4480-4503.
Reference [2]: Cho, Y., Julier, S. J., & Bianchi-Berthouze, N. (2019). Instant Stress: Detection of Perceived Mental Stress Through Smartphone Photoplethysmography and Thermal Imaging. JMIR mental health, 6(4), e10140.
Reference [3]: Cho, Y., Bianchi-Berthouze, N., Oliveira, M., Holloway, C., & Julier, S. (2019). Nose Heat: Exploring Stress-induced Nasal Thermal Variability through Mobile Thermal Imaging. In 2019 eighth International Conference on Affective Computing and Intelligent Interaction (ACII). IEEE.
'''

## eg. ROI_seq, tracked_img=thermal_tracker(data.thermal_matrix, 'MEDIANFLOW', False)     
def thermal_tracker(mat_3d, quantization_method='optimal', im_tracker_name = 'TLD', print_mode=False, nonfixed_ROI_size=True, img_viewer=True, min_T=28, max_T=38):

    # If using a version of openCV prior to 4.5.1, include all standard
    # trackers
    if version.parse(cv2.__version__) < version.parse("4.5.1"):
        trackers = {"TLD": cv2.TrackerTLD_create,
                    "MEDIANFLOW": cv2.TrackerMedianFlow_create,
                    "GOTURN": cv2.TrackerGOTURN_create,
                    "MOSSE": cv2.TrackerMOSSE_create,
                    "CSRT": cv2.TrackerCSRT_create,
                    "BOOSTING": cv2.TrackerBoosting_create,
                    "MIL": cv2.TrackerMIL_create,
                    "KCF": cv2.TrackerKCF_create}
    # If using a more recent version of OpenCV include legacy trackers
    else:
        trackers = {"TLD": cv2.legacy.TrackerTLD_create,
                    "MEDIANFLOW": cv2.legacy.TrackerMedianFlow_create,
                    "GOTURN": cv2.TrackerGOTURN_create,
                    "MOSSE": cv2.legacy.TrackerMOSSE_create,
                    "CSRT": cv2.TrackerCSRT_create,
                    "BOOSTING": cv2.legacy.TrackerBoosting_create,
                    "MIL": cv2.TrackerMIL_create,
                    "KCF": cv2.TrackerKCF_create}
    # If using a version of OpenCV newer than 4.5.2 then include the
    # DaSiamRPN tracker
    if version.parse(cv2.__version__) > version.parse("4.5.2"):
        trackers["DASIAMRPN"] = cv2.TrackerDaSiamRPN_create

    try:
        im_tracker = trackers[im_tracker_name]()
    except KeyError as E:
        print("Tracker could not be found, please check it is included in "
              "your version of OpenCV!")
        raise E
    greyscale = True
    if im_tracker_name in ["GOTURN", "BOOSTING", "DASIAMRPN"]:
        greyscale = False

    temp_mat = copy.deepcopy(mat_3d[:,:,0])    
    tracked_imgs = np.zeros(mat_3d.shape,np.uint8)
    
    if quantization_method == 'optimal':
        tracked_imgs[:,:,0] = optimal_quantization(temp_mat, print_mode)
    elif quantization_method == 'non-optimal':
        tracked_imgs[:,:,0] = nonoptimal_quantization(temp_mat, min_T, max_T, print_mode)
        
        
    # ROI selection 
    mROI = np.zeros((4, mat_3d.shape[2]), dtype=int)
    mROI[:,0] = cv2.selectROI('ROI selection', tracked_imgs[:,:,0], False)
    # mROI.shape
    cv2.destroyWindow('ROI selection')

#     cv2.namedWindow('TrackViewer')
    
    # Initialization of a tracker

    if greyscale:
        out = im_tracker.init(tracked_imgs[:,:,0], tuple(mROI[:,0]))
    else:
        stacked_img = np.stack((tracked_imgs[:, :, 0],) * 3, axis=-1)
        out = im_tracker.init(stacked_img, tuple(mROI[:, 0]))


    for i in range(1,mat_3d.shape[2]):
        if np.mod(i,20)==0:
            print("frame: %d"%i)
        temp_mat = copy.deepcopy(mat_3d[:,:,i])
        current_frame = np.zeros(tracked_imgs[:,:,0].shape)
        
        if quantization_method == 'optimal':
            current_frame = optimal_quantization(temp_mat, print_mode)
        elif quantization_method == 'non-optimal':
            current_frame = nonoptimal_quantization(temp_mat, min_T, max_T, print_mode)        
        
        
        # Updating the tracker
        if nonfixed_ROI_size:
            if greyscale:
                result, mROI[:,i] = im_tracker.update(current_frame)
            else:
                stacked_img = np.stack((current_frame,) * 3, axis=-1)
                result, mROI[:, i] = im_tracker.update(stacked_img)
        else:
            if greyscale:
                result, bbox = im_tracker.update(current_frame)
            else:
                stacked_img = np.stack((current_frame,) * 3, axis=-1)
                result, bbox = im_tracker.update(stacked_img)
                
            mROI[2,i]=mROI[2,0]
            mROI[3,i]=mROI[3,0]
            mROI[0,i]=bbox[0]+np.round(bbox[2]/2) -np.round(mROI[2,0]/2)
            mROI[1,i]=bbox[1]+np.round(bbox[3]/2) -np.round(mROI[3,0]/2)
            

        # Visualise the tracked ROI
        if result==True:        
            point1 = (int(mROI[0,i]), int(mROI[1,i]))
            point2 = (int(mROI[0,i] + mROI[2,i]), int(mROI[1,i] + mROI[3,i]))
            
            cv2.rectangle(current_frame, point1, point2, (255,255,0), 3, 0)
            tracked_imgs[:,:,i]=current_frame
        else :
            print("tracking error occurred at %d frame"%(i))
            tracked_imgs[:,:,i]=current_frame

        if img_viewer:
            cv2.imshow('TrackViewer',current_frame)
                    # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27 : break

#     cv2.namedWindow('TrackViewer')
#     count=0
#     while True:
#         cv2.imshow('TrackViewer',tracked_imgs[:,:,count])
#         if count<tracked_imgs.shape[2]-1:
#             count+=1
#         # Press ESC to exit
#         k = cv2.waitKey(1) & 0xff
#         if k == 27 : break
#     cv2.destroyAllWindows()

            
    return mROI, tracked_imgs
