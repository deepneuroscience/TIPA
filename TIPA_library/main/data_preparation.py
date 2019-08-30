# TIPA (Thermal Imaging-based Physiological and Affective computing) open-source project

## Author(s): Dr. Youngjun Cho*(Assistant Professor, UCL Computer Science)
## * http://youngjuncho.com

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets
from TIPA_library.main.thermal_image_processing import optimal_quantization_range as oq_range
    
class data_preparation_TIPA_protocol:
    def __init__(self, file_name, known_height, known_width):
        self.frame_height = known_height
        self.frame_width = known_width
        self.file = np.fromfile(file_name, dtype='uint16')
        self.file = np.reshape(self.file, [-1, self.frame_height*self.frame_width + 4])#
        self.file = (np.transpose(self.file).astype('uint16'))
        
        self.tracked_matrix=[]
        self.thermal_matrix = self.get_thermal_matrix(self.file)
        self.time_stamp = self.get_time_stamp(self.file)
        self.time_stamp[0]=0
        
    def get_thermal_matrix(self, file):
        # assigning pixels to image-frame matrix 'img_frames'
        num_frames = np.size(file, 1)
        file_segment=file[0:self.frame_height*self.frame_width, 0:num_frames]; 
        img_frames = np.reshape(file_segment[0:self.frame_height*self.frame_width, :], [self.frame_height, self.frame_width, num_frames])
        #img_frames = np.reshape(file[0:320*240, :], [frame_height, frame_width, num_frames])
        img_frames = (img_frames - 27315) / 100 # celcius
        return img_frames


    def get_time_stamp(self, file):
        # getting time array from the last two rows of read-in file
        pre_time_arr = file[self.frame_height*self.frame_width+2:, :]
        pre_time_arr_high = (np.round(pre_time_arr[:, :]) & 255) << 8
        pre_time_arr_low = np.round(pre_time_arr[:, :]) >> 8
        time_stamp = (pre_time_arr_high + pre_time_arr_low).astype('uint32')
        new_time_stamp = ((np.round(time_stamp[0, :]) << 16) + time_stamp[1, :]) / 1000
        return new_time_stamp


    # To show thermal 2D sequences interactively
    def interactive_imshow_cond(self, frame_number):
        plt.figure(2)
        plt.imshow(self.thermal_matrix[:,:,frame_number], cmap=plt.get_cmap('hot'))
        plt.colorbar()
    #     plt.show()
    
    def interactive_imshow_cond2(self, frame_number, thermal_range):
        plt.figure(2)
        plt.imshow(self.thermal_matrix[:,:,frame_number], cmap=plt.get_cmap('hot'), vmin=thermal_range[0], vmax=thermal_range[1])
        plt.colorbar()
    #     plt.show()
    
    def interactive_imshow_cond3(self, frame_number):
        plt.figure(2)
        min_T, max_T=oq_range(self.thermal_matrix[:,:,frame_number])
        plt.imshow(self.thermal_matrix[:,:,frame_number], cmap=plt.get_cmap('hot'), vmin=min_T, vmax=max_T)
        plt.colorbar()
    #     plt.show()
    
    def interactive_imshow_cond4(self, frame_number):
        plt.figure(2)
        plt.imshow(self.tracked_matrix[:,:,frame_number], cmap=plt.get_cmap('hot'))
        plt.colorbar()
    #     plt.show()    
    
class data_preparation_raw_matrix:
    def __init__(self, matrix, framerate):
        self.frame_height = matrix.shape[0]
        self.frame_width = matrix.shape[1]
        self.frame_length = matrix.shape[2]
        self.thermal_matrix = matrix
        self.time_stamp = np.arange(0,self.frame_length/framerate,1/framerate)