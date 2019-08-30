# TIPA (Thermal Imaging-based Physiological and Affective computing) open-source project

## Author(s): Dr. Youngjun Cho*(Assistant Professor, UCL Computer Science)
## * http://youngjuncho.com


# from ipywidgets import interact, interactive, fixed, interact_manual
# import ipywidgets as widgets
import matplotlib.pyplot as plt

## for python jupyter users, type %matplotlib notebook before calling this.
# Thermal image show.
def timshow(matrix):
    fig= plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.imshow(matrix, cmap=plt.get_cmap('hot'))
    plt.colorbar()

    
