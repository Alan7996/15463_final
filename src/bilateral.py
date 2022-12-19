import math
import numpy as np
from scipy.interpolate import interpn
from skimage import io
import cv2
import matplotlib.pyplot as plt
import os

dirname = os.path.dirname(__file__)

def bilateral(im1, im2, sigma_s, sigma_r):
    # bilateral filtering implementation as per Petschnigg's specifications
    
    return


def perform_all(save_dir, type, im_am, im_fl, sigma_s, sigma_r):
    ### call bilateral function to perform joint bilateral filtering step by step
    
    
    # Piecewise
    
    # Joint
    
    # Detail Transfer
    
    # Shadow and specularity masking

    # Combined
    
    return
    

sigma_s = 2
sigma_r = 0.1

### Original flash / no-flash ###
filename = os.path.join(dirname, '../data/lamp/lamp_')
save_dir = os.path.join(dirname, '../data/lamp/results/lamp_')

im_am = cv2.imread(filename + "ambient.tif", cv2.IMREAD_UNCHANGED)
im_fl = cv2.imread(filename + "flash.tif", cv2.IMREAD_UNCHANGED)

im_am = im_am[:,:,::-1] / 255
im_fl = im_fl[:,:,::-1] / 255

# perform_all(save_dir, 'fnf', im_am, im_fl, sigma_s, sigma_r)


### With dark flash ###
filename = os.path.join(dirname, '../data/bowls1/bowls1_')
save_dir = os.path.join(dirname, '../data/bowls1/results/bowls1_')

im_am = cv2.imread(filename + "low_amb.tiff", cv2.IMREAD_UNCHANGED)
im_fl = cv2.imread(filename + "low_uvir.tiff", cv2.IMREAD_UNCHANGED)

im_am = im_am[:,:,::-1] / 255
im_fl = im_fl[:,:,::-1] / 255

# perform_all(save_dir, 'dark', im_am, im_fl, sigma_s, sigma_r)


# Collection of dolls
filename = os.path.join(dirname, '../data/dolls1/dolls1_')
save_dir = os.path.join(dirname, '../data/dolls1/results/dolls1_')

im_am = cv2.imread(filename + "low_amb.tiff", cv2.IMREAD_UNCHANGED)
im_fl = cv2.imread(filename + "low_uvir.tiff", cv2.IMREAD_UNCHANGED)

im_am = im_am[:,:,::-1] / 255
im_fl = im_fl[:,:,::-1] / 255
# perform_all(save_dir, 'dark', im_am, im_fl, sigma_s, sigma_r)

# Person's face
filename = os.path.join(dirname, '../data/person1/person1_')
save_dir = os.path.join(dirname, '../data/person1/results/person1_')

im_am = cv2.imread(filename + "low_amb.tiff", cv2.IMREAD_UNCHANGED)
im_fl = cv2.imread(filename + "low_uvir.tiff", cv2.IMREAD_UNCHANGED)

im_am = im_am[:,:,::-1] / 255
im_fl = im_fl[:,:,::-1] / 255
# perform_all(save_dir, 'dark', im_am, im_fl, sigma_s, sigma_r)