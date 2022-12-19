import math
import numpy as np
from scipy.optimize import minimize
from skimage import io
import cv2
import os
from irls import IRLS

dirname = os.path.dirname(__file__)

def gradient(I):
    ### compute x,y gradients of the input image
    
    return

def cost_func(mp, Rjp, Rjp_p, Ajp, nab_F1, nab_F3, mj, k, a):
    likelihood = mj * mp * ((Rjp - Ajp) ** 2)
    spatial = k * mp * ((abs(Rjp - Rjp_p)) ** a) 
    ir_spectral = (abs(Rjp - Rjp_p - nab_F1)) ** a
    uv_spectral = (abs(Rjp - Rjp_p - nab_F3)) ** a
    
    return likelihood + spatial + ir_spectral + uv_spectral

def dark_flash(im_am, im_ir, im_uv, mask, k=1, a=0.7):
    ### NOTE: Failed attempts have been deleted to the most simplest form
    ### significant reconstructions may be needed to make this process functional
    
    R = np.empty(im_am.shape)
    
    mu_j = 0
    
    # for each pixel of each RGB channel, solve the minimization problem
    for j in range(3): #rgb
        if (j < 2):
            mu_j = 5
        else:
            mu_j = 10
        
        A_j = im_am[:,:,j]
        nab_F1 = gradient(im_ir[:,:,j])
        nab_F3 = gradient(im_uv[:,:,j])
        
        for y in range(A_j.shape[0]):
            for x in range(A_j.shape[1]):
                irls_y = np.array(nab_F1[y, x], nab_F3[y, x], 1)
                irls_X = np.empty((1))
                
                # need to compute irls_X by rearranging the cost function
                # could also try using scipy.optimize.minimize
                
                B = IRLS(y=irls_y, X=irls_X, maxiter=20)
                
                # may need to use incomplete Cholesky preconditioner to speed up convergence
                
                R[y,x] = B
        
    return R


def perform_all(save_dir, type, im_am, im_ir, im_uv, kappa, alpha):
    
    ### shadow and specularity mask generation redacted
    mask = np.empty()
    
    dark_rec = dark_flash(im_am, im_ir, im_uv, mask, kappa, alpha)

    combined_rec = (1 - mask) * dark_rec + mask * im_am
    
    # io.imsave(save_dir + type + "-result.png", combined_rec)
    

kappa = 1
alpha = 0.7

### Original flash / no-flash ###
filename = os.path.join(dirname, '../data/lamp/lamp_')
save_dir = os.path.join(dirname, '../data/lamp/results/lamp_')

im_am = cv2.imread(filename + "ambient.tif", cv2.IMREAD_UNCHANGED)
im_fl = cv2.imread(filename + "flash.tif", cv2.IMREAD_UNCHANGED)

im_am = im_am[:,:,::-1] / 255
im_fl = im_fl[:,:,::-1] / 255

# perform_all(save_dir, 'fnf', im_am, im_fl, im_fl, kappa, alpha)


### With dark flash ###
filename = os.path.join(dirname, '../data/bowls1/bowls1_')
save_dir = os.path.join(dirname, '../data/bowls1/results/bowls1_')

im_am = cv2.imread(filename + "low_amb.tiff", cv2.IMREAD_UNCHANGED)
im_fl = cv2.imread(filename + "low_uvir.tiff", cv2.IMREAD_UNCHANGED)

im_am = im_am[:,:,::-1] / 255
im_fl = im_fl[:,:,::-1] / 255

# perform_all(save_dir, 'dark', im_am, im_fl, im_fl, kappa, alpha)

# Collection of dolls
filename = os.path.join(dirname, '../data/dolls1/dolls1_')
save_dir = os.path.join(dirname, '../data/dolls1/results/dolls1_')

im_am = cv2.imread(filename + "low_amb.tiff", cv2.IMREAD_UNCHANGED)
im_fl = cv2.imread(filename + "low_uvir.tiff", cv2.IMREAD_UNCHANGED)

im_am = im_am[:,:,::-1] / 255
im_fl = im_fl[:,:,::-1] / 255
# perform_all(save_dir, 'dark', im_am, im_fl, im_fl, kappa, alpha)

# Person's face
filename = os.path.join(dirname, '../data/person1/person1_')
save_dir = os.path.join(dirname, '../data/person1/results/person1_')

im_am = cv2.imread(filename + "low_amb.tiff", cv2.IMREAD_UNCHANGED)
im_fl = cv2.imread(filename + "low_uvir.tiff", cv2.IMREAD_UNCHANGED)

im_am = im_am[:,:,::-1] / 255
im_fl = im_fl[:,:,::-1] / 255
# perform_all(save_dir, 'dark', im_am, im_fl, im_fl, kappa, alpha)