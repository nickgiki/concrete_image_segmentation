"""This module provides utility functions for 3D image processing"""
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2



def get_dimensions(filename):
    """Returns the dimensions of the raw picture from the filename"""
    return [int(x) for x in re.sub(f'(.+)-(\d+x\d+x\d+)(.+)','\\2',filename).split('x')]

def read_raw(filepath):
    """Reads and returns the image 3D array"""
    with open(filepath, 'rb') as f:
      img_str = f.read()
    arr = (2**16-1)- np.frombuffer(img_str, np.uint16)
    h,w,d = get_dimensions(filepath)
    return np.reshape(arr,(d,h,w))

def crop_image(image,px=40):
    """Crops an image""" 
    return image[px:(image.shape[0]-px),px:(image.shape[1]-px)]

def plot_image(image):
    """Plots an image"""
    plt.imshow(image, cmap=plt.cm.gray_r)
    plt.axis('off')
    return plt.show()

def plot_slice(im_array, i):
    """Plots a slice of a 3d image given its z index"""
    im_ = im_array[i,:,:]
    return plot_image(im_)

def quadrants(image, overlap=0.05):
    """Splits image into 4 parts using an overlap"""
    h,w = image.shape
    midy,midx = h//2,w//2
    oly = int(h*overlap)
    olx = int(w*overlap)
    return image[0:midy+oly,0:midx+olx],image[0:midy+oly,midx-olx:],image[midy-oly:,0:midx+olx],image[midy-oly:,midx-olx:]


def nineths(image, overlap=0.05):
    """Splits image into 6 parts using an overlap"""
    h,w = image.shape
    oly = int(h*overlap)
    olx = int(w*overlap)
    dy, dx = int(np.round(h/3)) + oly ,int(np.round(w/3)) + olx
    x1,y1 = dx, dy
    return (
        image[:dy,:x1],image[:dy,x1:x1+dx],image[:dy,-dx:],
        image[y1:y1+dy,:x1],image[y1:y1+dy,x1:x1+dx],image[y1:y1+dy,-dx:],
        image[-dy:,:x1],image[-dy:,x1:x1+dx],image[-dy:,-dx:],
    )
    
def cut_and_save(filepath, output_size = (512,512),crop=True,cut='quad', eight_bit=True):
    """Gets a raw 3D image file path and saves to new dir"""
    im_ar = read_raw(filepath)
    
    dir_name = filepath.replace('.raw','')
    os.makedirs(dir_name,exist_ok=True)
    
    #loop through 3d scan
    for i in range(im_ar.shape[0]):
        slice_ = im_ar[i,:,:]
        if crop:
            slice_ = crop_image(slice_.copy())
        
        if cut == 'quad':
            qd_dict = dict(zip(['00','01','10','11'],quadrants(slice_)))
        elif cut == 'nine':
            qd_dict = dict(zip(['00','01','02','10','11','12','20','21','22'],nineths(slice_)))
        else:
            raise ValueError(f'{cut} is not valid for cut. Must be one of ["quad","nine"].')
        
        for nm,im in qd_dict.items():
            im_ = (im/(2**16-1)).astype(float)
            if output_size:
                im_ = cv2.resize(im_,output_size,interpolation = cv2.INTER_AREA)
            im_ = ((1-im_)*(2**(8 if eight_bit else 16)-1)).astype(np.uint16 if not eight_bit else int)
            fname = f"{dir_name}/{str(i).rjust(4,'0')}_{nm}.png"
            cv2.imwrite(fname,im_)