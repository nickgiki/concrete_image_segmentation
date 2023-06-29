import numpy as np
import os
import re
import cv2
import imageio as io
from tqdm import tqdm

os.chdir(os.path.expanduser('~') + '/Downloads/preds')
fs = [f for f in os.listdir() if '.png' in f]
fs = [[f] + re.findall('\d{4}_\d{2}\.png',f)[0].replace('.png','').split('_') for f in fs]

slices = sorted(list(set([f[1] for f in fs])))
name = fs[0][0].replace('_' + fs[0][1] + '_' + fs[0][2],'').replace('.png','')

volume = list()

for slice in tqdm(slices,'Converting 2d images to 3d: '):
	images = sorted([f[0] for f in fs if f[1]==slice])
	volume += [
		np.concatenate([
			np.concatenate([cv2.imread(images[0],cv2.IMREAD_GRAYSCALE),cv2.imread(images[1],cv2.IMREAD_GRAYSCALE)],axis=1) ,
			np.concatenate([cv2.imread(images[2],cv2.IMREAD_GRAYSCALE),cv2.imread(images[3],cv2.IMREAD_GRAYSCALE)],axis=1) 
		], axis=0)
	
	]

volume = np.array(volume)
io.mimwrite(f"../{name}.tiff", volume)
	
	