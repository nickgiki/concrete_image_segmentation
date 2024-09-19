from concrete_image_segmentation.utils3d import *

# os.chdir("../downloads")
dnm = "../downloads/SMP04F-0d0p-1600x1600x1700-16b"
fs = [f"{dnm}/{o}" for o in os.listdir(dnm)]
predict_from_path("weights/unet_concrete.hdf5", fs, "preds")
