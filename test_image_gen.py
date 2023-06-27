from concrete_image_segmentation.data import *
import matplotlib.pyplot as plt

os.chdir("../downloads/images")

data_gen_args = dict(
    rotation_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
)

train_gen = trainGenerator(
    2,
    "train",
    "images",
    "label",
    data_gen_args,
    save_to_dir=None,
    seed=int(input("set seed\n")),
)

i = 0
while i < 3:
    try:
        x, y = train_gen.__next__()

        print(x.shape)
        print(y.shape)
        image = x[0, :, :, 0]
        mask = y[0, :, :, 0]
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap="gray")
        plt.show()
        i += 1
    except Exception as e:
        print(e)
        break
