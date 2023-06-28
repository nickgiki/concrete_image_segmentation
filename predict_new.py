import tensorflow as tf
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import os


def overlay_mask(x, y, title=""):
    plt.imshow(x[0, :, :, 0], cmap="gray")
    plt.imshow(y[0, :, :, 0], cmap="jet", alpha=0.2)
    plt.title(title)
    plt.show()


model = tf.keras.models.load_model("../downloads/weights/unet_concrete.hdf5")


trials = int(input("How many examples?\n- "))
pics = np.random.choice(
    os.listdir(
        os.path.join(
            os.path.expanduser("~"), "Downloads", "SMP04F-0d0p-1600x1600x1700-16b"
        )
    ),
    size=trials,
).tolist()

while pics:
    # pick random
    file_ = pics.pop()
    x_path = os.path.join(
        os.path.expanduser("~"), "Downloads", "SMP04F-0d0p-1600x1600x1700-16b", file_
    )

    # read
    x_ = io.imread(x_path, as_gray=True) / 255

    # reshape
    x_ = np.reshape(x_, x_.shape + (1,))
    x_ = np.reshape(x_, (1,) + x_.shape)

    # pass through model
    y_h = model.predict(x_)
    y_h[y_h > 0.5] = 1
    y_h[y_h <= 0.5] = 0
    y_h = y_h.astype(int)
    overlay_mask(x_, y_h, "Prediction - " + file_)
