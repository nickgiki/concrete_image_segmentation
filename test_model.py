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


def plot_image_mask(x, y, title=""):
    image = x[0, :, :, 0]
    mask = y[0, :, :, 0]
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title(title)
    plt.show()


def plot_diff(y, y_h, title=""):
    plt.imshow((y_h == y_).astype(int)[0, :, :, 0], cmap="gray")
    plt.title(title)
    plt.show()


model = tf.keras.models.load_model("../downloads/weights/unet_concrete.hdf5")

i = 0
trials = int(input("How many examples?\n- "))
while i < trials:
    # get random file from test model
    file_ = np.random.choice(
        os.listdir(
            os.path.join(
                os.path.expanduser("~"), "Downloads", "images", "test", "images"
            )
        )
    )
    y_path = os.path.join(
        os.path.expanduser("~"), "Downloads", "images", "test", "label", file_
    )
    x_path = y_path.replace("label", "images")

    assert os.path.isfile(y_path) and os.path.isfile(
        x_path
    ), f"Image or mask not found in\n- {x_path}\n- {y_path}"

    # read files
    x_ = io.imread(x_path, as_gray=True) / 255
    y_ = io.imread(y_path, as_gray=True) / 255

    # reshape
    x_ = np.reshape(x_, x_.shape + (1,))
    y_ = np.reshape(y_, y_.shape + (1,))
    x_ = np.reshape(x_, (1,) + x_.shape)
    y_ = np.reshape(y_, (1,) + y_.shape)

    # pass through model
    y_h = model.predict(x_)
    y_h[y_h > 0.5] = 1
    y_h[y_h <= 0.5] = 0
    y_h = y_h.astype(int)

    # plot
    overlay_mask(x_, y_, "Actual - " + file_)
    overlay_mask(x_, y_h, "Prediction - " + file_)
    plot_diff(y_, y_h, "Difference - " + file_)
    i += 1
