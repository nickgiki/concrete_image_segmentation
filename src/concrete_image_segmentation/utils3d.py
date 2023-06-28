"""This module provides utility functions for 3D image processing"""
import re
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from itertools import product
from shutil import move, make_archive
from tqdm import tqdm
import skimage.io as io


def extract_zip(zip_file_path):
    from zipfile import ZipFile

    with ZipFile(zip_file_path, "r") as zip:
        # printing all the contents of the zip file
        zip.printdir()
        print("Extracting all the files now...")
        zip.extractall()
        print("Done!")


def make_zip(dir_names, master_dir_name, zip_name):
    for dir_ in dir_names:
        move(dir_, f"{master_dir_name}/{dir_}")
    make_archive(zip_name, "zip", master_dir_name)


def get_dimensions(filename):
    """Returns the dimensions of the raw picture from the filename"""
    return [
        int(x) for x in re.sub(f"(.+)-(\d+x\d+x\d+)(.+)", "\\2", filename).split("x")
    ]


def read_raw(filepath):
    """Reads and returns the image 3D array"""
    with open(filepath, "rb") as f:
        img_str = f.read()
    arr = (2**16 - 1) - np.frombuffer(img_str, np.uint16)
    h, w, d = get_dimensions(filepath)
    return np.reshape(arr, (d, h, w))


def convert_to_float(image, bits=16):
    if image.max() > 1:
        img = (image / (2**bits - 1)).astype(np.float32)
        return img
    return image


def crop_image(image, px=40):
    """Crops an image"""
    if len(image.shape) > 2:
        return image[:, px : (image.shape[0] - px), px : (image.shape[1] - px)]
    return image[px : (image.shape[0] - px), px : (image.shape[1] - px)]


def overlay_mask(image, alpha, mask, beta, gamma=0, mask_color_rgb=(1, 0, 0)):
    # blend the images
    img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    msk = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    mask_color_rgb = 1 - np.array(mask_color_rgb)
    img = np.moveaxis(img, -1, 0)
    msk = np.moveaxis(msk, -1, 0)
    msk = np.array([msk[j, :, :] * mask_color_rgb[j] for j in range(3)])
    img_out = cv2.addWeighted(img, alpha, msk, beta, gamma)
    img_out = np.moveaxis(img_out, 0, -1)
    return img_out


def plot_image(image, to_float=False):
    """Plots an image"""

    if len(image.shape) > 2:
        plt.imshow(1 - convert_to_float(image) if to_float else image)
    else:
        plt.imshow(
            cv2.cvtColor(
                1 - convert_to_float(image) if to_float else image, cv2.COLOR_GRAY2RGB
            )
        )
    plt.axis("off")
    plt.show()


def plot_slice(im_array, i, to_float=False):
    """Plots a slice of a 3d image given its z index"""
    im_ = im_array[i, :, :]
    return plot_image(im_, to_float=to_float)


def quadrants(image, overlap=0.05):
    """Splits image into 4 parts using an overlap"""
    h, w = image.shape
    midy, midx = h // 2, w // 2
    oly = int(h * overlap)
    olx = int(w * overlap)
    return (
        image[0 : midy + oly, 0 : midx + olx],
        image[0 : midy + oly, midx - olx :],
        image[midy - oly :, 0 : midx + olx],
        image[midy - oly :, midx - olx :],
    )


def nineths(image, overlap=0.05):
    """Splits image into 6 parts using an overlap"""
    h, w = image.shape
    oly = int(h * overlap)
    olx = int(w * overlap)
    dy, dx = int(np.round(h / 3)) + oly, int(np.round(w / 3)) + olx
    x1, y1 = dx, dy
    return (
        image[:dy, :x1],
        image[:dy, x1 : x1 + dx],
        image[:dy, -dx:],
        image[y1 : y1 + dy, :x1],
        image[y1 : y1 + dy, x1 : x1 + dx],
        image[y1 : y1 + dy, -dx:],
        image[-dy:, :x1],
        image[-dy:, x1 : x1 + dx],
        image[-dy:, -dx:],
    )


def cut_and_save(
    filepath, output_size=(512, 512), crop=0, overlap=0.05, cut="quad", eight_bit=True
):
    """Gets a raw 3D image file path and saves to new dir"""
    im_ar = read_raw(filepath)

    dir_name = filepath.replace(".raw", "")
    os.makedirs(dir_name, exist_ok=True)

    # loop through 3d scan
    for i in range(im_ar.shape[0]):
        slice_ = im_ar[i, :, :]
        if crop:
            slice_ = crop_image(slice_.copy(), px=crop)

        if cut == "quad":
            qd_dict = dict(
                zip(["00", "01", "10", "11"], quadrants(slice_, overlap=overlap))
            )
        elif cut == "nine":
            qd_dict = dict(
                zip(
                    ["00", "01", "02", "10", "11", "12", "20", "21", "22"],
                    nineths(slice_, overlap=overlap),
                )
            )
        else:
            raise ValueError(
                f'{cut} is not valid for cut. Must be one of ["quad","nine"].'
            )

        for nm, im in tqdm(qd_dict.items()):
            im_ = convert_to_float(im)
            if output_size:
                im_ = cv2.resize(im_, output_size, interpolation=cv2.INTER_AREA)
            im_ = ((1 - im_) * (2 ** (8 if eight_bit else 16) - 1)).astype(
                np.uint16 if not eight_bit else int
            )
            fname = f"{dir_name}/{dir_name}_{str(i).rjust(4,'0')}_{nm}.png"
            cv2.imwrite(fname, im_)


def train_test_split(
    folder_path,
    train_p=0.75,
    drop_p=0.05,
    test_p=0.2,
    mask_kwd="Mask",
):
    """
    Gets a folder path with names generated from
    cut_and_save and splits to train, drop and test
    """
    cwd = os.getcwd()
    try:
        assert train_p + drop_p + test_p == 1, "percentages must add up to 1"
        os.chdir(folder_path)

        for s1, s2 in product(["train", "test", "drop"], ["images", "label"]):
            os.makedirs(f"{cwd}/{s1}", exist_ok=1)
            os.makedirs(f"{cwd}/{s1}/{s2}", exist_ok=1)

        indices = [
            (f, int(re.findall(r"\d{4}_\d{2}.png", f)[0].split("_")[0]))
            for f in os.listdir()
            if f.endswith(".png")
        ]

        n_ind = len(set([i for f, i in indices]))

        train_cutoff, test_cutoff = round(train_p * n_ind), round(
            (train_p + drop_p) * n_ind
        )

        for f, i in indices:
            if i < train_cutoff:
                move(f, f"{cwd}/train/{'label' if mask_kwd in f else 'images'}")
            elif i < test_cutoff:
                move(f, f"{cwd}/drop/{'label' if mask_kwd in f else 'images'}")
            else:
                move(f, f"{cwd}/test/{'label' if mask_kwd in f else 'images'}")
        for d in ["train", "drop", "test"]:
            for f in os.listdir(f"{cwd}/{d}/label"):
                if f.endswith(".png"):
                    os.rename(
                        f"{cwd}/{d}/label/{f}",
                        f"{cwd}/{d}/label/{f}".replace("-Mask", ""),
                    )
                    assert f.replace("-Mask", "") in os.listdir(f"{cwd}/{d}/images")
    except Exception as e:
        print(f"An error has occured:\n- {e}")
    finally:
        os.chdir(cwd)


def shuffle_names(dir_name, seed=1):
    cwd = os.getcwd()
    try:
        os.chdir(dir_name)
        fnames = [
            f"{d}/{sd}/{o}"
            for d in ["test", "train", "drop"]
            for sd in ["label"]
            for o in os.listdir(f"{d}/{sd}")
            if o.endswith(".png")
        ]
        print("found ", len(fnames), " pngs")
        dict_name = list(
            zip(
                [
                    str(x).rjust(5, "0") + ".png"
                    for x in np.random.permutation(len(fnames))
                ],
                fnames,
            )
        )
        for new, old in dict_name:
            os.rename(old, "/".join(old.split("/")[:-1] + [new]))
            old2 = old.replace("label", "images")
            os.rename(old2, "/".join(old2.split("/")[:-1] + [new]))
        with open("rename_dict.json", "w+") as f:
            json.dump(dict_name, f, indent=6)
    except Exception as e:
        print(e)
    finally:
        os.chdir(cwd)


def image_preproc(x):
    x_ = x.copy()
    x_ /= 255
    x_ = np.reshape(x_, x_.shape + (1,))
    x_ = np.reshape(x_, (1,) + x_.shape)
    return x_


def predict_mod(model, x):
    x = image_preproc(x)
    y = model.predict(x)
    y[y > thresh] = 1
    y[y <= thresh] = 0
    return y


def predict_from_path(model_path, x_paths, save_dir=None, thresh=0.5):
    assert os.path.isfile(model_path), "Model not found"
    model = tf.keras.model.load_model(model_path)

    if isinstance(x_paths, list):
        y_h = []
        for xp in xpaths:
            try:
                x = io.imread(xp)
                y = predict_mod(model, x)
                if save_dir:
                    io.imsave(f"{save_dir}/{xp}", y)
                else:
                    y_h += [y]
            except Exception as e:
                print(f"Could not process {xp}")
        return y_h
    elif os.path.isfile(x_paths) and xpaths.endswith(".png"):
        x = io.imread(x_paths)
        y = predict_mod(model, x)
        if save_dir:
            io.imsave(f"{save_dir}/{x_paths}", y)
        else:
            return y
    else:
        raise TypeError(f"{x_paths} not a list of file paths or a file path")
