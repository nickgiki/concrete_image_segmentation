import numpy as np
import skimage.io as io
import os
import re
from tqdm import tqdm
from skimage import img_as_uint
from skimage.transform import resize


def stitch_images(
    folder, output_name="stitched", overlap_pct=0.2, original_cropped_size=1400
):
    cwd = os.getcwd()
    try:
        os.chdir(folder)
        os.makedirs(output_name, exist_ok=True)

        image_dict2 = {}
        for f in os.listdir():
            if not os.path.isfile(f):
                continue
            nm = re.sub("(.+_)(\d{4})(_\d{2}.+)", "\\2", f)
            if nm in image_dict2.keys():
                image_dict2[nm] += [f]
            else:
                image_dict2[nm] = [f]

        print(f"found {len(image_dict2)} images")

        for im in tqdm(list(image_dict2.keys()), "Stiching: "):
            imgs = [io.imread(i, as_gray=True) / 255 for i in sorted(image_dict2[im])]
            overlap = int(imgs[0].shape[0] * overlap_pct)
            xsize = imgs[0].shape[0]
            x00, x01, x10, x11 = imgs
            upper_slice = np.concatenate(
                [
                    x00[:, : xsize - overlap],
                    (x00[:, xsize - overlap :] + x01[:, :overlap]) // 2,
                    x01[:, overlap:],
                ],
                axis=1,
            )
            lower_slice = np.concatenate(
                [
                    x10[:, : xsize - overlap],
                    (x10[:, xsize - overlap :] + x11[:, :overlap]) // 2,
                    x11[:, overlap:],
                ],
                axis=1,
            )
            all_ = np.concatenate(
                [
                    upper_slice[: xsize - overlap],
                    (upper_slice[xsize - overlap :,] + lower_slice[:overlap]) // 2,
                    lower_slice[overlap:],
                ],
                axis=0,
            )
            all_ = np.where(all_ < 0.5, 0, 1).astype(bool)
            all_ = resize(
                all_,
                (original_cropped_size, original_cropped_size),
                anti_aliasing=False,
            )
            io.imsave(output_name + "/" + im + ".png", img_as_uint(all_))

    except Exception as e:
        print(e)

    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    folder = input("Provide folder path:\n-")
    stitch_images(folder)
