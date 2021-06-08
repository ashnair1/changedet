import numpy as np


def scaleMinMax(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


def scaleStd(x):
    return (x - (np.nanmean(x) - np.nanstd(x) * 2)) / (
        (np.nanmean(x) + np.nanstd(x) * 2) - (np.nanmean(x) - np.nanstd(x) * 2)
    )


def scaleCCC(x, lower=2, upper=98):
    return (x - np.nanpercentile(x, lower)) / (
        np.nanpercentile(x, upper) - np.nanpercentile(x, lower)
    )


def convert(img, target_type_min, target_type_max, target_type):
    # Refer https://stackoverflow.com/a/59193141/10800115
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


def scale_image(img, target_type=np.uint8, method="CCC"):
    h, w, c = img.shape
    if method == "CCC":
        for i in range(c):
            img[:, :, i] = scaleCCC(img[:, :, i])

    # To check distribution
    import matplotlib.pyplot as plt

    plt.hist(img[:, :, 0].flatten(), bins=50)
    plt.show()

    img = convert(img, 0, 255, target_type)
    return img


# You could also use
# def scale_skimage(img, lower, upper):
#     p2, p98 = np.percentile(cmap, (2,98))
#     img = exposure.rescale_intensity(img, in_range=(p2,p98))
#     return img
