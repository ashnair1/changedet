import numpy as np


def contrast_stretch(img, *, target_type="uint8", stretch_type="default", percentile=(2, 98)):
    """
    Change image distribution to cover full range of target_type

    Args:
        img (numpy.ndarray): Input image
        target_type (dtype): Target type of rescaled image.
        stretch_type (str): Types of contrast stretching. One of ["default", "percentile"]
        percentile (tuple): Cut off percentiles. Used if stretch_type == "percentile

    Returns:
        numpy.ndarray: Rescaled image
    """

    type_info = np.iinfo(target_type)
    minout = type_info.min
    maxout = type_info.max

    if stretch_type == "percentile":
        lower, upper = np.nanpercentile(img, percentile)
    else:
        lower = np.min(img)
        upper = np.max(img)

    # Contrast Stretching
    a = (maxout - minout) / (upper - lower)
    b = minout - a * lower
    g = a * img + b
    scaled = np.clip(g, minout, maxout)
    return scaled


def histogram_equalisation(im, nbr_bins=256):
    # Refer http://www.janeriksolem.net/histogram-equalization-with-python-and.html
    # get image histogram
    imhist, bins = np.histogram(im.flatten(), nbr_bins)
    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    out = np.interp(im.flatten(), bins[:-1], cdf)
    return out.reshape(im.shape), cdf
