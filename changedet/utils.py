import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from termcolor import colored


def np_weight_stats(x, ws=None):
    """Calculate weighted mean and sample covariance.

    Args:
        x (numpy.ndarray): Data matrix of shape (N,D)
        ws (numpy.ndarray, optional): Weight vector of shape (N,). Defaults to None

    Returns:
        tuple:
        - wsigma (numpy.ndarray): Weighted covariance matrix
        - wmean (numpy.ndarray): Weighted mean
    """
    if not ws:
        ws = np.ones(x.shape[0])
    mean = np.ma.average(x, axis=0, weights=ws)
    wmean = np.expand_dims(mean.data, axis=1)  # (H*W,) -> (H*W,1)
    xm = x - mean
    # np.isnan(xm).any() # Check if any element is Nan
    # xm.mul(w, axis=0) === np.multiply(xm, ws[:, np.newaxis])
    sigma2 = 1.0 / (ws.sum() - 1) * np.multiply(xm, ws[:, np.newaxis]).T.dot(xm)
    wsigma = sigma2.data
    return wsigma, wmean


def contrast_stretch(img, *, target_type="uint8", stretch_type="minmax", percentile=(2, 98)):
    """Change image distribution to cover full range of target_type.

    Types of contrast stretching:
    - minmax (Default)
    - percentile

    Args:
        img (numpy.ndarray): Input image
        target_type (dtype): Target type of rescaled image. Defaults to "uint8".
        stretch_type (str): Types of contrast stretching. Defaults to "minmax".
        percentile (tuple): Cut off percentiles if stretch_type = "percentile". Defaults to (2, 98).

    Returns:
        scaled (numpy.ndarray): Rescaled image
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


class _ColorFormatter(logging.Formatter):
    """
    Color Logging Formatter

    Refer: https://github.com/tensorpack/dataflow/blob/master/dataflow/utils/logger.py
    """

    def format(self, record):
        date = colored("[%(asctime)s]:%(name)s:%(module)s:%(lineno)d:%(levelname)s:", "cyan")
        msg = "%(message)s"
        if record.levelno == logging.WARNING:
            fmt = date + " " + colored("WRN", "red", attrs=["blink"]) + " " + msg
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            fmt = date + " " + colored("ERR", "red", attrs=["blink", "underline"]) + " " + msg
        elif record.levelno == logging.DEBUG:
            fmt = date + " " + colored("DBG", "yellow", attrs=["blink"]) + " " + msg
        else:
            fmt = date + " " + msg
        if hasattr(self, "_style"):
            # Python3 compatibility
            self._style._fmt = fmt
        self._fmt = fmt
        return super(_ColorFormatter, self).format(record)


def init_logger(name="logger", output=None):
    """
    Initialise changedet logger

    Args:
        name (str, optional): Name of this logger. Defaults to "logger".
        output (str, optional): Path to folder/file to write logs. If None, logs are not written
    """
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.DEBUG)

    # Output logs to terminal
    streamhandler = logging.StreamHandler(sys.stdout)
    streamhandler.setLevel(logging.INFO)
    streamhandler.setFormatter(_ColorFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(streamhandler)

    # Output logs to file
    if output:
        output = Path(output)
        if output.suffix in [".txt", ".log"]:
            logfile = output
        else:
            logfile = output / "log.txt"
        Path.mkdir(output.parent)

        filehandler = logging.FileHandler(logfile)
        filehandler.setLevel(logging.DEBUG)
        filehandler.setFormatter(_ColorFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(filehandler)
    return logger


def histplot(xlist, xlabel, bins=50):
    """Plot multiple histograms in the same figure

    Args:
        xlist (list[]): Sequence
        xlabel (list[str]): Sequence label
        bins (int, optional): Histogram bins. Defaults to 50.

    Returns:
        matplotlib.pyplot.figure: Figure with histograms
    """
    f = plt.figure()
    for i, j in zip(xlist, xlabel):
        plt.hist(i[:, :, 0].flatten(), bins=bins, label=j)
    plt.legend()
    return f
