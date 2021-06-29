import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from termcolor import colored


class ICM:
    def __init__(self, mode="hist"):
        self.mode = mode
        self.gmm = GMM(3)

    def prepare(self, im1, im2):
        # Linear stretch
        im1 = contrast_stretch(im1, stretch_type="percentile")
        im2 = contrast_stretch(im2, stretch_type="percentile")

        ch1, r1, c1 = im1.shape
        ch2, r2, c2 = im2.shape

        m = r1 * c1
        N = ch1

        im1r = im1.reshape(N, m).T
        im2r = im2.reshape(N, m).T

        diff = np.abs(im1r - im2r)

        _, mean, cov, pi = self.gmm.fit(diff)
        # idx = np.argsort(mean, axis=0)
        # cov = cov[idx]
        # pi = pi[idx]

        # icov = np.linalg.inv(cov)
        # det = np.linalg.det(cov)
        # logprior = np.log(pi)

        # import pdb; pdb.set_trace()


class GMM:
    def __init__(self, K, niter=100, tol=1e-4, reg_covar=1e-6):
        self.n_components = K
        self.tol = tol
        self.niter = niter
        self.reg_covar = reg_covar

    def __repr__(self):
        rep = f"GMM(n_components={self.n_components})"
        return rep

    def e_step(self, X, resp, means, cov, pi, sample_inds):
        """Expectation step

        Shape notation:

            N: number of samples
            D: number of features
            K: number of mixture components

        Args:
            X (numpy.ndarray): Data matrix of shape (N, D)
            resp (numpy.ndarray): Responsibility matrix of shape (N,K)
            means (numpy.ndarray): Means array of shape (D, K)
            cov (numpy.ndarray): Covariance matrix of shape (K,D,D) - full
            pi (numpy.ndarray): Mixture weights of shape (N,)
            sample_inds (array-like): Samples to be considered

        Returns:
            tuple:
            - resp (numpy.ndarray): Responsibility matrix of shape (N,K)
            - wpdf (numpy.ndarray): Unnormalised responsibility matrix of shape (N,K)
        """
        for k in range(self.n_components):
            resp[sample_inds, k] = pi[k] * multivariate_normal.pdf(X[sample_inds], means[k], cov[k])
        wpdf = resp.copy()  # For log likelihood computation
        # Safe normalisation
        a = np.sum(resp, axis=1, keepdims=True)
        idx = np.where(a == 0)[0]
        a[idx] = 1.0
        resp = resp / a
        return resp, wpdf

    def m_step(self, X, resp, means, cov, pi):
        """Maximisation step

        Shape notation:

            N: number of samples
            D: number of features
            K: number of mixture components

        Args:
            X (numpy.ndarray): Data matrix of shape (N, D)
            resp (numpy.ndarray): Responsibility matrix of shape (N,K)
            means (numpy.ndarray): Means array of shape (D, K)
            cov (numpy.ndarray): Covariance matrix of shape (K,D,D) - full
            pi (numpy.ndarray): Mixture weights of shape (N,)

        Returns:
            tuple:
            - means (numpy.ndarray): Means array of shape (D, K)
            - cov (numpy.ndarray): Covariance matrix of shape (K,D,D) - full
            - pi (numpy.ndarray): Mixture weights of shape (N,)
        """
        # M step
        n_samples, n_features = X.shape
        for k in range(self.n_components):
            nk = resp[:, k].sum()
            pi[k] = nk / n_samples
            means[k] = resp[:, k].dot(X) / nk
            delta = X - means[k]
            cov[k] = np.dot(resp[:, k] * delta.T, delta) / nk + np.eye(n_features) * self.reg_covar
        return means, pi, cov

    def fit(self, X, resp=None, sample_inds=None):
        """
        Fit a GMM to X with initial responsibility resp. If sample_inds are specified, only those
        indexes are considered.


        Args:
            X (numpy.ndarray): Data matrix
            resp (numpy.ndarray, optional): Initial responsibility matrix. Defaults to None.
            sample_inds (array-like, optional): Sample indexes to be considered. Defaults to None.

        Returns:
            resp (numpy.ndarray): Responsibility matrix
        """

        n_samples, n_features = X.shape

        if sample_inds is None:
            sample_inds = range(n_samples)

        means = np.zeros((self.n_components, n_features))
        cov = np.zeros((self.n_components, n_features, n_features))

        # Initialise
        # Mean -> random data point
        # Cov -> spherical covariance
        for k in range(self.n_components):
            # Random point in X as mean
            means[k] = X[np.random.choice(n_samples)]
            cov[k] = np.eye(n_features)

        pi = np.ones(self.n_components) / self.n_components
        lls = []

        if resp is None:
            resp = np.random.rand(n_samples, self.n_components)
            resp /= resp.sum(axis=1, keepdims=True)
        else:
            means, pi, cov = self.m_step(X, resp, means, cov, pi)

        # EM algorithm
        for i in range(self.niter):
            # resp_old = resp + 0.0

            # E step
            resp, wpdf = self.e_step(X, resp, means, cov, pi, sample_inds)
            # M step
            means, pi, cov = self.m_step(X, resp, means, cov, pi)

            # resp_flat = resp.ravel()
            # resp_old_flat = resp_old.ravel()
            # idx = np.where(resp.flat)[0]
            # ll = np.sum(resp_old_flat[idx] * np.log(resp_flat[idx]))
            ll = np.log(wpdf.sum(axis=1)).sum()
            lls.append(ll)

            # print(f"Log-likelihood:{ll}")
            if i > 1:
                if np.abs(lls[i] - lls[i - 1]) < self.tol:
                    # print("Exiting")
                    break

        return resp, means, cov, pi


class OnlineWeightStats:
    def __init__(self, N):
        self.mean = np.zeros(N)
        self.wsum = 1e-7
        self.xpsum = np.zeros((N, N))  # Sum of cross-products

    def update(self, X, weights=None):
        if weights is None:
            weights = np.ones(X.shape[0])
        for d, w in zip(X, weights):
            self.wsum += w
            upcon = w / self.wsum
            delta = d - self.mean
            self.mean += delta * upcon
            self.xpsum += np.outer(delta, delta) * w * (1 - upcon)

        self.cov = self.xpsum / self.wsum


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
    if ws is None:
        ws = np.ones(x.shape[0])
    mean = np.ma.average(x, axis=0, weights=ws)
    wmean = np.expand_dims(mean.data, axis=1)  # (H*W,) -> (H*W,1)
    wsigma = np.cov(x, rowvar=False, aweights=ws)
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
