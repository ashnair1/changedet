import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from scipy.stats import multivariate_normal, norm
from termcolor import colored


def check_pos_semi_def(mat: np.ndarray) -> bool:
    """Test whether matrix is positive semi-definite

    Ref:
    - https://scicomp.stackexchange.com/a/12984/39306
    - https://stackoverflow.com/a/63911811/10800115

    Args:
        mat (np.ndarray): Input matrix

    Returns:
        bool: True if mat is positive semi-definite else False
    """

    try:
        reg_mat = mat + np.eye(mat.shape[0]) * 1e-3
        np.linalg.cholesky(reg_mat)
        return True
    except np.linalg.LinAlgError:
        return False


class InitialChangeMask:
    """Initial Change Mask

    Create a change mask to remove strong changes and enable better radiometric
    normalisation.


    References
    ----------
    - P. R. Marpu, P. Gamba and M. J. Canty, "Improving Change Detection Results
      of IR-MAD by Eliminating Strong Changes," in IEEE Geoscience and Remote
      Sensing Letters, vol. 8 no. 4, pp. 799-803, July 2011,
      doi:10.1109/LGRS.2011.2109697.
    """

    def __init__(self, mode: str = "hist") -> None:
        self.mode = mode
        self.gmm = GMM(3, cov_type="full")

    @staticmethod
    def plot(mean: np.ndarray, cov: np.ndarray, thresh: float) -> None:

        sigma = np.sqrt(cov)
        mix_names = ["No change", "Ambiguous", "Pure Change"]
        mix_colours = ["r", "g", "b"]
        # f1 = plt.figure()
        for m, sig, name, colour in zip(mean, sigma, mix_names, mix_colours):
            p = np.linspace(m - 3 * sig, m + 3 * sig, 100)
            plt.plot(p, norm.pdf(p, m, sig), label=name, color=colour)
        plt.legend()
        plt.tight_layout()
        plt.margins(0)
        plt.axvline(x=thresh, color="k", linestyle="--")
        plt.show()

    def prepare(self, im1: np.ndarray, im2: np.ndarray, plot: bool = True) -> np.ndarray:
        # Linear stretch
        im1 = contrast_stretch(im1, stretch_type="percentile")
        im2 = contrast_stretch(im2, stretch_type="percentile")

        ch1, r1, c1 = im1.shape

        m = r1 * c1
        N = ch1

        im1r = im1.reshape(N, m).T
        im2r = im2.reshape(N, m).T

        diff = np.abs(im1r - im2r)
        # Max difference
        diff = diff.max(axis=1)[:, np.newaxis]

        _, mean, cov, pi = self.gmm.fit(diff)

        mean = mean.flatten()
        cov = cov.flatten()
        pi = pi.flatten()

        # Sort in ascending order
        idx = np.argsort(mean)
        mean = mean[idx]
        cov = cov[idx]
        pi = pi[idx]

        # Refer https://gist.github.com/ashnair1/433ffbc1e747f80067f8a0439e346279
        # for derivation of the equation
        a = cov[1] - cov[0]
        b = -2 * (mean[0] * cov[1] - mean[1] * cov[0])
        c = (
            mean[0] ** 2 * cov[1]
            - mean[1] ** 2 * cov[0]
            + 2 * np.log((cov[0] * pi[1]) / (cov[1] * pi[0])) * (cov[0] * cov[1])
        )
        roots = np.roots([a, b, c])

        m1 = mean[0]
        m2 = mean[1]
        s1 = roots[0]
        s2 = roots[1]

        thresh = (
            ((m1 > m2) * (m1 > s1) * (m2 < s1) * s1)
            + ((m1 > m2) * (m1 > s2) * (m2 < s2) * s2)
            + ((m2 > m1) * (m2 > s1) * (m1 < s1) * s1)
            + ((m2 > m1) * (m2 > s2) * (m1 < s2) * s2)
        )

        # Plot distributions and threshold
        if plot:
            self.plot(mean, cov, thresh)

        if not thresh:
            return None

        icm = np.where(diff < thresh, 0, 1)
        icm = icm.reshape(r1, c1)
        return icm


def estimate_full_covariance(
    X: np.ndarray, resp: np.ndarray, nk: np.ndarray, means: np.ndarray, reg_covar: float
) -> np.ndarray:
    """Estimate full covariance matrix

    Shape notation:

            N: number of samples
            D: number of features
            K: number of mixture components

    Args:
        X (numpy.ndarray): Data matrix of shape (N, D)
        resp (numpy.ndarray): Responsibility matrix of shape (N,K)
        nk (numpy.ndarray): Total responsibility per cluster of shape (K,)
        means (numpy.ndarray): Means array of shape (K, D)
        reg_covar (float): Regularisation added to diagonal of covariance matrix \
            to ensure positive definiteness

    Returns:
        cov (numpy.ndarray): Covariance matrix of shape (K,D,D)
    """
    n_components, n_features = means.shape
    cov = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        delta = X - means[k]
        cov[k] = (
            np.dot(resp[:, k] * delta.T, delta) / nk[k] + np.eye(n_features) * reg_covar
        )
    return cov


def estimate_tied_covariance(
    X: np.ndarray, resp: np.ndarray, nk: np.ndarray, means: np.ndarray, reg_covar: float
) -> np.ndarray:
    """Estimate tied covariance matrix

    Shape notation:

            N: number of samples
            D: number of features
            K: number of mixture components

    Args:
        X (numpy.ndarray): Data matrix of shape (N, D)
        resp (numpy.ndarray): Responsibility matrix of shape (N,K)
        nk (numpy.ndarray): Total responsibility per cluster of shape (K,)
        means (numpy.ndarray): Means array of shape (K, D)
        reg_covar (float): Regularisation added to diagonal of covariance matrix \
            to ensure positive definiteness

    Returns:
        cov (numpy.ndarray): Covariance matrix of shape (K,D,D)
    """
    n_components, n_features = means.shape
    avg_X2 = np.dot(X.T, X)
    avg_means2 = np.dot(nk * means.T, means)
    cov = (avg_X2 - avg_means2) / nk.sum() + np.eye(n_features) * reg_covar

    # Convert (D,D) cov to (K,D,D) cov where all K cov matrices are equal
    cov = np.repeat(cov[np.newaxis], n_components, axis=0)
    return cov


class GMM:
    def __init__(
        self,
        K: int,
        niter: int = 100,
        *,
        cov_type: str = "full",
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
    ):
        self.n_components = K
        self.cov_type = cov_type
        self.tol = tol
        self.niter = niter
        self.reg_covar = reg_covar

    def init_cluster_params(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialse cluster parameters

        Shape notation:

            N: number of samples
            D: number of features
            K: number of mixture components

        Initialisation method:

            Initialise means to a random data point in X
            Initialse cov to a spherical covariance matrix of variance 1
            Initialse pi to uniform distribution

        Args:
            X (numpy.ndarray): Data matrix of shape (N,D)

        Returns:
            tuple:
            - means (numpy.ndarray): Means array of shape (K, D)
            - cov (numpy.ndarray): Covariance matrix of shape (K,D,D)
            - pi (numpy.ndarray): Mixture weights of shape (K,)
        """

        n_samples, n_features = X.shape
        means = np.zeros((self.n_components, n_features))
        cov = np.zeros((self.n_components, n_features, n_features))

        # Initialise
        # Mean -> random data point
        # Cov -> spherical covariance - all clusters have same diagonal cov
        # matrix and diagonal elements are all equal
        for k in range(self.n_components):
            means[k] = X[np.random.choice(n_samples)]
            cov[k] = np.eye(n_features)

        pi = np.ones(self.n_components) / self.n_components

        return means, cov, pi

    def __repr__(self) -> str:
        return f"GMM(n_components={self.n_components})"

    def e_step(
        self,
        X: np.ndarray,
        resp: np.ndarray,
        means: np.ndarray,
        cov: np.ndarray,
        pi: np.ndarray,
        sample_inds: ArrayLike,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Expectation step

        Shape notation:

            N: number of samples
            D: number of features
            K: number of mixture components

        Args:
            X (numpy.ndarray): Data matrix of shape (N, D)
            resp (numpy.ndarray): Responsibility matrix of shape (N,K)
            means (numpy.ndarray): Means array of shape (K, D)
            cov (numpy.ndarray): Covariance matrix of shape (K,D,D) - full
            pi (numpy.ndarray): Mixture weights of shape (K,)
            sample_inds (array-like): Samples to be considered

        Returns:
            tuple:
            - resp (numpy.ndarray): Responsibility matrix of shape (N,K)
            - wpdf (numpy.ndarray): Unnormalised responsibility matrix of shape (N,K)
        """
        for k in range(self.n_components):
            resp[sample_inds, k] = pi[k] * multivariate_normal.pdf(
                X[sample_inds], means[k], cov[k]
            )
        wpdf = resp.copy()  # For log likelihood computation
        # Safe normalisation
        a = np.sum(resp, axis=1, keepdims=True)
        idx = np.where(a == 0)[0]
        a[idx] = 1.0
        resp = resp / a
        return resp, wpdf

    def m_step(
        self, X: np.ndarray, resp: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Maximisation step

        Shape notation:

            N: number of samples
            D: number of features
            K: number of mixture components

        Args:
            X (numpy.ndarray): Data matrix of shape (N, D)
            resp (numpy.ndarray): Responsibility matrix of shape (N,K)

        Returns:
            tuple:
            - means (numpy.ndarray): Means array of shape (K, D)
            - cov (numpy.ndarray): Covariance matrix of shape (K,D,D) - full
            - pi (numpy.ndarray): Mixture weights of shape (K,)
        """
        # M step
        n_samples, _ = X.shape
        nk = resp.sum(axis=0)
        means = np.dot(resp.T, X) / nk[:, np.newaxis]
        pi = nk / n_samples

        if self.cov_type == "tied":
            cov = estimate_tied_covariance(X, resp, nk, means, self.reg_covar)
        else:
            cov = estimate_full_covariance(X, resp, nk, means, self.reg_covar)

        return means, pi, cov

    def fit(
        self,
        X: np.ndarray,
        resp: Optional[np.ndarray] = None,
        sample_inds: Optional[ArrayLike] = None,
    ) -> np.ndarray:
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

        n_samples, _ = X.shape

        if sample_inds is None:
            sample_inds = range(n_samples)

        means, cov, pi = self.init_cluster_params(X)
        lls = []

        if resp is None:
            resp = np.random.rand(n_samples, self.n_components)
            resp /= resp.sum(axis=1, keepdims=True)
        else:
            means, pi, cov = self.m_step(X, resp)

        # EM algorithm
        for i in range(self.niter):
            # resp_old = resp + 0.0

            # E step
            resp, wpdf = self.e_step(X, resp, means, cov, pi, sample_inds)
            # M step
            means, pi, cov = self.m_step(X, resp)

            # resp_flat = resp.ravel()
            # resp_old_flat = resp_old.ravel()
            # idx = np.where(resp.flat)[0]
            # ll = np.sum(resp_old_flat[idx] * np.log(resp_flat[idx]))
            ll = np.log(wpdf.sum(axis=1)).sum()
            lls.append(ll)

            # print(f"Log-likelihood:{ll}")
            if i > 1 and np.abs(lls[i] - lls[i - 1]) < self.tol:
                # print("Exiting")
                break

        return resp, means, cov, pi


class OnlineWeightStats:
    def __init__(self, N: int):
        self.mean = np.zeros(N)
        self.wsum = 1e-7
        self.xpsum = np.zeros((N, N))  # Sum of cross-products

    def update(self, X: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
        if weights is None:
            weights = np.ones(X.shape[0])
        for d, w in zip(X, weights):
            self.wsum += w
            upcon = w / self.wsum
            delta = d - self.mean
            self.mean += delta * upcon
            self.xpsum += np.outer(delta, delta) * w * (1 - upcon)

        self.cov = self.xpsum / self.wsum


def np_weight_stats(
    x: np.ndarray, ws: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
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


def contrast_stretch(
    img: np.ndarray,
    *,
    target_type: str = "uint8",
    stretch_type: str = "minmax",
    percentile: Tuple[int, int] = (2, 98),
) -> np.ndarray:
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
    return np.clip(g, minout, maxout)


def histogram_equalisation(
    im: np.ndarray, nbr_bins: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
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

    def format(self, record: logging.LogRecord) -> str:
        date = colored(
            "[%(asctime)s]:%(name)s:%(module)s:%(lineno)d:%(levelname)s:", "cyan"
        )
        msg = "%(message)s"
        if record.levelno == logging.WARNING:
            fmt = date + " " + colored("WRN", "red", attrs=["blink"]) + " " + msg
        elif record.levelno in [logging.ERROR, logging.CRITICAL]:
            fmt = (
                date
                + " "
                + colored("ERR", "red", attrs=["blink", "underline"])
                + " "
                + msg
            )
        elif record.levelno == logging.DEBUG:
            fmt = date + " " + colored("DBG", "yellow", attrs=["blink"]) + " " + msg
        else:
            fmt = date + " " + msg
        if hasattr(self, "_style"):
            # Python3 compatibility
            self._style._fmt = fmt
        self._fmt = fmt
        return super(_ColorFormatter, self).format(record)


def init_logger(name: str = "logger", output: Optional[str] = None) -> logging.Logger:
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
        output_path = Path(output)
        logfile = (
            output_path
            if output_path.suffix in [".txt", ".log"]
            else output_path / "log.txt"
        )
        Path.mkdir(output_path.parent)

        filehandler = logging.FileHandler(logfile)
        filehandler.setLevel(logging.DEBUG)
        filehandler.setFormatter(_ColorFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(filehandler)
    return logger


def histplot(xlist: ArrayLike, xlabel: List[str], bins: Optional[int] = 50) -> Figure:
    """Plot multiple histograms in the same figure

    Args:
        xlist (arraylike): Sequence
        xlabel (list[str]): Sequence label
        bins (int, optional): Histogram bins. Defaults to 50.

    Returns:
        matplotlib.figure.figure: Figure with histograms
    """
    f = plt.figure()
    for i, j in zip(xlist, xlabel):
        plt.hist(i[:, :, 0].flatten(), bins=bins, label=j)
    plt.legend()
    return f
