from pathlib import Path

import numpy as np
import rasterio as rio
from scipy import linalg
from scipy.stats import chi2
from tqdm import tqdm

from changedet.algos.base import MetaAlgo
from changedet.algos.catalog import AlgoCatalog
from changedet.utils.postprocess import scale_image


def np_weight_stats(x, ws):
    """Calculate weighted mean and sample covariance

    Args:
        x (numpy.ndarray): Data matrix
        ws (numpy.ndarray): Weight vector

    Returns:
        tuple: tuple containing:

            wsigma (numpy.array): Weighted covariance matrix
            wmean (numpy.array): Weighted mean
    """
    mean = np.ma.average(x, axis=0, weights=ws)
    wmean = np.expand_dims(mean.data, axis=1)  # (H*W,) -> (H*W,1)
    xm = x - mean
    # np.isnan(xm).any() # Check if any element is Nan
    # xm.mul(w, axis=0) === np.multiply(xm, ws[:, np.newaxis])
    sigma2 = 1.0 / (ws.sum() - 1) * np.multiply(xm, ws[:, np.newaxis]).T.dot(xm)
    wsigma = sigma2.data
    return wsigma, wmean


def irmad(im1, im2, niter, sig, logger):
    """Runs the IRMAD algorithm

    Args:
        im1 (numpy.ndarray): Image 1 array
        im2 (numpy.ndarray): Image 2 array
        niter (int): Number of iterations
        sig (float): Change map significance level

    Returns:
        cmap (numpy.ndarray): Change map
    """

    itr = 0
    ch1, r1, c1 = im1.shape
    ch2, r2, c2 = im2.shape

    if (ch1, r1, c1) != (ch2, r2, c2):
        logger.critical("Image array shapes do not match")
        raise AssertionError

    m = r1 * c1
    N = ch1

    im1r = im1.reshape(N, m).T
    im2r = im2.reshape(N, m).T

    # Center data
    im1r = im1r - np.mean(im1r, 0)
    im2r = im2r - np.mean(im2r, 0)

    x = np.concatenate((im1r, im2r), axis=1)

    # if not pvs:
    #     pvs = np.ones(r1 * c1)
    pvs = np.ones(r1 * c1, dtype=np.float32)

    with tqdm(total=niter) as pbar:
        while itr < niter:
            itr += 1
            sigma, mean = np_weight_stats(x, pvs)
            ms1 = mean[:N].T
            ms2 = mean[N:].T
            sigma11 = sigma[:ch1, :ch1]
            sigma12 = sigma[:ch1, ch1:]
            sigma21 = sigma[ch1:, :ch1]
            sigma22 = sigma[ch1:, ch1:]
            assert np.allclose(sigma21, sigma12.T)

            A1 = sigma12 @ np.linalg.inv(sigma22) @ sigma12.T
            A2 = sigma12.T @ np.linalg.inv(sigma11) @ sigma12

            # Scipy's linalg.eig returns normalised eigenvectors
            lamda1, eigvec1 = linalg.eigh(A1, sigma11)
            lamda2, eigvec2 = linalg.eigh(A2, sigma22)

            # sort eigenvalues & eigenvectors in descending order
            # TODO: Remove idx2. It's the same as idx
            idx = np.argsort(lamda1)
            lamda1 = lamda1[idx][::-1]
            eigvec1 = eigvec1[:, idx][:, ::-1]

            idx2 = np.argsort(lamda2)
            lamda2 = lamda2[idx2][::-1]
            eigvec2 = eigvec2[:, idx2][:, ::-1]

            rho2 = lamda1  # or lamda2 they're the same
            rho = np.sqrt(rho2)

            # canonical correlations

            # ensure positive correlation between each pair of canonical variates
            diag = np.diag(eigvec1.T @ sigma12 @ eigvec2)
            signs = np.diag(diag / np.abs(diag))  # Diagonal matrix of signs
            eigvec2 = eigvec2 @ signs

            # Calculate p value weights
            sig2s = 2 * (1 - rho)
            sig2s = np.reshape(np.tile(sig2s, [m]), (m, N))
            ms1 = np.reshape(np.tile(ms1[0], [m]), (m, N))
            ms2 = np.reshape(np.tile(ms2[0], [m]), (m, N))
            cv1 = (im1r - ms1) @ eigvec1
            cv2 = (im2r - ms2) @ eigvec2
            mads = cv1 - cv2

            chisqr = (np.square(mads) / sig2s).sum(axis=1)
            # Upper tailed test
            pvs = 1 - chi2(N).cdf(chisqr)
            # np.ma.average expects 1D weights
            pvs = pvs.squeeze()
            pbar.update()

    cmap = mads
    idx = np.where(pvs > sig)[0]
    cmap[idx, :] = 0.0

    cmap = cmap.reshape((r1, c1, ch1))  # Change map
    mads = mads.reshape((r1, c1, ch1))  # MAD variates
    chisqr = chisqr.reshape((r1, c1))

    return cmap


def lin2pcstr(x):
    #  2% linear stretch
    x = bytestr(x)
    hist, bin_edges = np.histogram(x, 256, (0, 256))
    cdf = hist.cumsum()
    lower = 0
    i = 0
    while cdf[i] < 0.02 * cdf[-1]:
        lower += 1
        i += 1
    upper = 255
    i = 255
    while cdf[i] > 0.98 * cdf[-1]:
        upper -= 1
        i -= 1
    fp = (bin_edges - lower) * 255 / (upper - lower)
    fp = np.where(bin_edges <= lower, 0, fp)
    fp = np.where(bin_edges >= upper, 255, fp)
    return np.interp(x, bin_edges, fp)


def bytestr(x):
    mx = np.max(x)
    mn = np.min(x)
    if mx - mn > 0:
        x = (x - mn) * 255.0 / (mx - mn)
    x = np.where(x < 0, 0, x)
    x = np.where(x > 255, 255, x)
    return x


@AlgoCatalog.register("irmad")
class IRMAD(MetaAlgo):
    """IRMAD algorithm

    Invariant to affine transforms

    Accepted flags:
    - niter = Number of iterations IRMAD should be run
    - sig = Change map significance level

    Reference:
    A. A. Nielsen, “The Regularized Iteratively Reweighted {MAD} Method for Change Detection in
    Multi- and Hyperspectral Data” IEEE Trans. Image Process., vol. 16, no. 2, pp. 463–478, 2007.


    """

    @classmethod
    def run(cls, im1, im2, flags):
        """Run IRMAD algorithm

        Args:
            im1 (str): Path to image 1
            im2 (str): Path to image 2
            flags (dict): Flags for the algorithm

        Run `changedet --algo irmad algo --help` for information on flags
        """
        niter = flags.get("niter", 1)
        sig = flags.get("sig", 0.0001)
        logger = flags.get("logger", None)
        logger.info(
            "Running IRMAD algorithm for %d iteration(s) with significance level %f", niter, sig
        )

        if Path(im1).exists() & Path(im2).exists():
            im1 = rio.open(im1)
            im2 = rio.open(im2)
            arr1 = im1.read()
            arr2 = im2.read()

            cmap = irmad(arr1, arr2, niter=niter, sig=sig, logger=logger)
            cmap = scale_image(cmap, np.uint8)

            profile = im1.profile
            outfile = "irmad_cmap.tif"

            with rio.Env():
                with rio.open(outfile, "w", **profile) as dst:
                    for i in range(im1.count):
                        dst.write(cmap[:, :, i], i + 1)
            logger.info("Change map written to %s", outfile)
            # Changemap by default does not appear as it should i.e. stretching
