from pathlib import Path

import numpy as np
import rasterio as rio
from scipy import linalg
from scipy.stats import chi2

from .base import MetaAlgo
from .catalog import AlgoCatalog


def np_covw(x, ws):
    # Do this after image reshaping
    # shape(x1) == shape(x2) == (C, H, W)
    # Reshape and transpose so shape(x1) == shape(x1) == (H*W,C)
    # Concatenate cols -> x = np.concatenate((x1,x2), axis=1)
    # Shape(x) = (H*W, 2*C)
    # Shape(ws) = (H*W,)
    # https://stats.stackexchange.com/questions/61225/correct-equation-for-weighted-unbiased-sample-covariance
    mean = np.ma.average(x, axis=0, weights=ws)
    mean2d = np.expand_dims(mean.data, axis=1)  # (H*W,) -> (H*W,1)
    xm = x - mean
    # np.isnan(xm).any() # Check if any element is Nan
    # xm.mul(w, axis=0) === np.multiply(xm, ws[:, np.newaxis])
    sigma2 = 1.0 / (ws.sum() - 1) * np.multiply(xm, ws[:, np.newaxis]).T.dot(xm)
    return sigma2.data, mean2d


def irmad(im1, im2, niter):
    """Runs the IRMAD algorithm

    Args:
        im1 (numpy.ndarray): Image 1 array
        im2 (numpy.ndarray): Image 2 array
        niter (int): Number of iterations

    Returns:
        tuple: (mad variates, chisqr)
    """

    itr = 0
    ch1, r1, c1 = im1.shape
    ch2, r2, c2 = im2.shape

    if (ch1, r1, c1) != (ch2, r2, c2):
        raise AssertionError(f"Image arrays should be of same shape")

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

    while itr < niter:
        itr += 1
        sigma, mean = np_covw(x, pvs)
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
        pvs = 1 - chi2(N).cdf(chisqr)
        # np.ma.average expects 1D weights
        pvs = pvs.squeeze()

    mads = np.reshape(mads, (r1, c1, ch1))
    chisqr = np.reshape(chisqr, (r1, c1))

    return mads, chisqr


@AlgoCatalog.register("irmad")
class IRMAD(MetaAlgo):
    """IRMAD algorithm"""

    @classmethod
    def run(cls, im1, im2, niter=1):
        if Path(im1).exists() & Path(im2).exists():
            im1 = rio.open(im1)
            im2 = rio.open(im2)
            arr1 = im1.read()
            arr2 = im2.read()

            mads, chisqr = irmad(arr1, arr2, niter=niter)

            profile = im1.profile
            profile["count"] = im1.count + 1
            outfile = "mad.tif"

            with rio.Env():
                with rio.open(outfile, "w", **profile) as dst:
                    for i in range(im1.count):
                        dst.write(mads[:, :, i], i + 1)
                    dst.write(chisqr, profile["count"])
            print(f"Change map written to {outfile}")

    @classmethod
    def help(cls):
        print(cls.__doc__)
