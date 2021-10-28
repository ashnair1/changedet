from typing import Any

import numpy as np
from scipy import linalg
from scipy.stats import chi2
from tqdm import tqdm

from changedet.algos.base import MetaAlgo
from changedet.algos.catalog import AlgoCatalog
from changedet.utils import InitialChangeMask, contrast_stretch, np_weight_stats


@AlgoCatalog.register("irmad")
class IRMAD(MetaAlgo):
    """Iteratively Reweighted Multivariate Alteration Detection

    The Multivariate Alteration Detection (MAD) algorithm aims to identify a linear transformation
    that minimises the correlation between the canonical components of the two images thereby
    maximising change information. Iteratively Reweighted (IR)-MAD is an improvement on the MAD
    approach where observations are iteratively reweighted in order to establish a better no change
    background which allows better separability between change and no-change.

    Accepted flags
    --------------
    - niter = Number of iterations IRMAD should be run
    - sig = Change map significance level
    - icm = Initial change mask

    References
    ----------
    - Nielsen, A. A. (2007). The regularized iteratively reweighted MAD method for change detection
    in multi- and hyperspectral data. IEEE Transactions on Image Processing, 16(2):463–478. Internet
    http://www2.compute.dtu.dk/pubdb/pubs/4695-full.html.


    """

    @classmethod
    def run(cls, im1: np.ndarray, im2: np.ndarray, flags: Any) -> np.ndarray:
        """Run IRMAD algorithm.

        Args:
            im1 (np.ndarray): Image 1 array
            im2 (np.ndarray): Image 2 array
            flags (dict): Flags for the algorithm

        Run `changedet --algo irmad algo_obj --help` for information on flags.
        """
        niter = flags.get("niter", 10)
        sig = flags.get("sig", 0.0001)
        apply_icm = flags.get("icm", False)
        logger = flags.get("logger", None)
        logger.info(
            "Running IRMAD algorithm for %d iteration(s) with significance level %f", niter, sig
        )

        ch1, r1, c1 = im1.shape

        m = r1 * c1
        N = ch1

        im1r = im1.reshape(N, m).T
        im2r = im2.reshape(N, m).T

        # Calculate ICM
        if apply_icm:
            icm = InitialChangeMask()
            change_mask = icm.prepare(im1, im2, plot=False)

            if change_mask is None:
                logger.warn("Invalid threshold. Skipping ICM")
                change_mask = np.ones((m, 1))
            else:
                change_mask = change_mask.reshape(m, 1)

            # Apply change mask
            im1r = im1r * change_mask
            im2r = im2r * change_mask

        # Center data
        im1r = im1r - np.mean(im1r, 0)
        im2r = im2r - np.mean(im2r, 0)

        x = np.concatenate((im1r, im2r), axis=1)

        # if not pvs:
        #     pvs = np.ones(r1 * c1)
        pvs = np.ones(r1 * c1, dtype=np.float32)
        itr = 0

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

                A1 = sigma12 @ np.linalg.solve(sigma22, sigma12.T)
                A2 = sigma12.T @ np.linalg.solve(sigma11, sigma12)

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

        cmap = np.copy(mads)
        idx = np.where(pvs > sig)[0]
        cmap[idx, :] = 0.0

        cmap = cmap.T.reshape(im1.shape)  # Change map
        mads = mads.T.reshape(im1.shape)  # MAD variates
        chisqr = chisqr.reshape((1, r1, c1))
        cmap = contrast_stretch(cmap, stretch_type="percentile")

        return cmap
