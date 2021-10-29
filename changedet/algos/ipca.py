from typing import Any

import numpy as np

from changedet.algos.base import MetaAlgo
from changedet.algos.catalog import AlgoCatalog
from changedet.utils import GMM, OnlineWeightStats


@AlgoCatalog.register("ipca")
class IteratedPCA(MetaAlgo):
    """Iterated PCA Change Classifier.

    The number of unchanged pixels usually outnumber changed pixels and since they're highly
    correlated over time, they should lie along the first principal axis while the changed pixels
    lie along the second axis. However, since the principal components are calculated from the
    covariance matrix computed for **all** pixels, the no-change axis might be poorly defined.
    Iterated PCA solves this by calculating the principal components iteratively and weighting
    each pixel by its probability to be no change pixels.

    Accepted flags
    --------------
    - niter = Number of iterations IPCA should be run

    References
    ----------
    - Wiemker, R. (1997). An iterative spectral-spatial Bayesian labeling approach for unsupervised
    robust change detection on remotely sensed multispectral imagery. In Proceedings of the 7th
    International Conference on Computer Analysis of Images and Patterns, volume LCNS 1296, pages
    263–370.
    - Canty, M.J. (2019). Image Analysis, Classification, and Change Detection in Remote Sensing:
    With Algorithms for Python (4th ed.). CRC Press. https://doi.org/10.1201/9780429464348
    """

    @classmethod
    def run(cls, im1: np.ndarray, im2: np.ndarray, **flags: Any) -> np.ndarray:
        """Run IPCA algorithm.

        Args:
            im1 (np.ndarray): Image 1 array
            im2 (np.ndarray): Image 2 array
            flags (dict): Flags for the algorithm

        Run `changedet --algo ipca algo --help` for information on flags
        """
        niter = flags.get("niter", 5)
        band_idx = flags.get("band")
        logger = flags.get("logger", None)

        band_str = "all bands" if band_idx == -1 else "band " + str(band_idx)

        logger.info("Running IPCA algorithm for %d iteration(s) on %s", niter, band_str)

        bands, rows, cols = im1.shape

        cmaps = []
        n_clusters = 2
        for band in range(bands):
            # TODO: Allow band sublist
            if band_idx == -1:
                logger.info(f"Processing band {band + 1}")
            else:
                logger.info(f"Processing band {band_idx}")

            fim1 = im1[band].flatten()
            fim2 = im2[band].flatten()

            X = np.zeros((rows * cols, 2))
            X[:, 0] = fim1 - np.mean(fim1)
            X[:, 1] = fim2 - np.mean(fim2)

            # Initial PCA
            ows = OnlineWeightStats(2)
            ows.update(X)
            eivals, eigvecs = np.linalg.eigh(ows.cov)
            eivals = eivals[::-1]
            eigvecs = eigvecs[:, ::-1]

            pcs = X @ eigvecs

            gmm = GMM(n_clusters, 60)

            for itr in range(niter):
                # Calculate responsibility matrix
                U = np.random.rand(X.shape[0], n_clusters)
                for _ in range(n_clusters):
                    sigma = np.sqrt(eivals[1])
                    unfrozen = np.where(np.abs(pcs[:, 1]) >= sigma)[0]
                    frozen = np.where(np.abs(pcs[:, 1]) < sigma)[0]
                    U[frozen, 0] = 1.0
                    U[frozen, 1] = 0.0
                    for j in range(2):
                        U[:, j] = U[:, j] / np.sum(U, 1)

                r, _, _, _ = gmm.fit(X, U, unfrozen)

                tmp = r[:, 0] + 1 - 1
                ows.update(X, tmp)
                eivals, eigvecs = np.linalg.eigh(ows.cov)
                eivals = eivals[::-1]
                eigvecs = eigvecs[:, ::-1]

                pcs = X @ eigvecs

            cmap = np.reshape(r[:, 1], (rows, cols))
            cmaps.append(cmap)

        return np.array(cmaps)
