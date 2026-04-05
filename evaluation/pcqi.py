import cv2
import numpy as np
from scipy.signal import correlate2d


def compute_pcqi(ref_img, test_img):
    """
    Compute PCQI between reference and test image.

    Inputs:
        ref_img  : RGB or grayscale image (numpy array)
        test_img : RGB or grayscale image (numpy array)

    Returns:
        mpcqi : mean PCQI score (float)
    """

    # convert to grayscale if needed
    if len(ref_img.shape) == 3:
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
    if len(test_img.shape) == 3:
        test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)

    ref_img = ref_img.astype(np.float64)
    test_img = test_img.astype(np.float64)

    # match size if needed
    if ref_img.shape != test_img.shape:
        test_img = cv2.resize(test_img, (ref_img.shape[1], ref_img.shape[0]))

    # gaussian window
    g = cv2.getGaussianKernel(11, 1.5)
    window = g @ g.T
    window = window / np.sum(window)

    mu1 = correlate2d(ref_img, window, mode="valid")
    mu2 = correlate2d(test_img, window, mode="valid")

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = correlate2d(ref_img * ref_img, window, mode="valid") - mu1_sq
    sigma2_sq = correlate2d(test_img * test_img, window, mode="valid") - mu2_sq
    sigma12 = correlate2d(ref_img * test_img, window, mode="valid") - mu1_mu2

    sigma1_sq = np.maximum(sigma1_sq, 0)
    sigma2_sq = np.maximum(sigma2_sq, 0)

    C = 3.0
    L = 256.0

    term1 = (4.0 / np.pi) * np.arctan((sigma12 + C) / (sigma1_sq + C))
    term2 = (sigma12 + C) / (np.sqrt(sigma1_sq) * np.sqrt(sigma2_sq) + C)
    term3 = np.exp(-np.abs(mu1 - mu2) / L)

    pcqi_map = term1 * term2 * term3
    mpcqi = float(np.mean(pcqi_map))

    return mpcqi