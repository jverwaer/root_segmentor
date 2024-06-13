"""
Module to process binary skeletons.
"""

import math

import numpy as np
from scipy.signal import convolve2d
from skimage import morphology, measure
from skimage.measure import label, regionprops
from skimage import morphology
import numpy as np


def retainLargestObject(binary_im: np.ndarray) -> np.ndarray:
    """ only retain largest object of a binarized image (perform labeling
    and delete all but largest object) """
    labeled = label(binary_im)
    max_area = max([props.area for props in regionprops(labeled)])

    return morphology.remove_small_objects(binary_im, max_area - 1)


def findJoints(skeleton: np.ndarray) -> np.ndarray:
    """
    Find pixels that correspond to joints (branches of the skeleton)
    returns booleans array indicating the location of the joints

    Parameters
    ----------
    skeleton : np.ndarray
        Boolean array representing the skeleton of an image

    Returns
    -------
    joints : np.ndarray
        Boolean array where True indicates that presence of a joint

    """
    mask = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]])
    convolved_down = convolve2d(skeleton, mask, mode='same')
    mask = np.rot90(mask)
    convolved_right = convolve2d(skeleton, mask, mode='same')
    mask = np.rot90(mask)
    convolved_top = convolve2d(skeleton, mask, mode='same')
    mask = np.rot90(mask)
    convolved_left = convolve2d(skeleton, mask, mode='same')

    plus = mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    joints = skeleton & (convolved_down > 0) & \
             (convolved_right > 0) & (convolved_top > 0) & \
             (convolved_left > 0) & \
             (convolve2d(skeleton, morphology.square(3), mode='same') > 3) | \
             (convolve2d(skeleton, plus, mode='same') == 4)
    return joints


def cleanSkeleton(skeleton: np.ndarray, minimalBranchLength: int = 100) -> np.ndarray:
    """
    Remove small branches from the skeleton

    Parameters
    ----------
    skeleton : np.ndarray
        Boolean array representing the skeleton of an image
    minimalBranchLength : int, optional
        Minimum length of branches to retain, by default 100

    Returns
    -------
    np.ndarray
        Skeleton with small branches removed
    """

    # find joints
    joints = findJoints(skeleton)

    # shatter the skeleton
    shattered = skeleton & (~joints)

    # remove small parts of skeleton
    return morphology.remove_small_objects(shattered, minimalBranchLength, connectivity=2) | joints


def segment_skeleton(skeleton: np.ndarray, widen_joints: bool = True) -> np.ndarray:
    """
    Segment the skeleton by removing small branches and labeling the remaining components.

    Parameters
    ----------
    skeleton : np.ndarray
        Boolean array representing the skeleton of an image
    widen_joints : bool, optional
        Flag indicating whether to widen the joints, by default True

    Returns
    -------
    np.ndarray
        Labeled image of the segmented skeleton (joints are removed so the original skeleton is shattered)
    """
    # select joints
    joints = findJoints(skeleton)

    if widen_joints:
        joints = morphology.dilation(joints, morphology.disk(10))

    # shatter the skeleton
    shattered = skeleton & (~joints)

    return measure.label(shattered)


def computeSkeletonLength_simple(skeleton: np.ndarray) -> float:
    """
    Compute length of skeleton (taking into account diagonal vs horizontal/vertical directions).
    Small error in calculations at T-sections as the diagonals are falsely counted there so only
    use when the number of T-sections is negligible.

    Parameters
    ----------
    skeleton : np.ndarray
        Boolean array representing the skeleton of an image

    Returns
    -------
    float
        Length of the skeleton
    """
    skeleton = np.array(skeleton, dtype = bool)
    mask = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
    convolved_diag = convolve2d(skeleton, mask, mode='same') * math.sqrt(2)/2
    convolved_diag[~skeleton] = 0
    mask = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    convolved_cross = convolve2d(skeleton, mask, mode='same') * 1/2
    convolved_cross[~skeleton] = 0
    return np.sum(convolved_diag + convolved_cross)



