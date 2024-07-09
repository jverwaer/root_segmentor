import numpy as np
from scipy.signal import convolve2d
from skimage import morphology, measure
import math

def retainLargestObject(binary_im):
    """ only retain largest object of a binarized image (perform labeling
    and delete all but largest object) """
    labeled = measure.label(binary_im)
    max_area = max([props.area for props in measure.regionprops(labeled)])

    return morphology.remove_small_objects(binary_im, max_area - 1)


def findJoints(skeleton):
    """

    Find pixels that correspond to joints (branches of the skeleton)
    returns booleans array indicating the location of the joints

    Parameters
    ----------
    skeleton : np.array
        Boolean array representing the skeleton of an image

    Returns
    -------
    joints : np.array
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


def cleanSkeleton(skeleton, minimalBranchLength = 100):
    """ Remove small branches from the skeleton """

    # find joints
    joints = findJoints(skeleton)

    # shatter the skeleton
    shattered = skeleton & (~joints)

    # remove small parts of skeleton
    return morphology.remove_small_objects(shattered, minimalBranchLength, connectivity=2) | joints


def segment_skeleton(skeleton, widen_joints=True):
    # select joints
    joints = findJoints(skeleton)

    if widen_joints:
        joints = morphology.dilation(joints, morphology.disk(10))

        # shatter the skeleton
    shattered = skeleton & (~joints)

    return measure.label(shattered)


def computeSkeletonLength_simple(skeleton):
    """
    Compute length of skeleton (taking into account diagonal vs horizontal/vertical directions).
    Small error in calculationts at T-sections as the diagonals are falsely counted there so only
    use when the number of T-sections is neglegible

    """
    skeleton = np.array(skeleton, dtype = bool)
    mask = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
    convolved_diag = convolve2d(skeleton, mask, mode='same') * math.sqrt(2)/2
    convolved_diag[~skeleton] = 0
    mask = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    convolved_cross = convolve2d(skeleton, mask, mode='same') * 1/2
    convolved_cross[~skeleton] = 0
    return np.sum(convolved_diag + convolved_cross)



