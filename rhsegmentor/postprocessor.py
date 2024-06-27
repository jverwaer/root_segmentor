"""
Module that allows to post-process segmented images and measure 
"""

import math

import numpy as np

from skimage import (draw, measure, morphology)
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
import pandas as pd
import pylineclip as lc

from . import skeleton_processor as sp


def clean_predicted_roots(predicted_segmentation : np.ndarray,
                          small_objects_threshold : int = 500,
                          closing_diameter : int = 6) -> np.ndarray:
    """
    Functions that takes a predicted segmentation as inputs (roots are assumed to have value 1)
    and returns a cleaned (boolean) root image
    
    Parameters:
    ----------
    predicted_segmentation : np.ndarray
        The predicted segmentation image where roots are assumed to have a value of 1.
        
    small_objects_threshold : int, optional
        The threshold size (in pixels) below which small objects will be removed from the segmented image.
        Default is 500.
        
    closing_diameter : int, optional
        The diameter (in pixels) of the diamond-shaped structuring element used for morphological closing.
        Default is 6.
        
    Returns:
    -------
    np.ndarray
        The cleaned (boolean) root image where small objects have been removed and gaps have been closed.
    """
    seg = predicted_segmentation == 1
    no_small = morphology.remove_small_objects(seg, small_objects_threshold)
    closed = morphology.closing(no_small, morphology.diamond(closing_diameter))
    return closed

def measure_roots(clean_root_image: np.ndarray,
                  quantile: float = 0.95,
                  root_thickness: float = 7,
                  minimalBranchLength: int = 20) -> pd.DataFrame:
    """
    Function to compute length, orientation, center of each detected root (95% of feret diamter) and append it to a dataframe
    
    Parameters:
    ----------
    clean_root_image : np.ndarray
        The cleaned (boolean) root image where small objects have been removed and gaps have been closed.
        
    quantile : float, optional
        The quantile of the distance distribution to use for validation. Default is 0.95.
        
    root_thickness : float, optional
        The typical thickness of a root in terms of the number of pixels. Default is 7.
        
    minimalBranchLength : int, optional
        The minimal length of a branch in the skeletonized root image. Default is 20.
        
    Returns:
    -------
    pd.DataFrame
        A dataframe containing the X and Y coordinates of the root centroid, the root orientation, the root length (assuming linearity),
        the linear approximation check, the adjusted length in case the linear approximation is not okay, and the adjusted length with improved
        length computation (diagonal counts for sqrt(2)).
    """
    labeled = measure.label(clean_root_image)
    reg_props = measure.regionprops(labeled)
    lengths = []
    X = []
    Y = []
    orientations = []
    approx_check = []
    adjusted_length = []
    adjusted_length_with_sqrt2 = []
    for props in reg_props:

        # compute if root region can be approximated well using a line
        if check_linear_approximation(props, quantile = quantile, root_thickness = root_thickness):
            approx_check.append("OK")
            adjusted_length.append(props.feret_diameter_max*0.95)
            adjusted_length_with_sqrt2.append(props.feret_diameter_max*0.95)
        else:
            approx_check.append("Bad approx")

            # skeletonize and clean
            skeleton = skeletonize(props.image)
            cleaned = sp.cleanSkeleton(skeleton, minimalBranchLength=minimalBranchLength)
            #cleaned = morphology.remove_small_objects(cleaned, 10)
            adjusted_length.append(cleaned.sum())
            adjusted_length_with_sqrt2.append(sp.computeSkeletonLength_simple(cleaned))
            
        # compute line properties
        lengths.append(props.feret_diameter_max*0.95)
        X.append(props.centroid[1])
        Y.append(props.centroid[0])
        orientations.append(props.orientation)
    df = pd.DataFrame({"X" : X,                                                         # X coordinate centroid of root
                       "Y" : Y,                                                         # Y coordinate centroid of root
                       "orientation" : orientations,                                    # orientation root direction
                       "length" : lengths,                                              # lenght of root (assuming linearity)
                       "approx_check" : approx_check,                                   # check if linear approximation is okay
                       "adjusted_length" : adjusted_length,                             # adjusted length in case linear approximation is not okay (otherwise copy)
                       'adjusted_length_with_sqrt2' : adjusted_length_with_sqrt2})      # adjusted length with improved lenght computatation (diagonal counts for sqrt(2))
    return df
        

def check_linear_approximation(props: measure._regionprops.RegionProperties,
                               quantile: float = 0.95,
                               root_thickness: float = 7) -> bool:
    """
    Function to check if the linear approximation of the roots is acceptable for once connected component

    Parameters:
    ----------
    props : skimage.measure._regionprops.RegionProperties
        Region properties of one connected component

    quantile : float, optional
        Quantile of distance distribution to use for validation. Default is 0.95.

    root_thickness : float, optional
        Typical thickness of root (number of pixels). Default is 7.

    Returns:
    -------
    bool
    """
    # get bbox-binary
    bin = props.image
    # get line approximation, adjusted for bbox-translation to origin (x0, y0) -> (0, 0)
    y0, x0 = props.centroid
    x0 = x0 - props.bbox[1]
    y0 = y0 - props.bbox[0]
    orientation = props.orientation
    x2 = x0 - math.sin(orientation) * 0.5 * props.feret_diameter_max * 0.95
    y2 = y0 - math.cos(orientation) * 0.5 * props.feret_diameter_max * 0.95
    x3 = x0 + math.sin(orientation) * 0.5 * props.feret_diameter_max * 0.95
    y3 = y0 + math.cos(orientation) * 0.5 * props.feret_diameter_max * 0.95
    # get intersection with bbox
    x3,y3,x4,y4 = lc.cohensutherland(0,
                                    bin.shape[0]-1, 
                                    bin.shape[1]-1,
                                    0, 
                                    x2, 
                                    y2, 
                                    x3, 
                                    y3)
    # draw line
    line_coord = draw.line(round(y3), round(x3), round(y4), round(x4))
    A = np.zeros(bin.shape, dtype = bool)
    A[line_coord] = True
    
    # get distribution of distances 
    dt = distance_transform_edt(~A)
    return np.quantile(dt[bin], quantile) < root_thickness * 3
