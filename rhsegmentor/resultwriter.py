
"""
Module that allows to visualize and save root detection results. 
"""
import math
import numpy as np

import matplotlib.pyplot as plt
from skimage import (measure, segmentation)
from skimage.morphology import skeletonize

from . import skeleton_processor as sp
from . import postprocessor as pp


def show_traces(vertices_s: np.ndarray,
                vertices_e: np.ndarray,
                im: np.ndarray = None) -> None:
    """
    Function to show root traces on image
    
    Parameters
    ----------
    vertices_s : np.ndarray
        Coordinates in xy coordinates of start of trace 
    vertices_e : np.ndarray
        Coordinates in xy coordinates of end of trace 
    im : np.ndarray, optional
        RGB image or root image (binary) or root_bg_image. If None, only traces are 
        plotted (typically for overlay with existing background image)
    
    Returns
    -------
    None
    """
    if not im is None:
        plt.imshow(im)
    for i in range(len(vertices_e)):
        plt.plot((vertices_s[i,0], vertices_e[i,0]), (vertices_s[i,1], vertices_e[i,1]), '-*')
        

def show_predicted_segmentation(im: np.ndarray,
                                predicted_segmentation: np.ndarray) -> None:
    """
    Function to create a contour plot of the predicted segmentation overlayed on a background image.

    Parameters
    ----------

    - im: np.ndarray
        The background image.
    - predicted_segmentation: np.ndarray
        The predicted segmentation.

    Returns
    -------

    None
    """
    
    # show segmentation result
    plt.imshow(segmentation.mark_boundaries(im, predicted_segmentation, mode='thick'))


def draw_detected_roots(clean_root_image: np.ndarray, 
                        original_im: np.ndarray, 
                        figSize: tuple = None,
                        quantile: float = 0.95,
                        root_thickness: float = 7,
                        minimalBranchLength: int = 20) -> np.array:
    """
    Visualize detected roots and their lengths (major axis with 95% of Feret diameter)

    PARAMETERS
    ----------

    clean_root_image : np.ndarray
        Binary image of the detected roots
    
    original_im : np.ndarray
        Original image on which the roots are detected
    
    figSize : tuple, optional
        Size of the figure (width, height) in inches. If not provided, the default size is used.
    
    quantile : float
        Quantile for detecting deviation from linearity (see check_linear_approximation)
    
    root_thickness : float
        Thickness of root for detecting deviation from linearity (see check_linear_approximation)

    minimalBranchLength : int
        Minimal length of a branches in the skeleton to retain (used in skeleton pruning)


    Returns
    -------
    np.ndarray
        The modified original image with the detected roots visualized

    """

    if figSize is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize = figSize)
    #to_show = original_im.copy()
    #to_show[clean_root_image, 0] = 255
    #ax.imshow(to_show)

    original_im = original_im.copy()
    ax.imshow(segmentation.mark_boundaries(original_im, clean_root_image, mode='thick'))

    labeled = measure.label(clean_root_image)
    reg_props = measure.regionprops(labeled)
    for i, props in enumerate(reg_props):
        y0, x0 = props.centroid
        orientation = props.orientation
        x2 = x0 - math.sin(orientation) * 0.5 * props.feret_diameter_max * 0.95
        y2 = y0 - math.cos(orientation) * 0.5 * props.feret_diameter_max * 0.95
        x3 = x0 + math.sin(orientation) * 0.5 * props.feret_diameter_max * 0.95
        y3 = y0 + math.cos(orientation) * 0.5 * props.feret_diameter_max * 0.95
        if pp.check_linear_approximation(props, quantile = quantile, root_thickness = root_thickness):
            ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
            ax.plot((x0, x3), (y0, y3), '-r', linewidth=2.5)
        else:
            ax.plot((x0, x2), (y0, y2), '-b', linewidth=2.5)
            ax.plot((x0, x3), (y0, y3), '-b', linewidth=2.5)

            # skeletonize and clean
            skeleton = skeletonize(props.image)
            cleaned = sp.cleanSkeleton(skeleton, minimalBranchLength=minimalBranchLength)
            #cleaned = morphology.remove_small_objects(cleaned, 3)

            # add to image
            bbox_slice = original_im[props.bbox[0]:props.bbox[2], props.bbox[1]:props.bbox[3],:] 
            bbox_slice[cleaned,0] = 255
            bbox_slice[cleaned,1] = 0
            bbox_slice[cleaned,2] = 0
        ax.text(x0, y0, str(i))

        ax.plot(x0, y0, '.g', markersize=15)
    ax.imshow(segmentation.mark_boundaries(original_im, clean_root_image, mode='thick'))
    return original_im

def save_detected_roots_im(clean_root_image: np.ndarray, 
                            original_im: np.ndarray, 
                            fname: str, 
                            quantile: float = 0.95,
                            root_thickness: float = 7,
                            minimalBranchLength: int = 20) -> None:
    """
    Save Visualization of detected roots and their lengths.

    This function saves a visualization of the detected roots and their lengths using the non-interactive mode of matplotlib.
    
    Parameters:
    - clean_root_image: The image containing the detected roots.
    - original_im: The original image.
    - fname: The filename to save the visualization.
    - quantile: The quantile value used for visualization (default: 0.95).
    - root_thickness: The thickness of the root lines in the visualization (default: 7).
    - minimalBranchLength: The minimal length of a branch to be considered as a root (default: 20).
    """
    FIGSIZE = (50, 30)
    plt.ioff()
    draw_detected_roots(clean_root_image,
                        original_im, 
                        figSize = FIGSIZE,
                        quantile = quantile,
                        root_thickness = root_thickness,
                        minimalBranchLength = minimalBranchLength)
    plt.savefig(fname)
    plt.close('all')
    plt.ion()