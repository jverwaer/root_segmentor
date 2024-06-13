
"""
Module that allows to visualize and save root detection results. 
"""
import math

import matplotlib.pyplot as plt
from skimage import (measure, segmentation)
from skimage.morphology import skeletonize

from . import skeleton_processor as sp
from . import postprocessor as pp


def show_traces(vertices_s, vertices_e, im = None):
    
    """
    Function to show root traces on image
    
    PARAMETERS
    ----------
     
    vertices_s : (k x 2) np.array
        coordinates in xy coordinates of start of trace 
        
    vertices_e : (k x 2) np.array
        coordinates in xy coordinates of  end of trace 
        
    im : np.array
        RGB image or root image (binary) or root_bg_image if None, only traces are 
        plotted (typically for overlay with existing background image)
    
    """
    
    if not im is None:
        plt.imshow(im)
    for i in range(len(vertices_e)):
        plt.plot((vertices_s[i,0], vertices_e[i,0]),
                   (vertices_s[i,1], vertices_e[i,1]), '-*')
        

def show_predicted_segmentation(im, predicted_segmentation):
    
    """
    Function to create contour plot of predicted_segmentation overlayed on
    a background image
    """
    
    # show segmentation result
    plt.imshow(segmentation.mark_boundaries(im, predicted_segmentation, mode='thick'))


def draw_detected_roots(clean_root_image, 
                        original_im, 
                        figSize = None,
                        quantile = 0.95,
                        root_thickness = 7,
                        minimalBranchLength = 20):
    """
    Visualize detected roots and their lenghts (major axis with 95% of Feret diameter)

    PARAMETERS
    ----------

    quantile : float
        Quantile for detecting deviation from linearity (see check_linear_approximation)
    
    root_thickness : float
        Thickness of root for detecting deviation from linearity (see check_linear_approximation)

    minimalBranchLength : int
        Minimal lenght of a branch in the skeleton of a set of connected roots (for pruning)

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

def save_detected_roots_im(clean_root_image, original_im, fname, 
                            quantile = 0.95,
                            root_thickness = 7,
                            minimalBranchLength = 20):
    """
    Save Visualization of detected roots and their lenghts (calls draw_detected_roots using
    non-interactive mode of matplotlib)
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