"""
Module that allows to create masks from root annotations and extract features for 
root segmentation.
"""    

import numpy as np
from skimage import draw, morphology, feature
from scipy.ndimage import distance_transform_edt
from functools import partial


def root_segmentation_mask(im : np.ndarray,
                           vertices_s_RC : np.ndarray,
                           vertices_e_RC : np.ndarray,
                           dilatation_radius : int =7,
                           buffer_radius : int =15,
                           no_root_radius : int =100) -> np.ndarray:
    """
    Function to create a root segmentation mask that allows to use the start/end points of the annotated roots
    to create an integer image  image distinguishing: root (1), no-root (2) and unknown (0).
    
    Pixels crossed by on annotated root segments are set to 1, the line-strucures obtaine that way are dilated with
    `dilatation_radius` pixels (the thickness of the root is thus 2 * `dilatation_radius`). The thus obtained root
    pixels are surrounded by a bufferof unknown (0) pixels with radius `buffer_radius` (as these areas are hard precisely segment). 
    Beyound this buffer, the remaining pixels are treated as non-root (2) up to a distance `no_root_radius` of the roots. 
    Pixels beyond `no_root_radius` are set to unknown (0) to avoid an over-representation of non-relevant background for training. NOTE: if the
    entire background should be treated as such, set `no_root_radius` to None.

    
    PARAMETERS
    ----------
    im : np.array
        RGB image of roots
    
    vertices_s_RC : (k x 2) np.array
        coordinates in row-column coordinates of start of trace 
    
    vertices_e_RC : (k x 2) np.array
        coordinates in row-column coordinates of  end of trace 
        
    dilatation_radius : int
        radius used for dilating the root traces
    
    buffer_radius : int
        size of buffer zone around root pixels

    no_root_radius : int
        all pixels between buffer_radius and no_root_radius are used as training data
        instances for 'no root' class (gets label 2). If None, it is set to nr of rows
        in the image meaning that it is ignored. Default is 100
    
    RETURNS
    -------
    root_mask : np.array of dilated root traces
    """

    root_mask = create_root_mask(im, vertices_s_RC, vertices_e_RC, dilatation_radius)
    return create_root_buffer_background_image(root_mask, buffer_radius, no_root_radius)




def create_root_mask(im, vertices_s_RC, vertices_e_RC, dilatation_radius=7):
    """
    Function to create a root mask, allows to use the start/end points of the annotated roots
    to create a binary image (root = 1, no-root = 0). Note: no exclusion zone is used here around
    the roots 
    
    PARAMETERS
    ----------
    im : np.array
        RGB image of roots
    
    vertices_s_RC : (k x 2) np.array
        coordinates in row-column coordinates of start of trace 
    
    vertices_e_RC : (k x 2) np.array
        coordinates in row-column coordinates of  end of trace 
        
    dilatation_radius : int
        radius used for dilating the root traces
    
    RETURNS
    -------
    root_mask : np.array of dilated root traces
    """
    root_mask = np.zeros(im.shape[:2], dtype=bool)
    for i in range(len(vertices_e_RC)):
        rr, cc = draw.line(*vertices_s_RC[i, :], *vertices_e_RC[i, :])
        root_mask[rr, cc] = True
    root_mask = morphology.dilation(root_mask, morphology.disk(dilatation_radius))
    return root_mask
    

def create_root_buffer_background_image(root_mask, buffer_radius=15, no_root_radius=100):
    """
    Function to create a segmented image distinguishing: root (1), no-root (2)
    and unknown (0) where unknown is a buffer zone around the roots
    
    PARAMETERS
    ----------
    root_mask : np.array
        binary image where root pixels are one
    
    buffer_radius : int
        size of buffer zone around root pixels

    no_root_radius : int
        all pixels between buffer_radius and no_root_radius are used as training data
        instances for 'no root' class (gets label 2). If None, it is set to nr of rows
        in the image meaning that it is ignored. Default is 100
    
    RETURNS
    -------
    root_buffer_background_image : np.array 
    """
    if no_root_radius is None:
        no_root_radius = root_mask.shape[0]

    dt = distance_transform_edt(~root_mask)
    no_root_mask = (dt > buffer_radius) & (dt < no_root_radius)
    
    root_buffer_background_image = np.array(root_mask, dtype=np.uint8)
    root_buffer_background_image[no_root_mask] = 2
    
    return root_buffer_background_image


    
def create_training_image(root_buffer_background_image):
    """
    Function to create a root segementation mask. 
    
    PARAMETERS
    ----------
    root_buffer_background_image : np.ndarray
        segmented image containing root (1), no-root (2), and unknown (0)
    
    RETURNS
    -------
    training_data : np.ndarray
        training dataset with a specific region of the image
    """
    sum_bg_row = root_buffer_background_image.shape[1] * 2
    tmp = np.where(np.sum(root_buffer_background_image == 1, 1) > 1)
    low_idx = np.min(tmp)
    high_idx = np.max(tmp)
    
    training_data = np.copy(root_buffer_background_image)
    training_data[:, round(training_data.shape[1] / 2):] = 0
    training_data[:(low_idx - 20), :] = 0
    training_data[(high_idx + 20):, :] = 0
    
    return training_data
    

def im2features(im, sigma_min=1, sigma_max=16):
    """
    Simple wrapper around feature.multiscale_basic_features to compute 
    features of a given image
    
    PARAMETERS
    ----------
    im : np.array
        RGB image
    
    sigma_min : int
        minimal value for smooting kernel bandwidth
        
    sigma_max : int
        maximal value for smooting kernel bandwidth
    
    RETURNS
    -------
    np.array with size (nrow x ncol x k) where k is the number of features
    """
    features_func = partial(feature.multiscale_basic_features,
                            intensity=True, edges=True, texture=True,
                            sigma_min=sigma_min, sigma_max=sigma_max,
                            channel_axis=-1)
    return features_func(im)
