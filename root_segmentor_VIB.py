# %% basic imports
import math
import os
import traceback
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from scipy.ndimage import distance_transform_edt
from skimage import (data, draw, feature, future, io, measure, morphology,
                     segmentation, transform)
from sklearn.ensemble import RandomForestClassifier
import pylineclip as lc
from skimage.morphology import skeletonize
import skeleton_processor as sp

# %% FUNCTION DEFENITIONS

def load_training_image(img_file = "EOS 550D_046.JPG",
                        root_traces_file = "EOS 550D_046 vertices.csv",
                        auto_transform = False):
    
    """
    
    Function to load a root training image as numpy RGB and its adjoining root traces
    
    PARAMETERS
    ----------
    img_file : str
        File name of RGB training image
    
    root_traces_file : str
        File containing root traces (a csv file, for an example see EOS 550D_046 vertices.csv)
        
    RETURNS
    -------
    
    im : np.array
        training image
    
    names : np.array of strings
        array of names of the roots
        
    vertices_s : (n x 2) np.array of floats where n is the nr of roots
        starting point of vertices (first coordinate in root_traces_file) where first column contains
        X-coordinates and 2nd column contains Y-coordinates
    
    vertices_e : (n x 2) np.array of floats where n is the nr of roots
        ending point of vertices (second coordinate in root_traces_file) where first column contains
        X-coordinates and 2nd column contains Y-coordinates
    
    """
    
    
    im = io.imread(img_file)    #[2000:3200,:,:]
    vertices = np.loadtxt(root_traces_file,
                          skiprows= 1,
                          usecols = (2,3),
                          delimiter=";")
    names = np.loadtxt(root_traces_file,
                          skiprows= 1,
                          usecols = (1),
                          delimiter=";",
                          dtype=str)

    # check if each trace contains exactly two vertices
    if  len(names) == 2*len(np.unique(names)):
        # slect names
        names = names[0::2]
        # split start and end vertices  (out: vertices_s, vertices_e)
        vertices_s = vertices[0::2, :]
        vertices_e = vertices[1::2, :]
    else: # in case curved roots are present
        vertices_s = []
        vertices_e = []
        names_list = []
        for name in np.unique(names):
            v = vertices[names == name,:]
            reps = np.tile(v[1:-1,:],2).reshape((-1,2))
            v = np.concatenate((v[np.newaxis,0,:], reps, v[np.newaxis,-1,:]), axis = 0)
            vertices_s.append(v[0::2, :])
            vertices_e.append(v[1::2, :])
            names_list.append(np.repeat(name, v.shape[0]-1))
        vertices_s = np.concatenate(vertices_s)
        vertices_e = np.concatenate(vertices_e)
        names = np.concatenate(names_list)

    # transform coordinates if needed (default is not to do this)
    if auto_transform:
        src = [[2683, 75],
            [2472, 82],
            [2682, 3373],
            [2536, 3370]]

        dst = [[83, 2501],
                [86, 2715],
                [3375, 2493],
                [3375, 2648]]

        vertices_s = transform_coordinates(src, dst, vertices_s)
        vertices_e = transform_coordinates(src, dst, vertices_e)

    return im, names, vertices_s, vertices_e
    
    
def flip_XY_RC(XY_or_RC):
    
    """
    Function to transform XY coordinates in ROW-COL coordinates or the other way around
    
    PARAMETERS
    ---------
    XY_or_RC : (n x 2) np.array of coordinates (XY or RC)
        XY or ROW-COL coordinates
        
    RETURNS
    -------
    (n x 2) np.array of coordinates (XY or RC)
    
    """
    
    return np.array(np.fliplr(XY_or_RC), dtype = int)


def transform_coordinates(src, dst, vertices):
    
    """
    Function to transform the coordinates in vertices based on an affine transform
    derived from scr and dst, both are (k x 2) arrays where the former contains
    the XY coordinates of k point in the original coordinate system and dst contains
    the XY' coordinates in the new coordinate system.
    
    Using src and dst, a transformation matrix is estimated and used to transform
    vertices to the new coordinate system.
    
    PARAMETERS
    ----------
    
    src : (k x 2) np.array
        coordinates in original coordinate system
    
    dst : (k x 2) np.array
        coordinates in new coordinate system
    
    RETURNS
    -------
    
    transformed_vertices : (n x 2) np.array of transformed vertices
    
    """
    
    tform = transform.estimate_transform('affine', np.array(src), np.array(dst))
    transformed_vertices = tform(vertices)
    
    return transformed_vertices

    
def create_root_mask(im, vertices_s_RC, vertices_e_RC, dilatation_radius = 7):
    
    """
    Function to create a root mask, allows to use the start/end points of the annotated traces
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
    
    # make mask with traces on it (out: root_mask)
    root_mask = np.zeros(im.shape[:2], dtype = bool)
    for i in range(len(vertices_e_RC)):
        rr, cc = draw.line(*vertices_s_RC[i,:], *vertices_e_RC[i,:])
        root_mask[rr, cc] = True
    root_mask = morphology.dilation(root_mask, morphology.disk(dilatation_radius))
    
    return root_mask
    

def create_root_buffer_background_image(root_mask,
                                        buffer_radius = 15,
                                        no_root_radius = 100):
    
    """
    Function to create a segmented image containing distinguishing: root (1), no-root (2)
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
    # check if no_root_radius is None
    if no_root_radius is None:
        no_root_radius = root_mask.shape[0]

    # part of image that is certainly no part of the roots
    dt = distance_transform_edt(~root_mask)
    no_root_mask = (dt > buffer_radius) & (dt < no_root_radius)
    
    # create training set
    root_buffer_background_image = np.array(root_mask, dtype = np.uint8) # root gets 1
    root_buffer_background_image[no_root_mask] = 2                       # no_root gets 2
    
    return root_buffer_background_image
    
    
def create_training_data(root_buffer_background_image):
    
    # find row indices that bound the root (such that: low_idx < root < high_idx)
    sum_bg_row = root_buffer_background_image.shape[1]*2
    tmp = np.where(np.sum(root_buffer_background_image == 1, 1) > 1)
    low_idx = np.min(tmp)
    high_idx = np.max(tmp)
    
    # create training dataset by only considering left half of image between 
    # bounds found above (with a tolerance)
    training_data = np.copy(root_buffer_background_image)
    training_data[:, round(training_data.shape[1]/2):] = 0
    training_data[:(low_idx-20), :] = 0
    training_data[(high_idx+20):, :] = 0
    
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
    


def train_segmentor(features, training_labels):
    
    """
    Function to train a random forest classifier given a feature image and a label image
    
    PARAMETERS
    ----------
    
    features : np.array
        (nrow x nkol x k) feature cube where k is the number of features (typically
        generated using the function im2features)
        
    training_labels : np.array
        (nrow x nkol) array where 0 = unlabeled; 1 = root; 2 = background
        
    RETURNS
    -------
    RandomForestClassifier instance (scikit-learn object)
    
    """
    
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                 max_depth=10, max_samples=0.05)
    clf = future.fit_segmenter(training_labels, features, clf)
    
    return clf
    
    


def predict_segmentor(clf, 
                   features_test_im):
    
    """
    Function to predict on a test image
    
    PARAMETERS
    ----------
    
    clf : RandomForestClassifier (actually any scikit-learn classifer)
        
    features_test_im : np.array
        (nrow x nkol x k) array of features
        
    RETURNS
    -------
    np.array of predicted labels (predicted segmentation)
    
    """
    

    result = future.predict_segmenter(features_test_im, clf)
    return result


def clean_predicted_roots(predicted_segmentation, small_objects_threshold = 500, closing_diameter = 6):
    """
    Functions that takes a predicted segmentation as inputs (roots are assumed to have value 1)
    and retunrs cleaned (boolean) root image
    """
    seg = predicted_segmentation == 1
    no_small = morphology.remove_small_objects(seg, small_objects_threshold)
    closed = morphology.closing(no_small, morphology.diamond(closing_diameter))
    return closed

def measure_roots(clean_root_image, quantile = 0.95, root_thickness = 7, minimalBranchLength = 20):
    """
    Function to compute length, orientation, center of each detected root (95% of feret diamter) and append it to a dataframe
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
        

def check_linear_approximation(props, quantile = 0.95, root_thickness = 7):
    """
    Function to check if the linear approximation of the roots is acceptable for once connected component

    PARAMETERS
    ----------

    props : skimage.measure._regionprops.RegionProperties
        Region properties of one connected component

    qualtile : float
        Quantile of distance distrubtion to use for validations

    root_thickness : float
        Typical thickness of root (nr of pixels)

    RETURNS
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




# %% Functions for visualization

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
        if check_linear_approximation(props, quantile = quantile, root_thickness = root_thickness):
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


# %% functions for large-scale data handling (computing features for multiple images, saving and reading)

def imgs_to_XY_data(img_file_list = None,
                    root_traces_file_list = None,
                    auto_transform = True,
                    dilatation_radius = 7,
                    buffer_radius = 15,
                    no_root_radius = 100,
                    sigma_max = 16):
    """
    Function to transform a set of images and adjoining root trace files to Features (X) and Labels (Y)
    that can be used for training a segmentation model. 

    PARAMTERS
    ---------

    img_file_list : list
        list of file names of images (typically to jpg-images) e.g. [./sample_data/EOS 550D_046.JPG"]
    root_traces_file_list : list
        list of file names of traces e.g. "./sample_data/EOS 550D_046 vertices.csv"
    
    RETURNS
    -------

    Function retursn None, and per image features and labels are saved in a numpy array ...FEATURES.npy and LABELS.npy
    """
    if img_file_list is None:
        img_file_list = ["./sample_data/EOS 550D_046.JPG", "./sample_data/EOS 550D_044.JPG"]
    if root_traces_file_list is None:
        root_traces_file_list = [fname[:-4] + " vertices.csv" for fname in img_file_list]

    for img_file, root_traces_file in zip(img_file_list, root_traces_file_list):
            if not os.path.isfile(img_file[:-4] + "FEATURES.npy"):
                try:
                    # load image
                    im, names, vertices_s, vertices_e = load_training_image(img_file,
                                                                            root_traces_file,
                                                                            auto_transform = auto_transform)

                    # transform coordinates
                    # OUTCOMMENTED as now handled by the function 'load_training_data'
                    #src = [[2683, 75],
                    #    [2472, 82],
                    #    [2682, 3373],
                    #    [2536, 3370]]

                    #dst = [[83, 2501],
                    #        [86, 2715],
                    #        [3375, 2493],
                    #        [3375, 2648]]

                    #vertices_s = transform_coordinates(src, dst, vertices_s)
                    #vertices_e = transform_coordinates(src, dst, vertices_e)
                    vertices_s_RC = flip_XY_RC(vertices_s)
                    vertices_e_RC = flip_XY_RC(vertices_e)

                    # create root mask
                    root_mask = create_root_mask(im, 
                                                 vertices_s_RC, 
                                                 vertices_e_RC,
                                                 dilatation_radius = dilatation_radius)
                    root_buffer_background = create_root_buffer_background_image(root_mask,
                                                                                 buffer_radius = buffer_radius,
                                                                                 no_root_radius = no_root_radius)

                    # create training data labels
                    training_labels = create_training_data(root_buffer_background)

                    # compute features
                    features = im2features(im, sigma_max = sigma_max)

                    # flatten labels and features
                    features_flat = features[training_labels > 0,:]
                    label_flat = training_labels[training_labels > 0]

                    # save features
                    np.save(img_file[:-4] + "FEATURES", features_flat)
                    np.save(img_file[:-4] + "LABELS", label_flat)
                except Exception as e:
                    print("Problem processing " + img_file)
                    traceback.print_exc()
            else:
                print("skipping " + img_file, "as it FEATURES exist already")



def compile_training_dataset_from_precomputed_features(features_file_list = None,
                                                       labels_file_list = None,
                                                       sample_fraction = (0.3, 0.1),
                                                       seed = 1):

    """
    Function to compile a training set from a set of FEATUERS and LABELS npy-files by
    loading each file, taking a sample of relative size sample_fraction[0] for roots and
    sample_fraction[1] for background.

    RETURNS
    -------

    first output is X set, second output is Y set 
    
    """
    np.random.seed(seed)
    if features_file_list is None:
        features_file_list = ["./sample_data/EOS 550D_046FEATURES.npy", "./sample_data/EOS 550D_044FEATURES.npy"]
    if labels_file_list is None:
        labels_file_list = [fname[:-12] + "LABELS.npy" for fname in features_file_list]

    features_list = []
    labels_list = []

    for feature_file, label_file in zip(features_file_list, labels_file_list):
        X = np.load(feature_file)
        Y = np.load(label_file)

        # nr of root (1) and bg (2) pixels in Y
        n_1 = np.sum(Y == 1)
        n_2 = np.shape(Y)[0] - n_1

        # subsample
        s_1 = np.random.choice(n_1, int(n_1*sample_fraction[0]))
        s_2 = np.random.choice(n_2, int(n_2*sample_fraction[1]))

        sampleX1 = X[Y == 1,:][s_1,:]
        sampleY1 = Y[Y == 1][s_1]

        sampleX2 = X[Y == 2,:][s_2,:]
        sampleY2 = Y[Y == 2][s_2]

        features_list.append(sampleX1)
        features_list.append(sampleX2)
        labels_list.append(sampleY1)
        labels_list.append(sampleY2)
        
    return np.concatenate(features_list), np.concatenate(labels_list)

