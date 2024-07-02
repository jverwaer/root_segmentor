
"""
Module that allows to facilitate batch processing of images

This module provides functions for batch processing of images and adjoining root trace files to generate
features and labels that can be used for training a segmentation model. The module includes the following
functions:

- `imgs_to_XY_data`: Transforms a set of images and root trace files to features (X) and labels (Y) and
  saves them as .npy files.
- `compile_training_dataset_from_precomputed_features`: Compiles a training set from precomputed features
  and labels.

Please refer to the individual function docstrings for more details on their usage and parameters.
"""

import os
import numpy as np
import traceback
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
import pandas as pd
from sklearn.base import BaseEstimator

from . import dataloader as dl
from . import featureextractor as fe
from . import postprocessor as pp
from . import pixelclassifier as pc
from . import resultwriter as rw
from . import utils


# a palette for saving segmentation masks as rgb images
# Make a palette
palette = [255,0,0,    # 0=red
           0,255,0,    # 1=green
           0,0,255,    # 2=blue
           255,255,0,  # 3=yellow
           0,255,255]  # 4=cyan
# Pad with zeroes to 768 values, i.e. 256 RGB colours
palette = palette + [0]*(768-len(palette))



def imgs_to_XY_data(img_file_list : list[str] = None,
                    root_traces_file_list : list[str] = None,
                    auto_transform : bool = True,
                    dilatation_radius : int = 7,
                    buffer_radius : int = 15,
                    no_root_radius : int = 100,
                    sigma_max : int = 16,
                    save_dir : str = './',
                    save_masks_as_im : bool = False) -> None:
    """
    Function to transform a set of images and adjoining root trace files to Features (X) and Labels (Y)
    that can be used for training a segmentation model. Computed Features (X) and Labels (Y) are stored as
    .npy files. See NOTES for details.

    PARAMETERS
    ----------
    img_file_list : list[str], optional
        List of file names of images (typically jpg-images). If not provided, default images will be used.
    root_traces_file_list : list[str], optional
        List of file names of root trace files. If not provided, default trace files will be used.
    auto_transform : bool, optional
        Flag indicating whether to automatically transform the coordinates of root traces. Default is True.
    dilatation_radius : int, optional
        Radius for dilating the root traces. Default is 7.
    buffer_radius : int, optional
        Radius for creating the root buffer background image. Default is 15.
    no_root_radius : int, optional
        Radius for excluding the root region from the buffer background image. Default is 100.
    sigma_max : int, optional
        Maximum sigma value for computing image features. Default is 16.
    save_dir : str (or None)
        Directory to save to (defaults to working directory). When set to None, the results will be saved
        in the directory of the original images
    save_masks_as_im : bool (default False)
        Save the segmentation masks (used for training) as png-images
    
    RETURNS
    -------
    None
    
    NOTES
    -----
    - For each image and root trace file pair, the function performs the following steps:
        1. Loads the image and root trace data.
        2. Transforms the coordinates of the root traces if auto_transform is True.
        3. Creates a root mask using the transformed coordinates.
        4. Creates a root buffer background image using the root mask.
        5. Creates training data labels using the root buffer background image.
        6. Computes image features using the original image.
        7. Saves the flattened features and labels as numpy arrays.
    - If an error occurs during processing, the function prints an error message and the traceback.
    - If the features for an image already exist, the function skips processing that image.
    """
    if img_file_list is None:
        img_file_list = ["./sample_data/EOS 550D_046.JPG", "./sample_data/EOS 550D_044.JPG"]
    if root_traces_file_list is None:
        root_traces_file_list = [fname[:-4] + " vertices.csv" for fname in img_file_list]

    for img_file, root_traces_file in zip(img_file_list, root_traces_file_list):
            
            # file name for saving (if None save in same directory as reading)
            features_fname = utils.get_save_fname(img_file, save_dir, "FEATURES.npy")
            labels_fname = utils.get_save_fname(img_file, save_dir, "LABELS.npy")
            img_fname = utils.get_save_fname(img_file, save_dir, "MASK.png")

            if not os.path.isfile(features_fname):
                try:
                    # load image
                    im, names, vertices_s, vertices_e = dl.load_training_image(img_file,
                                                                            root_traces_file,
                                                                            auto_transform = auto_transform)
                    
                    vertices_s_RC = dl.flip_XY_RC(vertices_s)
                    vertices_e_RC = dl.flip_XY_RC(vertices_e)

                    # create root mask                    
                    root_mask = fe.create_root_mask(im, 
                                                 vertices_s_RC, 
                                                 vertices_e_RC,
                                                 dilatation_radius = dilatation_radius)
                    root_buffer_background = fe.create_root_buffer_background_image(root_mask,
                                                                                 buffer_radius = buffer_radius,
                                                                                 no_root_radius = no_root_radius)

                    # create training data labels
                    # training_labels = fe.create_training_image(root_buffer_background)
                    training_labels = root_buffer_background



                    # compute features
                    features = fe.im2features(im, sigma_max = sigma_max)

                    # flatten labels and features
                    features_flat = features[training_labels > 0,:]
                    label_flat = training_labels[training_labels > 0]

                    # save features
                    np.save(features_fname, features_flat)
                    np.save(labels_fname, label_flat)

                    if save_masks_as_im:
                        # save segmentation mask as image
                        pi = Image.fromarray(training_labels,'P')
                        # Put the palette in
                        pi.putpalette(palette)
                        # Display and save
                        pi.save(img_fname)

                except Exception as e:
                    print("Problem processing " + img_file)
                    print("Traceback of -- ", img_file )
                    traceback.print_exc()
                    print("End traceback -- ", img_file )
            else:
                print("skipping " + img_file, "as it FEATURES exist already")



def compile_training_dataset_from_precomputed_features(features_file_list: list[str] = None,
                                                       labels_file_list: list[str] = None,
                                                       sample_fraction: tuple[float, float] = (0.3, 0.1),
                                                       seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to compile a training set from a set of FEATURES and LABELS npy-files by
    loading each file, taking a sample of relative size sample_fraction[0] for roots and
    sample_fraction[1] for background.

    PARAMETERS
    ----------
    features_file_list : list[str], optional
        List of file names of features npy-files. If not provided, default files will be used.
    labels_file_list : list[str], optional
        List of file names of labels npy-files. If not provided, default files will be used.
    sample_fraction : tuple[float, float], optional
        Tuple of two floats representing the relative size of the sample for roots and background.
        Default is (0.3, 0.1).
    seed : int, optional
        Seed value for random number generation. Default is 1.

    RETURNS
    -------
    tuple[np.ndarray, np.ndarray]
        The first output is the X set (features), and the second output is the Y set (labels).
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
        s_1 = np.random.choice(n_1, int(n_1 * sample_fraction[0]))
        s_2 = np.random.choice(n_2, int(n_2 * sample_fraction[1]))

        sampleX1 = X[Y == 1, :][s_1, :]
        sampleY1 = Y[Y == 1][s_1]

        sampleX2 = X[Y == 2, :][s_2, :]
        sampleY2 = Y[Y == 2][s_2]

        features_list.append(sampleX1)
        features_list.append(sampleX2)
        labels_list.append(sampleY1)
        labels_list.append(sampleY2)

    return np.concatenate(features_list), np.concatenate(labels_list)



def extract_rh_props(im : np.ndarray, 
                     clf : BaseEstimator,
                     fname : str = 'None',
                     sigma_max : int = 10,
                     small_objects_threshold : int = 150,
                     closing_diameter : int = 4,
                     root_thickness : float = 7.0,
                     minimalBranchLength : int = 10,
                     save_dir : str = "./"):
    
    """
    Extracts root properties from an RGB image using a trained classifier.

    Parameters:
    - im (np.ndarray): An RGB image.
    - clf (BaseEstimator): A trained classifier for root segmentation.
    - fname (str): A file name (added as a column in the resulting dataframe).
    - sigma_max (int, optional): The maximum sigma value for feature computation. Defaults to 10.
    - small_objects_threshold (int, optional): The threshold for removing small objects in the predicted segmentation. Defaults to 150.
    - closing_diameter (int, optional): The diameter for the closing operation in root cleaning. Defaults to 4.
    - root_thickness (float, optional): The assumed thickness of the roots. Defaults to 7.0.
    - minimalBranchLength (int, optional): The minimal length of a root branch. Defaults to 10.
    - save_dir (str, optional): The directory to save the output images. Defaults to "./".

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the computed root properties for the image.

    """

    # compute features
    features = fe.im2features(im, sigma_max = sigma_max)
    # predict
    predicted_segmentation = pc.predict_segmentor(clf, features)
    # clean detected roots
    roots = pp.clean_predicted_roots(predicted_segmentation,
                                    small_objects_threshold=small_objects_threshold,
                                    closing_diameter = closing_diameter)
    # compute root properties
    results_df = pp.measure_roots(roots,
                                    root_thickness = root_thickness,
                                    minimalBranchLength = minimalBranchLength)
    results_df["fname"] = fname
    # save image for quality check
    fname_save = utils.get_save_fname(fname = fname,
                                    save_dir = save_dir,
                                    suffix = "result.png")
    rw.save_detected_roots_im(clean_root_image = roots,
                            original_im = im,
                            fname = fname_save,
                            root_thickness = 7,
                            minimalBranchLength = 10)

    return results_df


def batch_extract_rh_props(file_list : list[str], **kwargs):
    """
    Extracts root properties from a list of images using a trained classifier.

    Parameters:
    - file_list (list[str]): A list of file paths to the input images.

    kwargs includes: 
    - clf (BaseEstimator): A trained classifier for root segmentation.
    - sigma_max (int, optional): The maximum sigma value for feature computation. Defaults to 10.
    - small_objects_threshold (int, optional): The threshold for removing small objects in the predicted segmentation. Defaults to 150.
    - closing_diameter (int, optional): The diameter for closing operation in root cleaning. Defaults to 4.
    - root_thickness (float, optional): The assumed thickness of the roots. Defaults to 7.0.
    - minimalBranchLength (int, optional): The minimal length of a root branch. Defaults to 10.
    - save_dir (str, optional): The directory to save the output images. Defaults to "./".

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the computed root properties for each image.
    """
    
    all_results = []

    for fname in file_list:
        # read image
        im = io.imread(fname)
        # process image
        results_df = extract_rh_props(im, fname = fname, **kwargs)
        all_results.append(results_df)

    # concatenate and save in excel-format
    return pd.concat(all_results)
