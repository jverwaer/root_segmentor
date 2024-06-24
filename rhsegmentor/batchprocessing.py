
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

from . import dataloader as dl
from . import featureextractor as fe


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
            
            # file name for saving
            raise NotImplementedError

            if not os.path.isfile(img_file[:-4] + "FEATURES.npy"):
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
                    np.save(img_file[:-4] + "FEATURES", features_flat)
                    np.save(img_file[:-4] + "LABELS", label_flat)

                    if save_masks_as_im:
                        # save segmentation mask as image
                        pi = Image.fromarray(training_labels,'P')
                        # Put the palette in
                        pi.putpalette(palette)
                        # Display and save
                        pi.save(img_file[:-4] + 'MASK.png')

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

