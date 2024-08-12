"""
This module provides functions to generate sample training and test data for the root_segmentor package.
"""

import pkg_resources
from os import path
import os
import shutil

training_imgs = ["img1.jpg", "img2.jpg", "img3.jpg"]
training_labels = ["img1 vertices.csv", "img2 vertices.csv", "img3 vertices.csv"]

test_imgs = ["img4.jpg", "img5.jpg"]
test_labels = ["img4 vertices.csv", "img5 vertices.csv"]

def get_image_path(fname):
    # Get the file path to the binary file within the package
    image_path = pkg_resources.resource_filename('rhsegmentor', path.join("data", fname))
    return image_path

def get_label_path(fname):
    # Get the file path to the binary file within the package
    image_path = pkg_resources.resource_filename('rhsegmentor', path.join("data", fname))
    return image_path

def create_training_data(savepath = "./trainData"):
    """
    Creates training data by copying images and labels to a specified directory.

    Args:
        savepath (str, optional): The directory path where the training data will be saved. Defaults to "./trainData".

    Returns:
        None
    """
    try:
        os.mkdir(savepath)
    except:
        print("Appending images to existing directory")
    for fname in training_imgs:
        shutil.copy(get_image_path(fname), path.join(savepath, fname))
    for fname in training_labels:
        shutil.copy(get_label_path(fname), path.join(savepath, fname))

def create_test_data(savepath = "./testData"):
    """
    Creates test data by copying images and labels to a specified directory.

    Parameters:
    - savepath (str): The directory path where the test data will be saved. Default is "./testData".

    Returns:
    - None

    Raises:
    - None

    """
    try:
        os.mkdir(savepath)
    except:
        print("Appending images to existing directory")
    for fname in test_imgs:
        shutil.copy(get_image_path(fname), path.join(savepath, fname))
    for fname in test_labels:
        shutil.copy(get_label_path(fname), path.join(savepath, fname))


    
