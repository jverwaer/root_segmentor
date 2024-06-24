"""
This module implements functions for loading image data and annotations from various file types.

Functions:
- load_training_image: Load a root training image as a numpy RGB array and its adjoining root traces.
- flip_XY_RC: Transform XY coordinates to ROW-COL coordinates or vice versa.
- transform_coordinates: Transform coordinates based on an affine transform derived from source and destination points.
"""

import numpy as np
import pandas as pd
from skimage import io, transform


# defaults for transforming image coordinates (only for backward compatebility)
src = [[2683, 75],[2472, 82],[2682, 3373],[2536, 3370]]
dst = [[83, 2501],[86, 2715],[3375, 2493],[3375, 2648]]
SRC_DST = (src, dst)


def load_training_image(img_file: str = "EOS 550D_046.JPG",
                        root_traces_file: str = "EOS 550D_046 vertices.csv",
                        auto_transform: bool = False,
                        src_dst: tuple[list] = SRC_DST) -> tuple:
    """
    Load a root training image as a numpy RGB array and its adjoining root traces.

    Parameters:
    - img_file (str): File name of the RGB training image.
    - root_traces_file (str): File containing root traces (a csv file).
    - auto_transform (bool): Flag to automatically transform coordinates (default: False).
    - src_dst (tuple[list]): Tuple containing source and destination coordinates for transformation. 
        Example: src_dst[0] = [[2683, 75],[2472, 82],[2682, 3373],[2536, 3370]]
                 src_dst[1] = [[83, 2501],[86, 2715],[3375, 2493],[3375, 2648]]
        where src_dst[0] contains the coördinates of several points in the original coordinate system and 
        src_dst[1] contains the coördinates of these points in the new coördinate system.

    Returns:
    - im (np.array): Training image as a numpy RGB array.
    - names (np.array): Array of names of the roots.
    - vertices_s (np.array): Starting points of vertices as a (n x 2) numpy array of floats.
    - vertices_e (np.array): Ending points of vertices as a (n x 2) numpy array of floats.
    """
    im = io.imread(img_file)
    vertices = np.loadtxt(root_traces_file, skiprows=1, usecols=(2, 3), delimiter=";")
    names = np.loadtxt(root_traces_file, skiprows=1, usecols=(1), delimiter=";", dtype=str)

    if len(names) == 2 * len(np.unique(names)):
        names = names[0::2]
        vertices_s = vertices[0::2, :]
        vertices_e = vertices[1::2, :]
    else:
        vertices_s = []
        vertices_e = []
        names_list = []
        for name in np.unique(names):
            v = vertices[names == name, :]
            reps = np.tile(v[1:-1, :], 2).reshape((-1, 2))
            v = np.concatenate((v[np.newaxis, 0, :], reps, v[np.newaxis, -1, :]), axis=0)
            vertices_s.append(v[0::2, :])
            vertices_e.append(v[1::2, :])
            names_list.append(np.repeat(name, v.shape[0] - 1))
        vertices_s = np.concatenate(vertices_s)
        vertices_e = np.concatenate(vertices_e)
        names = np.concatenate(names_list)

    if auto_transform:
        vertices_s = transform_coordinates(src_dst[0], src_dst[1], vertices_s)
        vertices_e = transform_coordinates(src_dst[0], src_dst[1], vertices_e)

    return im, names, vertices_s, vertices_e


def flip_XY_RC(XY_or_RC: np.ndarray) -> np.ndarray:
    """
    Transform XY coordinates to ROW-COL coordinates or vice versa.

    Parameters:
    - XY_or_RC (np.ndarray): Array of coordinates (XY or ROW-COL).

    Returns:
    - np.ndarray: Array of coordinates (XY or ROW-COL).
    """
    return np.array(np.fliplr(XY_or_RC), dtype=int)


def transform_coordinates(src, dst, vertices):
    """
    Transform the coordinates in vertices based on an affine transform derived from source and destination points.

    Parameters:
    - src (np.ndarray-like): Coordinates in the original coordinate system.
    - dst (np.ndarray-like): Coordinates in the new coordinate system.
    - vertices (np.ndarray): Vertices to be transformed.

    Returns:
    - transformed_vertices (np.ndarray): Transformed vertices.
    """
    tform = transform.estimate_transform('affine', np.array(src), np.array(dst))
    transformed_vertices = tform(vertices)
    return transformed_vertices

