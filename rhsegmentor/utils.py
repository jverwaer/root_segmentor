import numpy as np
import os


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

def get_save_fname(fname, save_dir, suffix = ".dummy"):
    """
    Function to split the absolute path string into directory and file name and append a suffix.
    If save_dir is None, the directory will be joined with the file name where the extension (after the last .)
    is replaced with `suffix`. Else, the directory of the file name will be replace with `save_dir`.
    
    PARAMETERS
    ---------
    fname : str
        Absolute path string of the file
    save_dir : str
        Directory where the file will be saved
    suffix : str
        Suffix to be added to the file name
        
    RETURNS
    -------
    str
        Absolute path string of the saved file
    """
    directory, filename = os.path.split(fname)
    if save_dir is None:
        return os.path.join(directory, f"{filename.split('.')[0]}_{suffix}")
    return os.path.join(save_dir, f"{filename.split('.')[0]}_{suffix}")

def listdir_with_path(path, suffix=""):
    """
    Returns a list of extended file paths in the given directory that end with the specified suffix. The `path` will be prepended
    to the file names (so when `path` is absolute, the result will be an absolute path).

    PARAMETERS
    ----------
        path (str): The directory path.
        endswith (str, optional): The suffix to filter the file names. Defaults to "".

    Returns
    -------
        list: A list of file paths.

    """
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(suffix)]
  


