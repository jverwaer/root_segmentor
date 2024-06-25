import numpy as np
import os
from joblib import dump, load

# alias for dumping model to binary file
dump_model = dump

# alias for loading model from binary file
load_model = load

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
    Function to split the absolute path string into directory and file name and append a suffix
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
  


