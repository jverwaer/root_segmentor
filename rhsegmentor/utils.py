import numpy as np

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