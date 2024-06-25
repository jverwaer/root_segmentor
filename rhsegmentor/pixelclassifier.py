"""
Module that allows to train a pixel classifier
"""

from sklearn.ensemble import RandomForestClassifier
from skimage import future
import numpy as np
from joblib import dump, load

# alias for dumping model to binary file
dump_model = dump

# alias for loading model from binary file
load_model = load

def train_segmentor(features: np.ndarray, training_labels: np.ndarray) -> RandomForestClassifier:
    """
    Function to train a random forest classifier given a feature image and a label image
    
    PARAMETERS
    ----------
    features : np.ndarray
        (nrow x nkol x k) feature cube where k is the number of features (typically
        generated using the function im2features)
        
    training_labels : np.ndarray
        (nrow x nkol) array where 0 = unlabeled; 1 = root; 2 = background
        
    RETURNS
    -------
    RandomForestClassifier instance (scikit-learn object)
    """
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                 max_depth=10, max_samples=0.05)
    clf = future.fit_segmenter(training_labels, features, clf)
    return clf
    
def predict_segmentor(clf: RandomForestClassifier, 
                      features_test_im: np.ndarray) -> np.ndarray:
    """
    Function to predict on a test image
    
    PARAMETERS
    ----------
    clf : RandomForestClassifier (or any scikit-learn classifier)
        The trained classifier used for prediction.
        
    features_test_im : np.ndarray
        (nrow x nkol x k) array of features representing the test image.
        
    RETURNS
    -------
    np.ndarray
        Array of predicted labels representing the predicted segmentation.
    """
    result = future.predict_segmenter(features_test_im, clf)
    return result
