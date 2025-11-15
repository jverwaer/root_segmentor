#########################################
# Define the input directories and files#
#########################################

# choose the model to be used (Keyence_NTrees-500_2.joblib is a model that predict well all keyence images taken after November 2023)

model = "../models/Keyence_NTrees-500_2.joblib"


# select the folder that contain the images to be analyzed

images = "../images/"


# select the name of the output file

output = "../measurements.xlsx"



#####################################################
# do not change anything from here (unless you know what you are doing)

# Import the required modules
import os
import sys
sys.path.append("..")
import root_segmentor_VIB as rs
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend
import matplotlib.pyplot as plt
import pandas as pd


# Load a saved model
clf = rs.load(model)


#run the analysis for all images in a certain folder

# list all test images in folder
features_file_list = [(images)+f for f in os.listdir(images) if f[-3:].lower() == "jpg"]

all_results = []

for fname in features_file_list:
    # read image
    im = rs.io.imread(fname)
    # compute features
    features = rs.im2features(im, sigma_max = 10)
    # predict
    predicted_segmentation = rs.predict_segmentor(clf, features)
    # clean detected roots
    roots = rs.clean_predicted_roots(predicted_segmentation, small_objects_threshold=150, closing_diameter = 4)
    # compute root properties
    results_df = rs.measure_roots(roots, root_thickness = 7, minimalBranchLength = 10)
    results_df["fname"] = fname
    # append to results list
    all_results.append(results_df)
    # save image for quality check
    rs.save_detected_roots_im(roots, im, fname[:-4] + "result.png", root_thickness = 7, minimalBranchLength = 10)


pd.concat(all_results).to_excel(output)
