#########################################
# Define the input directories and files#
#########################################

# gives your new model a name (should end with joblib)

newmodel = "../models/newmodel.joblib"

# specify the location of the training data

training_data = "../training_data/"

# specify the number of estimators (the more, the slower the model. default 500)

Nest = 500


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

#Compile a dataset for training
#The tracings of multiple images are combined to learn a pixel-classifier. To achieve this goal, the following steps are taken:

#All images and tracings in some_folder are listed
#Per image, pixel-level features are computed (texture, gradient image etc.)
#The per image, the function create_root_buffer_background_image is used to comput the label of every pixel
#A fraction is points is sampled (reducing training dataset size and rebalancing it somewhat)
#The preveous steps are applied to all images in some_folder and combined in a features dataset X and labels dataset Y
#The first step only computes labels and features per image and stores them as npy files.

# compute FEATURES and LABELS for each image in a given folder
files_list = [(training_data)+f for f in os.listdir(training_data) if f[-3:] == "JPG"]
rs.imgs_to_XY_data(img_file_list = files_list,
                    root_traces_file_list = None,
                    auto_transform = False,
                    dilatation_radius = 2,
                    buffer_radius = 5,
                    no_root_radius = 30,
                    sigma_max = 10)

# The second step combines the generated files to create X and Y
# create training datasets
features_file_list = [(training_data)+f for f in os.listdir(training_data) if f[-3:] == "npy" and "FEATURES" in f]
X, Y = rs.compile_training_dataset_from_precomputed_features(features_file_list, sample_fraction=(1.0, 1.0))

# fit random forest classifier
clf = rs.RandomForestClassifier(n_estimators=500, n_jobs=-1,
                            max_depth=10, max_samples=0.05)
clf.fit(X, Y)
# dump the model to a file
rs.dump(clf, newmodel)
