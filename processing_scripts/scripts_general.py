# %% setup
#############################################################################

import os
import sys
if os.path.basename(os.getcwd()) == "processing_scripts":
    sys.path.append("..\\..\\root_segmentor")

from root_segmentor_VIB import *



# %% TEST CASE 1: analyzing a single image 
#############################################################################

# load image
im, names, vertices_s, vertices_e = load_training_image(img_file = "../sample_data/EOS 550D_046.JPG",
                                                        root_traces_file = "../sample_data/EOS 550D_046 vertices.csv")

# transform coordinates
src = [[2683, 75],
    [2472, 82],
    [2682, 3373],
    [2536, 3370]]

dst = [[83, 2501],
        [86, 2715],
        [3375, 2493],
        [3375, 2648]]

vertices_s = transform_coordinates(src, dst, vertices_s)
vertices_e = transform_coordinates(src, dst, vertices_e)
vertices_s_RC = flip_XY_RC(vertices_s)
vertices_e_RC = flip_XY_RC(vertices_e)

# create root mask
root_mask = create_root_mask(im, vertices_s_RC, vertices_e_RC)
root_buffer_background = create_root_buffer_background_image(root_mask)

# create training data labels
training_labels = create_training_data(root_buffer_background)

# show results
plt.subplot(1, 3, 1)
show_traces(vertices_s, vertices_e, im)
plt.subplot(1, 3, 2)
show_traces(vertices_s, vertices_e, root_buffer_background)
plt.subplot(1, 3, 3)
show_traces(vertices_s, vertices_e, training_labels)

# compute features
features = im2features(im)

# train model
mdl = train_segmentor(features, training_labels)

# use model to make predictions
predicted_segmentation = predict_segmentor(mdl, features)

# show result
show_predicted_segmentation(im, predicted_segmentation)
show_traces(vertices_s, vertices_e)

# %% TEST CASE 2: training on multiple images
#############################################################################

# compute FEATURES and LABELS for each image in a given folder
files_list = ['../sample_uniform_1/'+f for f in os.listdir('../sample_uniform_1') if f[-3:] == "JPG"]
imgs_to_XY_data(files_list)


# create training datasets
features_file_list = ['../sample_uniform_1/'+f for f in os.listdir('../sample_uniform_1') if f[-3:] == "npy" and "FEATURES" in f]
X, Y = compile_training_dataset_from_precomputed_features(features_file_list)

# fit random forest classifier
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                            max_depth=10, max_samples=0.05)
clf.fit(X, Y)

# load a random forest classifier
# clf = load('RF_550D_042_078_NTrees-100.joblib')

# dump the model to a file
# dump(clf, 'RF_550D_042_078_NTrees-100.joblib')


# use model to make predictions
im, names, vertices_s, vertices_e = load_training_image(img_file = "../sample_uniform_1/EOS 550D_042.JPG",
                                                        root_traces_file = "../sample_uniform_1/EOS 550D_042 vertices.csv",
                                                        auto_transform=True)
features = im2features(im)
predicted_segmentation = predict_segmentor(clf, features)

show_predicted_segmentation(im, predicted_segmentation)
show_traces(vertices_s, vertices_e)

# %% TEST CASE 2 - BIS: clean up and show results
#############################################################################

im, names, vertices_s, vertices_e = load_training_image(img_file = "../sample_uniform_1/EOS 550D_042.JPG",
                                                        root_traces_file = "../sample_uniform_1/EOS 550D_042 vertices.csv",
                                                        auto_transform=True)
# load prediction for "550D_042"
predicted_segmentation = np.load("pred_550D_042_nTrees-500.npy")
# clean detected roots
roots = clean_predicted_roots(predicted_segmentation)
# draw detected roots
draw_detected_roots(roots, im)

# %% predict using for new image

im, names, vertices_s, vertices_e = load_training_image(img_file = "../sample_uniform_1/test_cases/EOS 550D_084.JPG",
                                                        root_traces_file = "../sample_uniform_1/test_cases/EOS 550D_084 vertices.csv",
                                                        auto_transform=True)
# load model
mdl = load("RF_550D_042_078_NTrees-100.joblib")
# compute features
features = im2features(im)
# predict
predicted_segmentation = predict_segmentor(mdl, features)
# clean detected roots
roots = clean_predicted_roots(predicted_segmentation)
# draw detected roots
draw_detected_roots(roots, im)



# %% TESTS TO IMPORT DATA FROM NEW CAMERA
#############################################################################

# load image
im, names, vertices_s, vertices_e = load_training_image(img_file = "../sample_newMachine/AG00IHWX_000000.jpg",
                                                        root_traces_file = "../sample_newMachine/AG00IHWX_000000 vertices.csv")

# transform coordinates


vertices_s_RC = flip_XY_RC(vertices_s)
vertices_e_RC = flip_XY_RC(vertices_e)

# create root mask
root_mask = create_root_mask(im, vertices_s_RC, vertices_e_RC, dilatation_radius = 2)
root_buffer_background = create_root_buffer_background_image(root_mask, buffer_radius = 5, no_root_radius = 30)

# create training data labels
training_labels = create_training_data(root_buffer_background)


# show results
plt.subplot(1, 3, 1)
show_traces(vertices_s, vertices_e, im)
plt.subplot(1, 3, 2)
show_traces(vertices_s, vertices_e, root_buffer_background)
plt.subplot(1, 3, 3)
show_traces(vertices_s, vertices_e, training_labels)

# compute features
features = im2features(im, sigma_max = 10)

# train model
mdl = train_segmentor(features, training_labels)

# use model to make predictions
predicted_segmentation = predict_segmentor(mdl, features)

# show result
show_predicted_segmentation(im, predicted_segmentation)
show_traces(vertices_s, vertices_e)

# clean detected roots
roots = clean_predicted_roots(predicted_segmentation, small_objects_threshold=150, closing_diameter = 4)
# draw detected roots
draw_detected_roots(roots, im)


# %% TESTS TO IMPORT DATA FROM NEW CAMERA -> train on multiple images
#############################################################################

# compute FEATURES and LABELS for each image in a given folder
files_list = ['../sample_newMachine/'+f for f in os.listdir('../sample_newMachine') if f[-3:] == "jpg"]

imgs_to_XY_data(img_file_list = files_list,
                    root_traces_file_list = None,
                    auto_transform = False,
                    dilatation_radius = 2,
                    buffer_radius = 5,
                    no_root_radius = 30,
                    sigma_max = 10)


# create training datasets
features_file_list = ['../sample_newMachine/'+f for f in os.listdir('../sample_newMachine') if f[-3:] == "npy" and "FEATURES" in f]
X, Y = compile_training_dataset_from_precomputed_features(features_file_list,
                                                          sample_fraction = (1.0, 1.0))

# fit random forest classifier
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                            max_depth=10, max_samples=0.05)
clf.fit(X, Y)

# load a random forest classifier
# clf = load('RF_550D_042_078_NTrees-100.joblib')

# dump the model to a file
# dump(clf, 'RF_550D_042_078_NTrees-100.joblib')


# use model to make predictions
im, names, vertices_s, vertices_e = load_training_image(img_file = "../sample_newMachine/AG00DA6Z_000002.jpg",
                                                        root_traces_file = "../sample_newMachine/AG00DA6Z_000002 vertices.csv",
                                                        auto_transform=False)
features = im2features(im, sigma_max = 10)
predicted_segmentation = predict_segmentor(clf, features)

show_predicted_segmentation(im, predicted_segmentation)
show_traces(vertices_s, vertices_e)

# clean detected roots
roots = clean_predicted_roots(predicted_segmentation, small_objects_threshold=150, closing_diameter = 4)
# draw detected roots
draw_detected_roots(roots, im)