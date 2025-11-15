# rhsegmentor: root hair segmentation

**rhsegmentor** is a Python package simplifies the bulk segmentation and analysis of root hair images.

<center>
<img src="image.png" alt="example" width="500"/>
</center>

## Using rhsegmentor

rhsegmentor is available as a Python package and can be pip-installed. However, users who prefer not to pip-install the package will find most functionalities of the package in `bundle` folder (the Python modules `root_segmentor_VIB.py` and `skeleton_processor.py` in that folder).

- **Option 1: step-by-step analysis.** Users who want to learn about the functionalities of the package and the structure of the analysis pipeline are advised to follow the steps as described in *Installation* and *Tutorial* below.

- **Option 2: all-in-one script.** Users who prefer to use a pre-configured pipeline (without caring too much about the details of the pipeline) can copy-paste the `bundle` folder or `bundle.zip` on their hard drive and use to following files (also see ``bundle/README.txt` for more info):
    - `scripts/train.py`: a Python script to train a custom root hair detector (data and output folders can be set inside the script). Can be run from command line: `cd` to the `bundle/scripts` folder and run the command
    `python train.py` in the console.

    - `scripts/analyze.py`: a Python script to apply the root detection pipeline (in bulk) on a folder of root images. Can be run from command line: `cd` to the `bundle/scripts` folder and run the command
    `python analyze.py` in the console.

    - **NOTE** that, when the package has not been pip-installed, make sure to install all packages listed in `requirements.txt`


## Installation

`rhsegmentor` can be installed directly form the git repo. We recommend to use a dedicated environment (using `venv` or `conda`).

`pip install git+https://github.com/jverwaer/root_segmentor.git`

## Tutorial

This tutorial guides you throuhg the main use cases of the `rhsegmentor` package.

### Step 0: imports

Import the required modules


```python
import os
import sys
sys.path.append("..")

# imports the rhsegmentor (most important functions are available at the top level of the package)
import rhsegmentor as rh
from rhsegmentor import utils
from rhsegmentor import sample_data_generator

# basic imports for visualization, image loading and classification
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from skimage import io

# magic function (ony for interactive useage)
%matplotlib tk

```

### Step 1: look at some sample data

The functions `create_training_data` and `create_test_data` in the code fragment below create training and test data (both images and labels) that will be used in this tutorial.

The method `load_training_image` allows to read an image and the tracings for training as well. The `auto_transform` option allows to automatically transform the tracings coordinates into the coordinate system of the image.


```python
# create train and test folders
sample_data_generator.create_training_data()
sample_data_generator.create_test_data()

# load first image
im, names, vertices_s, vertices_e = rh.load_training_image(img_file = "./trainData/img1.jpg",
                                                        root_traces_file = "./trainData/img1 vertices.csv",
                                                        auto_transform=False)

#transform into row-column coordinates
vertices_s_RC = utils.flip_XY_RC(vertices_s)
vertices_e_RC = utils.flip_XY_RC(vertices_e)
```

To create training data from the loaded images, the tracings are first transformed into a root-segmentation mask with `root_segmentation_mask`. This function create a np.ndarray mask image containing root-pixels (1), relevant background pixels for making a classification (2) and unclassified pixels (3). To do that, buffer zones are used around the images.



```python
# create segmentation mask
mask = rh.root_segmentation_mask(im = im,
                          vertices_s_RC = vertices_s_RC,
                          vertices_e_RC = vertices_e_RC,
                          dilatation_radius= 2,
                          buffer_radius = 5,
                          no_root_radius = 30)
```

The function `show_traces` allows to plot an image with the traincing on top (similar to imshow). Use `%matplotlib tk` for pop-up viewer


```python
plt.subplot(1, 2, 1)
rh.show_traces(vertices_s, vertices_e, im)
plt.subplot(1, 2, 2)
rh.show_traces(vertices_s, vertices_e, mask)
```

### Step 2: Compile a dataset for training

The tracings of multiple images are combined to learn a pixel-classifier. To achieve this goal, the following steps are taken:
* All images and tracings in `./trainData` are listed
* The function `imgs_to_XY_data` performs the following tasks:
    * Per image, pixel-level features are computed (texture, gradient image etc.)
    * Subsequently, per image, the the label of every pixel is computed (using a call to `create_root_buffer_background_image`)
* The function `compile_training_dataset_from_precomputed_features` performs the following tasks:
    * A fraction of  training points is sampled (reducing training dataset size and rebalancing it somewhat)
    * Selected points are and combined in a features dataset `X` and labels dataset `Y`

The first step only computes labels and features per image and stores them as `npy` files.


```python
# compute FEATURES and LABELS for each image in a given folder
files_list = utils.listdir_with_path('./trainData', suffix = ".jpg")
rh.imgs_to_XY_data(img_file_list = files_list,
                    root_traces_file_list = None,
                    auto_transform = False,
                    dilatation_radius = 2,
                    buffer_radius = 5,
                    no_root_radius = 30,
                    sigma_max = 10,
                    save_masks_as_im = True,
                    save_dir = './trainData')
```

The second step combines the generated files to create `X` and `Y`


```python
# create training datasets
features_file_list = utils.listdir_with_path('./trainData', suffix = "FEATURES.npy")
X, Y = rh.compile_training_dataset_from_precomputed_features(features_file_list, sample_fraction=(1.0, 1.0))
```

### Step 3: Train a model and save it

The compiled dataset is used to train a random forest classifier


```python
# fit random forest classifier (any other classifier)
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                            max_depth=10, max_samples=0.05)
clf.fit(X, Y)
# dump the model to a file
os.mkdir("./models")
rh.dump_model(clf, './models/RF_demo.joblib')
```

### Step 4: Load a saved model

Select a saved model and load it


```python
clf = rh.load_model('./models/RF_demo.joblib')
```

### Step 5: Load new image to make predictions (and compare with tracings)


```python
im = io.imread("./testData/img4.jpg")
# compute features
features = rh.im2features(im, sigma_max = 10)
# predict
predicted_segmentation = rh.predict_segmentor(clf, features)
# clean detected roots
roots = rh.clean_predicted_roots(predicted_segmentation, small_objects_threshold=150, closing_diameter = 4)
```

Visualize the results


```python
# draw detected roots
im_out = rh.draw_detected_roots(roots, im, root_thickness = 7, minimalBranchLength = 10)
# measure root properties and show as table
rh.measure_roots(roots, root_thickness = 7, minimalBranchLength = 10)
```

### Step 6: Export the results to a file

The lenths, orientation, position etc. of the roots can be exported to a file


```python
results_df = rh.measure_roots(roots)
results_df.to_excel("./measurements.xlsx")
```

### Step 7: Automate classification per folder

List all files in `./testData`, detect roots and save the results in a xlsx file. All detected roots are saved for quality checking (in `save_dir`).


```python
#list all .jpg files in ./testData
img_list = utils.listdir_with_path('./testData', suffix = ".jpg")
# batch processs all test images
save_dir = "./testData"
result_df = rh.batch_extract_rh_props(file_list=img_list,
                                      clf = clf,
                                      save_dir=save_dir)
# save final result in xlsx format
result_df.to_excel("measurements_all.xlsx")

```
