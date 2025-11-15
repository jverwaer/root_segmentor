# rhsegmentor

rhsegmentor detects and measure root hairs. 
This folder contains everything required to train new models or to analyze a set of images based on a model.

Copy paste the root folder rhsegmentor and all subfolders + files in your own user folder.

The folder "scripts" contain python scripts that you can use to execute training or analysis.

The folder "models" contain models to be used for the analysis. New models can be made as well and are stored in this folder.

The folder "images" should contain all images that you want to analyze. Importantly, make sure that they have uniquely identifiable names, as the filename will be used as sole identification in the output file. 

###############
# 1. ANALYSIS #
###############

-------------------------------------------------------------
a) (only if you want to change the model or other parameters)
-------------------------------------------------------------

go to the "scripts" folder and open analyze.py in an editor.  

define the path and/or the filename of model, images and output. Original input is:

# choose the model to be used

model = "../models/Keyence_NTrees-500_2.joblib"


# select the folder that contain the images to be analyzed

images = "../images/"


# select the name of the output file

output = "../measurements.xlsx" 


-----------------------
b) perform the analysis
-----------------------

copy the images you want to analyze in the "images" folder (or alternatively navigate to the appropriate path in the analyze.py file)

navigate to the "scripts" folder in a terminal and run analyze.py:

python analyze.py


########################
# 2. TRAIN A NEW MODEL #
########################

If 
- you have new images that look slightly different than others used to train a model 
- the root hairs of your treatments look different compared to other treatments
- you are not happy with the detection of root hairs in a set of images
you might want to train a new model.

make per image that you want to include in the training data a csv file that contain Image;Tracing;X [pix];Y [pix] (make sure to use ; as a seperator).
the name of the csv files should be "'filename of image (without jpg)' vertices.csv". 

You can easily do this with NeuronJ.

Put all images and their vertices files in the folder "training_data"

optional: give a name for your new model and specify the path of the training_data and the model by opening train.py in an editor and changing them 

Navigate to the "scripts" folder in a terminal and run train.py:

python train.py

You will find the new model in the "models" folder (unless you changed the path in the train.py file).

Check if you are happy with the model by running the analyze.py script for one or few images. If not, you might want to add more training data. 
Training data on 4 images gives in general good results when the root hairs are clear,
but 6-8 images could improve the results. Make sure to include different phenotypes or root hair appearences in the training data.





