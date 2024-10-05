import streamlit as st

st.title("Background Editor")
st.write("\n")

with st.container(height=400, border=True):
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.image("assets/segmentation/background_editor/real.jpeg", caption="Real Image")
    with col2:
        st.image("assets/segmentation/background_editor/background_edited.jpeg", caption="Backgroud Edited")

st.write("\n")
st.subheader("Project Overview", divider=True)
st.write("""
In this project, I have utilized semantic segmentation for image or video background editing. I have
developed a GUI-based application using which the backgrounds behind people in images or videos can be removed or
replaced by other background images.
""")

st.write("\n")
st.subheader("Training Pipeline", divider=True)
st.write("##### Training Data")
st.write("""
In the case of this project, only one yolov9e model has been utilized, for person segmentation.

Multiple datasets were used to train the model. I had placed a lot of effort in finding the best
freely available person segmentation datasets that had almost pixel-perfect annotations. The datasets used for training
are listed below.

1. [Human Segmentation MADS Dataset](https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-mads-dataset)
2. [Human Segmentation Dataset - Supervise.ly](https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset)
3. [Human Segmentation Dataset - TikTok Dances](https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-tiktok-dancing-dataset)

The datasets were preprocessed and the images had a size of 640 pixels in width and height. The datasets did not have
YOLO format .txt segmentation labels, rather they came with binary masks, therefore I used an open-source script to
convert those binary masks into YOLO .txt files, having the same name as the images they belonged to.

The dataset was re-structured according to the YOLO format, having two folders "images" and "labels", both of which
had sub-directories "train", "test", and "val".
""")

st.write("##### Model Used")
st.write("""
As mentioned above, only one model has been used in this project: a **yolov9e** model for person segmentation. 
The model is the largest variant of its kind (as inferred by the "e" in the name of the model), and thus has the 
highest mAP (Mean Average Precision) value, as benchmarked on the COCO dataset.

In the case of this project, a pre-trained model (yolov9e.pt) was not used, rather a new model (yolov9e.yaml) was trained 
from scratch (this resulted in better precision and recall).

The model was accessed and trained using the Ultralytics YOLO API, as provided in the **"ultralytics"** Python
package.

For more information regarding the YOLO models, refer to the
[Ultralytics YOLO docs](https://docs.ultralytics.com/)
""")

st.write("##### Training Analysis")
st.write("""
The model was trained using the ".train" function in the ultralytics YOLO class.

The maximum amount of epochs that the model could train for was set to 800, with the patience of 80 epochs
(patience is a hyper-parameter that defines how many epochs will pass before EarlyStopping stops the training due to no
improvement). The model was trained using Kaggle cloud computing; thus, two Nvidia Tesla T4 GPUs were used for
training the model. The optimizer was set to "auto", due to which "Stochastic Gradient Descent (aka SGD)" was used as 
the optimizer. Default values were used for all other hyper-parameters.

The model was trained for several hours before EarlyStopping terminated the training.
""")

st.write("##### Model Links")
st.write("""
The trained model for this project can be 
found on my [Kaggle](https://www.kaggle.com/models/pythonistasamurai/yolov9-semantic-segmentation-background-editor).
""")

st.subheader("Prediction Pipeline", divider=True)

st.write("""
The prediction pipeline of this project consists of several steps as described below.
""")

st.write("##### Specifying Imports")
st.write("""
A total of five imports are specified at the beginning of the prediction pipeline. Those being:
""")
st.code("""
import os
from tkinter import filedialog as fd

import customtkinter as ctk
import cv2
from ultralytics import YOLO
""", language="Python")

st.write("##### Specifying Various Paths, Parameters, and Instantiating the Model")
st.write("""
In this part of the prediction pipeline, various variables are declared to store the paths to the
source images/videos, the output directory, and the custom background image (if needed). All the variables are
initially None type, later when the user selects files through the GUI, paths to the selected files are stored in these
variables.

In addition to the paths, the confidence score of the model is also specified, that to be 0.5. Later the user can
change this confidence score using a slider in the GUI. Moreover, all acceptable image and video extensions are 
specified in the form of lists, this is to filter the files in the file selection dialog of the GUI.

Afterward, the trained yolov9e model for person detection is instantiated using the ultralytics.YOLO class, accessed
through the "**ultralytics**" Python package.
""")

st.write("##### Defining Functions for Background Editing")
st.write("""
In this part of the prediction pipeline, two functions are defined for background editing, one for images
and the other for videos. All code ranging from model inference handling to numpy bitwise operations is written in these
functions. These functions are later attached to a button placed on the GUI of the application.
""")

st.write("##### Designing the GUI")
st.write("""
In the last part of the prediction pipeline, the GUI is coded. In the case of this project, a Python library known as
"**customtkinter**" is used for the development of the GUI. This project features a simple and minimalistic GUI,
containing a few buttons and a slider (for model confidence threshold variation).
""")

st.write("##### Code Link")
st.write("""
The code for this project can be found on my
[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/segmentation/yolov9e-semantic-segmentation-background-editor).
""")
