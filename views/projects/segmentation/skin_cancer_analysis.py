import streamlit as st

st.title("Skin Cancer Analysis")
st.write("\n")

with st.container(height=410, border=True):
    st.video("assets/segmentation/skin_cancer_analysis/input_video.mp4", muted=True)

st.write("\n")
st.subheader("Project Overview", divider=True)
st.write("""
In this project I have utilized semantic segmentation for the purpose of skin cancer analysis. This project shows the
potential that computer vision holds for medicinal purposes. Using the model trained in this project, and computer
vision in general, patients can get their diseases diagnosed in the comfort of their homes.
""")

st.write("\n")
st.subheader("Training Pipeline", divider=True)
st.write("##### Training Data")
st.write("""
In the case of this project, only one yolov8x model has been utilized, that for the purpose of skin cancer segmentation.

The dataset used to train the model is the
[Skin cancer: HAM10000](https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification) 
dataset provided by "**Suraj Ghuwalewala**" on Kaggle.

The dataset has only one class for segmentation, that is "cancer".

The datasets were preprocessed and the images had a size of 640 pixels in width and height. The datasets did not have
YOLO format .txt segmentation labels, rather they came with binary masks, therefore I used an open-source script to
convert those binary masks into YOLO .txt files, having the same name as the images they belonged to.

The dataset was re-structured according to the YOLO format, having two folders "images" and "labels", both of which
had sub-directories "train", "test", and "val".
""")

st.write("##### Model Used")
st.write("""
As mentioned above, only one model has been used in this project, that is a **yolov8x** model for skin cancer
segmentation. The model is the largest variant of its kind (as inferred by the "x" in the name of the model), and thus 
has the highest mAP (Mean Average Precision) value, as benchmarked on the COCO dataset.

In the case of this project, pretrained model (yolov8x.pt) was not used, rather a new model (yolov8x.yaml) was trained 
from scratch (this resulted in better precision and recall).

The model was accessed and trained using the Ultralytics YOLO API, as provided in the **"ultralytics"** Python
package.

For more information regarding the YOLO models, refer to the
[Ultralytics YOLO docs](https://docs.ultralytics.com/)
""")

st.write("##### Training Analysis")
st.write("""
The model was trained using the ".train" function present in the ultralytics YOLO class.

The maximum amount of epochs that the model could train for was set to 600, with a patience of 50 epochs
(patience is a hyper-parameter that defines how many epochs will pass before EarlyStopping stops the training due to no
improvement). The model was trained using Kaggle cloud computing and thus two Nvidia Tesla T4 GPUs were used for
training the model. The optimizer was set to "auto", due to which "Stochastic Gradient Descent (aka SGD)" was used as 
the optimizer. Default values were used for all other hyper-parameters.

The model was trained for several hours before EarlyStopping terminated the trainings.
""")

st.write("##### Model Links")
st.write("""
The trained model for this project can be 
found on my [Kaggle](https://www.kaggle.com/models/pythonistasamurai/yolov8x-ham10000-segmentation).
""")

st.subheader("Prediction Pipeline", divider=True)

st.info(
    body="**For detailed in-line comment explanation of the prediction pipeline visit my "
         "[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/segmentation/yolov8x-supervision-skin-cancer-analysis).**",
    icon="ℹ️"
)

st.write("""
The prediction pipeline of this project consists of several steps as described below.
""")

st.write("##### Specifying Imports")
st.write("""
A total of five imports are specified at the beginning of the prediction pipeline. Those being:
""")
st.code("""
from ultralytics import YOLO
import supervision as sv
import cv2
import os
""", language="Python")

st.write("##### Initializing Parameters")
st.write("""
In this part of the prediction pipeline, various parameters are initialized for the purpose of annotating the input
images with textual information representing skin cancer analysis.
""")

st.write("##### Initializing a cv2.VideoWriter Object")
st.write("""
In this part of the prediction pipeline, a cv2.VideoWriter object is initialized, along with some required parameters,
for the purpose of merging the processed input images into the output video.
""")

st.write("##### Instantiating the Model and Initializing an Annotator")
st.write("""
In this part of the prediction pipeline, the trained yolov8x model for skin caner segmentation is initialized into 
memory, for the purpose of carrying inference on the input images.

Moreover, a supervision MaskAnnotator object is initialized for the purpose of annotating the input images with the skin
cancer segmentations inferred from the model.
""")

st.write("##### Getting Model's Inference and Carrying out Annotations")
st.write("""
Lastly in the prediction pipeline, the trained model's inference is carried out on the input images. In addition to
that, the input images are annotated with their respective segmentations and the textual information representing
skin cancer analysis. The processed images are finally merged into the output video.
""")

st.write("#### Code Link")
st.write("""
The code for this project can be found on my
[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/segmentation/yolov8x-supervision-skin-cancer-analysis).
""")
