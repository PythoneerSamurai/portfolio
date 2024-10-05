import streamlit as st

st.title("Road Area Estimation")
st.write("\n")

with st.container(height=410, border=True):
    st.video(data="assets/segmentation/road_area_estimation/input_video.mp4", muted=True)

st.write("\n")
st.subheader("Project Overview", divider=True)
st.write("""
In this project, I have utilized semantic segmentation to estimate the total area occupied by roads in
an image or video.
""")

st.write("\n")
st.subheader("Training Pipeline", divider=True)
st.write("##### Training Data")
st.write("""
In the case of this project, only one yolov9e model has been utilized, that for road segmentation.

The dataset used to train the model is the
[DeepGlobe Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)
provided by "**Balraj Ashwath**" on Kaggle.

The dataset has only one class for segmentation, that is "road".

The datasets were preprocessed and the images had a size of 640 pixels in width and height. The datasets did not have
YOLO format .txt segmentation labels, rather they came with binary masks, therefore I used an open-source script to
convert those binary masks into YOLO .txt files, having the same name as the images they belonged to.

The dataset was re-structured according to the YOLO format, having two folders "images" and "labels", both of which
had sub-directories "train", "test", and "val".
""")

st.write("##### Model Used")
st.write("""
As mentioned above, only one model has been used in this project: a **yolov9e** model for road segmentation. 
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

The maximum amount of epochs that the model could train for was set to 600, with the patience of 50 epochs
(patience is a hyper-parameter that defines how many epochs will pass before EarlyStopping stops the training due to no
improvement). The model was trained using Kaggle cloud computing; thus, two Nvidia Tesla T4 GPUs were used for
training the model. The optimizer was set to "auto", due to which "Stochastic Gradient Descent (aka SGD)" was used as 
the optimizer. Default values were used for all other hyper-parameters.

The model was trained for several hours before EarlyStopping terminated the training.
""")

st.write("##### Model Links")
st.write("""
The trained model for this project can be 
found on my [Kaggle](https://www.kaggle.com/models/pythonistasamurai/yolov9e-deepglobe-road-segmentation).
""")

st.subheader("Prediction Pipeline", divider=True)

st.info(
    body="**For detailed in-line comment explanation of the prediction pipeline visit my "
         "[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/segmentation/yolov9e-supervision-deepglobe-road-area-estimation).**",
    icon="ℹ️"
)

st.write("""
The prediction pipeline of this project consists of several steps as described below.
""")

st.write("##### Specifying Imports")
st.write("""
A total of four imports are specified at the beginning of the prediction pipeline. Those being:
""")
st.code("""
from ultralytics import YOLO
import supervision as sv
import cv2
import os
""", language="Python")

st.write("##### Initializing Parameters")
st.write("""
In this part of the prediction pipeline, various parameters are initialized to annotate the input
images with textual information representing road area analysis.
""")

st.write("##### Initializing a cv2.VideoWriter Object")
st.write("""
In this part of the prediction pipeline, a cv2.VideoWriter object is initialized, along with some required parameters,
to merge the processed input images into the output video.
""")

st.write("##### Instantiating the Model and Initializing an Annotator")
st.write("""
In this part of the prediction pipeline, the trained yolov9e model for road segmentation is initialized into memory, for
the purpose of carrying inference on the input images.

Moreover, a supervision MaskAnnotator object is initialized to annotate the input images with the road
segmentations inferred from the model.
""")

st.write("##### Getting Model's Inference and Carrying out Annotations")
st.write("""
Lastly in the prediction pipeline, the trained model's inference is carried out on the input images. In addition to
that, the input images are annotated with their respective segmentations and the textual information representing
road area analysis. The processed images are finally merged into the output video.
""")

st.write("#### Code Link")
st.write("""
The code for this project can be found on my
[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/segmentation/yolov9e-supervision-deepglobe-road-area-estimation).
""")
