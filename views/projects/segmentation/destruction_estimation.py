import streamlit as st

st.title("Destruction Estimation")
st.write("\n")

with st.container(height=380, border=True):
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.image(
            image="assets/segmentation/destruction_estimation/before_destruction_segmentation.png",
            caption="Before Destruction Segmentation"
        )
    with col2:
        st.image(
            image="assets/segmentation/destruction_estimation/after_destruction_segmentation.png",
            caption="After Destruction Segmentation"
        )

st.write("\n")
st.subheader("Project Overview", divider=True)
st.write("""
In this project, I have utilized various computer vision techniques, including but not limited to, semantic segmentation
and SAHI (slicing-aided-hyper-inference), to estimate the destruction caused to construction in an
area. I have estimated the amount of buildings lost due to destruction with a high accuracy.
""")

st.write("\n")
st.subheader("Training Pipeline", divider=True)
st.write("##### Training Data")
st.write("""
In the case of this project, only one yolov8x model has been utilized, for building segmentation.

The dataset used to train the buildings segmentation model is the
[Massachusetts Buildings Dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset)
provided by "**Balraj Ashwath**" on Kaggle.

The dataset has only one class for segmentation, that is "buildings".

The datasets were preprocessed and the images had a size of 640 pixels in width and height. The datasets did not have
YOLO format .txt segmentation labels, rather they came with binary masks, therefore I used an open-source script to
convert those binary masks into YOLO .txt files, having the same name as the images they belonged to.

The dataset was re-structured according to the YOLO format, having two folders "images" and "labels", both of which
had sub-directories "train", "test", and "val".
""")

st.write("##### Model Used")
st.write("""
As mentioned above, only one model has been used in this project, that is a **yolov8x** model for buildings 
segmentation. The model is the largest variant of its kind (as inferred by the "x" in the name of the model), and thus 
has the highest mAP (Mean Average Precision) value, as benchmarked on the COCO dataset.

In the case of this project, a pre-trained model (yolov8x.pt) was not used, rather a new model (yolov8x.yaml) was trained 
from scratch (this resulted in better precision and recall).

The model was accessed and trained using the Ultralytics YOLO API, as provided in the **"ultralytics"** Python
package.

For more information regarding the YOLO models, refer to the
[Ultralytics YOLO docs](https://docs.ultralytics.com/)
""")

st.write("##### Training Analysis")
st.write("""
The model was trained using the ".train" function in the ultralytics YOLO class.

The maximum amount of epochs that the model could train for was set to 1000, with the patience of 80 epochs
(patience is a hyper-parameter that defines how many epochs will pass before EarlyStopping stops the training due to no
improvement). The model was trained using Kaggle cloud computing; thus, two Nvidia Tesla T4 GPUs were used for
training the model. The optimizer was set to "auto", due to which "Stochastic Gradient Descent (aka SGD)" was used as 
the optimizer. Default values were used for all other hyper-parameters.

The model was trained for several hours before EarlyStopping terminated the training.
""")

st.write("##### Model Links")
st.write("""
The trained model for this project can be 
found on my [Kaggle](https://www.kaggle.com/models/pythonistasamurai/yolov8x-sahi-construction-damage-estimation).
""")

st.subheader("Prediction Pipeline", divider=True)

st.info(
    body="**For detailed in-line comment explanation of the prediction pipeline visit my "
         "[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/segmentation/yolov8x-SAHI-construction-damage-estimation).**",
    icon="ℹ️"
)

st.write("""
The prediction pipeline of this project consists of several steps as described below.
""")

st.write("##### Specifying Imports")
st.write("""
A total of three imports are specified at the beginning of the prediction pipeline. Those being:
""")
st.code("""
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
""", language="Python")

st.write("##### Specifying Parameters and Paths")
st.write("""
In this part of the prediction pipeline, four parameters are specified for later use, those being the minimum confidence
threshold for registering the model's segmentations, the device to be used for carrying out model inference, and the image
slice width and height to be used for SAHI.

In addition to that, various paths are specified for the proper functioning of the code, those being the paths to images of
an area before and after destruction, and the paths to the export directories of the segmented images.
""")

st.write("##### Instantiating the SAHI AutoDetectionModel")
st.write("""
In this part of the prediction pipeline, the SAHI AutoDetectionModel is instantiated using the trained yolov8x model.
In the case of this project, inferences are not simply carried out using the plain yolov8x model, rather than that
SAHI AutoDetectionModel is used, which is later used to carry out sliced inferences. This is done to increase the total
number of segmentations because the images contain very small buildings, slicing the image makes it easier for the model
to segment these buildings.
""")

st.write("##### Code for Destruction Estimation")
st.write("""
Lastly in the prediction pipeline, the entire code for destruction estimation is written, ranging from inference
handling to estimating the amount of buildings lost. Various analytics regarding destruction are produced, such as
the number of buildings lost, the ratio of occupancy before and after destruction, and the total area covered by the
buildings before and after destruction.
""")

st.write("#### Code Link")
st.write("""
The code for this project can be found on my
[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/segmentation/yolov8x-SAHI-construction-damage-estimation).
""")
