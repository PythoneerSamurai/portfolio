import streamlit as st

st.title("License Plate Recognition")
st.write("\n")

with st.container(height=400, border=True):
    st.video("assets/object_detection/license_plate_recognition/input_video.mp4", muted=True)

st.write("\n")
st.subheader("Project Overview", divider=True)
st.write("""
In this project I have utilized various computer vision techniques, including but not limited to, object detection,
object tracking, and optical character recognition, for the purpose of vehicle license plate recognition. Note that
object detection is the task of merely localizing all the objects present in an image, whereas object recognition
involves analyzing those objects and recognizing the information stored or represented by them.
""")

st.write("\n")
st.subheader("Training Pipeline", divider=True)
st.write("##### Training Data")
st.write("""
In the case of this project, only one yolov8x model has been utilized, that for the purpose of vehicle license plate
detection.

I couldn't find the dataset that I used for training this model, still qualitative information regarding the dataset
is added below.

The dataset had only one class for detection, that being "license plate".

The dataset was standardized and the images had a size of 640 pixels in width and height. The object detection
labels were in the YOLO format, stored in .txt files sharing the same name as the images they belonged to.

The dataset was re-structured according to the YOLO format, having two folders "images" and "labels", both of which
had sub-directories "train", "test", and "val".
""")

st.write("##### Model Used")
st.write("""
As mentioned above, only one model has been used in this project, that is a **yolov8x** model for vehicle license 
plate detection. The model is the largest variant of its kind (as inferred by the "x" in the name of the model), and 
thus has the highest mAP (Mean Average Precision) value, as benchmarked on the COCO dataset.

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
found on my [Kaggle](https://www.kaggle.com/models/pythonistasamurai/yolov8-license-plate-detection).
""")

st.subheader("Prediction Pipeline", divider=True)

st.info(
    body="**For detailed in-line comment explanation of the prediction pipeline visit my "
         "[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/object-detection/yolov8x-supervision-easyocr-license-plate-recognition).**",
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
import cv2
import easyocr
import supervision as sv
from ultralytics import YOLO
""", language="Python")

st.write("##### Loading the Input Video, Instantiating the Model, and Initializing a VideoWriter Object")
st.write("""
In this part of the prediction pipeline, The trained yolov8x model for vehicle license plate detection is instantiated 
using the ultralytics.YOLO class. Moreover, the input video is loaded in to memory using the cv2.VideoCapture class. 
Afterwards, the width, height, and FPS information of the input video is extracted for later use. In addition to that,
a cv2.VideoWriter object is initialized for the purpose of merging the processed frames into the output video.
""")

st.write("##### Initializing Annotators")
st.write("""
In this part of the prediction pipeline, supervision annotators, for annotating the input video frames with the desired 
information, are initialized.

1) LabelAnnotator is used to annotate the frames with the the license plate numbers of each vehicle, along with the time 
it spends, in the video.
2) BoxCornerAnnotator is used to annotate the frames with cornered boxes around the license plates of each vehicle.
""")

st.write("##### Initializing the Tracker and OCR object")
st.write("""
In this part of the prediction pipeline, a tracker (sv.ByteTrack) is initialized for the purpose of tracking the license
plates throughout the video frames. This is important to keep track of the total time spent by the vehicles in the 
footage. In addition to the tracker, an EasyOCR object is initialized for the purpose of reading the license plate 
numbers from the detected license plates.
""")

st.write("##### Initializing dictionaries for data handling and defining the time estimator function")
st.write("""
In this part of the training pipeline, three dictionary objects are initialized for the purpose of data handling.

In addition to that, a time estimator function is defined for the purpose of estimating the time each vehicle spends in 
the footage. This function uses the previously stored entry frame of each vehicle in the footage, and subsequent frames
(up until the exit frame of the vehicle), for the purpose of time estimation.
""")

st.write("##### Defining the Frame Processor Function")
st.write("""
In this part of the prediction pipeline, the main frame processing function is defined. This function deals with all
frame annotations. All code ranging from tracking to the appending of the annotated frame to the output video is
written in this function. The code for reformating the license plate numbers recognized by the OCR object (according to
the standard license plate configuration of the respective country of this video), is also written in this function.
""")

st.write("##### Main Loop")
st.write("""
Lastly, the main loop that breaks the input video into frames and calls relevant functions is started. This loop
gets the model's inference on the frames and then calls the frameProcessor() function for inference handling and 
frame processing.
""")

st.write("#### Code Link")
st.write("""
The code for this project can be found on my
[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/object-detection/yolov8x-supervision-easyocr-license-plate-recognition).
""")
