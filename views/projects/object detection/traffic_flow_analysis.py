import streamlit as st

st.title("Traffic Flow Analysis")
st.write("\n")

with st.container(height=400, border=True):
    st.video("assets/object_detection/traffic_flow_analysis/input_video.mp4", muted=True)

st.write("\n")
st.subheader("Project Overview", divider=True)
st.write("""
In this project, I have utilized various computer vision techniques, including but not limited to, region-based object 
detection, object tracking, and object counting, to carry out traffic flow analysis. I have
successfully traced the path followed by each vehicle on the roundabout to analyze the flow of 
traffic.
""")

st.write("\n")
st.subheader("Training Pipeline", divider=True)
st.write("##### Training Data")
st.write("""
In the case of this project, only one yolov10x model has been utilized, for vehicle detection
based upon aerial images.

The dataset used to train the head detection model is the
[Roundabout Aerial Images for Vehicle Detection](https://www.kaggle.com/datasets/javiersanchezsoriano/roundabout-aerial-images-for-vehicle-detection) 
dataset provided by "**Javier Sanchez-Soriano and Enrique Puertas**" on kaggle.

The dataset had four classes for detection, those being:

1. Car
2. Truck
3. Cycle
4. Bus

The dataset was standardized and the images had a size of 1920 pixels in width and 1080 pixels in height. The images
were preprocessed to the size of 640 pixels in width and height. The object detection labels were in .xml file format
and were converted to YOLO .txt format files sharing the same name as the images they belonged to.

The dataset was re-structured according to the YOLO format, having two folders "images" and "labels", both of which
had sub-directories "train", "test", and "val".
""")

st.write("##### Model Used")
st.write("""
As mentioned above, only one model has been used in this project: a **yolov10x** model for vehicle detection. 
The model is the largest variant of its kind (as inferred by the "x" in the name of the model), and thus has the 
highest mAP (Mean Average Precision) value, as benchmarked on the COCO dataset.

In the case of this project, a pre-trained model (yolov10x.pt) was not used, rather a new model (yolov10x.yaml) was trained 
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
found on my [Kaggle](https://www.kaggle.com/models/pythonistasamurai/yolov10x-supervision-roundabout-traffic-analysis).
""")

st.subheader("Prediction Pipeline", divider=True)

st.info(
    body="**For detailed in-line comment explanation of the prediction pipeline visit my "
         "[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/object-detection/yolov10x-supervision-roundabout-traffic-analysis).**",
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
import numpy as np
import supervision as sv
from ultralytics import YOLO
""", language="Python")

st.write("##### Instantiating the Model and Loading the Input Video")
st.write("""
In this part of the prediction pipeline, The trained yolov10x model for vehicle detection is instantiated using the 
ultralytics.YOLO class. Moreover, the input video is loaded into memory using the cv2.VideoCapture class.
""")

st.write("##### Defining and Initializing Polygon Zones for Filtering Detections")
st.write("""
In this part of the prediction pipeline, the polygon zones highlighting different regions on the roundabout in the
input video are defined. The zones are grouped into two categories, those being "entry polygon zones" highlighting
those regions through which vehicles can enter the roundabout, and "exit polygon zones" highlighting those regions
through which vehicles can exit the roundabout.

Afterwards, sv.PolygonZone objects are initialized for the purpose of converting the numpy nd-array polygons defined
above into supervision format, which will make it easier to filter detections and annotate the polygon zones
on the input video frames.
""")

st.write("##### Initializing Annotators and the Tracker object")
st.write("""
In this part of the prediction pipeline, supervision annotators, for annotating the input video frames with the desired 
information, are initialized.

1) LabelAnnotator is used to annotate the frames with the tracker ids of each vehicle along with its status on the
roundabout (IN for entry and OUT for exit).
2) BoxAnnotator is used to annotate the frames with boxes around each vehicle.
3) PolygonZoneAnnotator is used to annotate the frames with the polygon zones defined and initialized above.
4) TraceAnnotator is used to annotate the frames with lines tracing the paths followed by each vehicle in the video,
this is important for analyzing the flow of traffic.

In addition to the annotators, a tracker (sv.ByteTrack) object is also initialized to track the
vehicles throughout the video. This is important for tracing the path followed by each vehicle for 
traffic flow analysis.
""")

st.write("##### Initializing the VideoWriter object")
st.write("""
In this part of the prediction pipeline, a cv2.VideoWriter object is initialized to merge the
processed frames into the output video.
""")

st.write("##### Defining the Frame Processor Function")
st.write("""
In this part of the prediction pipeline, the main frame processing function is defined. This function deals with all
frame annotations. All code ranging from tracking to the appending of the annotated frame to the output video is
written in this function.
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
[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/object-detection/yolov10x-supervision-roundabout-traffic-analysis).
""")
