import streamlit as st

st.title("Vehicle Speed Estimation")
st.write("\n")

with st.container(height=400, border=True):
    st.video("assets/object_detection/vehicle_speed_estimation/input_video.mp4", muted=True)

st.write("\n")
st.subheader("Project Overview", divider=True)
st.write("""
In this project, I have utilized various computer vision techniques, including but not limited to, region-based object 
detection, object tracking, and perspective transformations, to accurately estimate vehicle speeds.
""")

st.write("\n")
st.subheader("Training Pipeline", divider=True)
st.write("##### Training Data")
st.write("""
In the case of this project, only one yolov10x model has been utilized, that for vehicle detection.

I couldn't find the dataset that I used for training this model, still qualitative information regarding the dataset
is added below.

The dataset was standardized and the images had a size of 640 pixels in width and height. The object detection
labels were in the YOLO format, stored in .txt files sharing the same name as the images they belonged to.

The dataset was re-structured according to the YOLO format, having two folders "images" and "labels", both of which
had sub-directories "train", "test", and "val".
""")

st.write("##### Model Used")
st.write("""
As mentioned above, only one model has been used in this project, that is a **yolov10x** model for vehicle detection. 
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
found on my [Kaggle](https://www.kaggle.com/models/pythonistasamurai/yolov10x-supervision-vehicle-speed-estimation).
""")

st.subheader("Prediction Pipeline", divider=True)

st.info(
    body="**For detailed in-line comment explanation of the prediction pipeline visit my "
         "[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/object-detection/yolov10x-supervision-vehicle-speed-estimation).**",
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
import cv2
from collections import defaultdict, deque
import numpy as np
import supervision as sv
from ultralytics import YOLO
""", language="Python")

st.write("##### Instantiating the Model, Loading the Input Video, and Initializing the VideoWriter Object")
st.write("""
In this part of the prediction pipeline, The trained yolov10x model for vehicle detection is instantiated using the 
ultralytics.YOLO class. Moreover, the input video is loaded into memory using the cv2.VideoCapture class. 
Afterwards, the width, height, and FPS information of the input video is extracted for later use. In addition to that,
a cv2.VideoWriter object is initialized to merge the processed frames into the output video.
""")

st.write("##### Perspective Transformations")
st.write("""
In this part of the prediction pipeline, the source and target matrices for perspective transformations are defined.

When trying to estimate an object's speed using a single camera, it becomes very inaccurate if we use the pixel values 
that the object travels to calculate its speed, that is if the camera is placed at the front (not a top-down 
view). Taking the example of a highway, in an image the farther part of the highway is going to be smaller, and the 
pixel values travelled by the objects are going to be less, even though in reality the highway is equally distanced in 
real life. This is why we have to transform the highway from the perspective of a single camera to a top-down view of the
highway, the latter having width and height the same as that of the highway in reality. The implementation is quite simple, 
we just have to define the source matrix (having the polygon coordinates of the section of the highway whose perspective
we want to transform), and the target matrix (having the polygon coordinates of the actual highway section in real 
life), afterwards we can just call the cv2.getPerspectiveTransform() function to transform the source perspective to the
target perspective. Now all that's left is to convert the coordinates travelled by the objects from the source 
perspective to the target perspective, for the calculation of the distance, using the target pixel values. This is 
easily accomplished by calling the cv2.perspectiveTransform() function, passing the source points and the transformed 
matrix (output of cv2.getPerspectiveTranform() function) as arguments. Lastly, we can use these transformed points to 
calculate the distance travelled by the objects. Note that we need to know the actual width and height of the target 
highway section to accomplish this with high accuracy. In this implementation, I wasn't quite sure of the width and 
height, still I read an article and calculated the width by adding the standard lane widths, and shoulder widths. 
However I had to make a raw guess of the height of the section, which I assumed to be 60 meters long, this gave me 
acceptable vehicle speeds.
""")

st.write("##### Defining and Initializing the Polygon Zone for Filtering Detections")
st.write("""
In this part of the prediction pipeline, the polygon zone highlighting the highway section, in the input video, whose 
perspective is to be transformed is defined.

Afterwards, sv.PolygonZone object is initialized for the purpose of converting the numpy nd-array polygon defined
above into supervision format, which will make it easier to filter detections and annotate the polygon zone
on the input video frames.
""")

st.write("##### Initializing Annotators and the Tracker object")
st.write("""
In this part of the prediction pipeline, supervision annotators, for annotating the input video frames with the desired 
information, are initialized.

1) LabelAnnotator is used to annotate the frames with the speeds of each vehicle.
2) BoxAnnotator is used to annotate the frames with boxes around each vehicle.
3) PolygonZoneAnnotator is used to annotate the frames with the polygon zone defined and initialized above.
4) TraceAnnotator is used to annotate the frames with lines tracing the paths followed by each vehicle in the video,
this is important for analyzing the flow of traffic.

In addition to the annotators, a tracker (sv.ByteTrack) object is also initialized to track the
vehicles throughout the video. This is important for the correct estimation of vehicle speeds based on the distance
travelled by vehicles throughout the video.
""")

st.write("##### Defining the Frame Processor Function")
st.write("""
In this part of the prediction pipeline, the main frame processing function is defined. This function deals with all
frame annotations. All code ranging from tracking to the appending of the annotated frame to the output video is
written in this function. The speed estimation code is also written in this function.
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
[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/object-detection/yolov10x-supervision-vehicle-speed-estimation).
""")
