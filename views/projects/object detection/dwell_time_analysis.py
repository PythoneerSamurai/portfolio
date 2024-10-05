import streamlit as st

st.title("Dwell Time Analysis")
st.write("\n")

with st.container(height=400, border=True):
    st.video("assets/object_detection/dwell_time_analysis/input_video.mp4", muted=True)

st.write("\n")
st.subheader("Project Overview", divider=True)
st.write("""
In this project, I have utilized region-based object detection and object tracking to accurately estimate the amount
of time each customer dwells in cashier counters. This project is useful for keeping a check on the efficiency
of cashiers, depending upon the time they spend on processing each customer's orders.
""")

st.write("\n")
st.subheader("Training Pipeline", divider=True)
st.write("##### Training Data")
st.write("""
In the case of this project, only one yolov10x model has been utilized, that for customer head detection.

The dataset used to train the head detection model is the
[SCUT-HEAD](https://www.kaggle.com/datasets/hoangxuanviet/scut-head) dataset
provided by "**Hoàng Xuân Việt**" on kaggle.

The dataset had only one class for detection, that being "head".

The dataset was standardized and the images had a size of 640 pixels in width and height. The object detection
labels were in the YOLO format, stored in .txt files sharing the same name as the images they belonged to.

The dataset was re-structured according to the YOLO format, having two folders "images" and "labels", both of which
had sub-directories "train", "test", and "val".
""")

st.write("##### Model Used")
st.write("""
As mentioned above, only one model has been used in this project: a **yolov10x** model for customer head 
detection. The model is the largest variant of its kind (as inferred by the "x" in the name of the model), and thus has 
the highest mAP (Mean Average Precision) value, as benchmarked on the COCO dataset.

In the case of this project, a pre-trained model (yolov10x.pt) was not used, rather a new model (yolov10x.yaml) was 
trained from scratch (this resulted in better precision and recall).

The model was accessed and trained using the Ultralytics YOLO API, as provided in the **"ultralytics"** Python
package.

For more information regarding the YOLO models, refer to the [Ultralytics YOLO docs](https://docs.ultralytics.com/)
""")

st.write("##### Training Analysis")
st.write("""
The model was trained using the ".train" function present in the ultralytics YOLO class.

The maximum amount of epochs that the model could train for was set to 600, with the patience of 50 epochs
(patience is a hyper-parameter that defines how many epochs will pass before EarlyStopping stops the training due to no
improvement). The model was trained using Kaggle cloud computing and thus two Nvidia Tesla T4 GPUs were used for
training the model. The optimizer was set to "auto", due to which "Stochastic Gradient Descent (aka SGD)" was used as 
the optimizer. Default values were used for all other hyper-parameters.

The model was trained for several hours before EarlyStopping terminated the training.
""")

st.write("##### Model Links")
st.write("""
The trained model for this project can be 
found on my [Kaggle](https://www.kaggle.com/models/pythonistasamurai/yolov10x-supervision-queue-count-model).
""")

st.subheader("Prediction Pipeline", divider=True)

st.info(
    body="**For detailed in-line comment explanation of the prediction pipeline visit my "
         "[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/object-detection/yolov10x-supervision-dwell-time-analysis).**",
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

st.write("##### Creating a Class for Time Estimation")
st.write("""
In this part of the prediction pipeline, a class is created for estimation of the time spent by each customer in the
cashier counter queues. Time is estimated based upon the number of frames each customer, present in the cashier
counter queues, spends in the video. The entry frame of each customer in the cashier queues is added as value to a
dictionary, with the tracker ID of the customer, as the key. A tick function is defined within the class, which deals
with incrementing the frames, and at the same time estimating the time by using the entry frame of each tracker ID and
the current frame count.

The following formula is used for time estimation:
""")

st.code("""
time = (current_frame_count - entry_frame_count) / FPS
""", language="Python")

st.write("##### Loading the Input Video and Instantiating the Model")
st.write("""
In this part of the prediction pipeline, the input video is loaded into memory using the cv2.VideoCapture class. The
trained yolov10x model for customer head detection is also instantiated using the ultralytics.YOLO class.
""")

st.write("##### Defining and Initializing Polygon Zones for Filtering Detections")
st.write("""
In this part of the prediction pipeline, the polygon zones bordering the cashier counter queues are defined and are 
then converted to supervision format using the sv.PolygonZone class. These zones will be used to filter the customers 
from the cashiers.
""")

st.write("##### Initializing Annotators")
st.write("""
In this part of the prediction pipeline, supervision annotators, for annotating the input video frames with the desired 
information, are initialized.

1) LabelAnnotator is used to annotate the frames with the time each customer spends in the cashier counter queues.
2) BoxAnnotator is used to annotate the frames with boxes around the customers.
3) PolygonZoneAnnotators are used to annotate the frames with the polygon zones defined above.
""")

st.write("##### Initializing the Tracker and the Output Video Writer")
st.write("""
In this part of the prediction pipeline, a tracker (sv.ByteTrack) is initialized to track the 
customers throughout the video frames. This is important to keep track of the total time spent by the customers in the 
cashier counter queues. In addition to the tracker, a cv2.VideoWriter object is also initialized to merge the annotated frames into the output video.
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
[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/object-detection/yolov10x-supervision-dwell-time-analysis).
""")
