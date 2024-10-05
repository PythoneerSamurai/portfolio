import streamlit as st

st.title("Advance Tennis Analysis")
st.write("\n")

with st.container(height=350, border=True):
    st.video("assets/object_detection/advance_tennis_analysis/output_video.mp4", muted=True)

st.write("\n")
st.subheader("Project Overview", divider=True)
st.write("""
In this project, I have utilized advanced computer vision skills, including but not limited to, region-based object
detection, object tracking, object speed estimation, and multiple perspective transformations, for
carrying out a detailed analysis of a tennis match.

I have accurately estimated ball speeds, classified players, and produced a radar view of the court using multiple 
perspective transformations!
""")

st.write("\n")
st.subheader("Training Pipeline", divider=True)

st.write("##### Training Data")
st.write("""
Two models, yolov8x and yolov10x, have been trained for this project, the former for tennis ball object detection and
the latter for player object detection.

The dataset used to train the tennis ball detection model is the
[tennis ball detection](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection)
provided by "**Viren Dhanwani**" on Roboflow Universe.

The dataset used to train the player detection model is the
[person detection](https://universe.roboflow.com/konstantin-sargsyan-wucpb/person-detection-lc325)
provided by "**Konstantin Sargsyan**" on Roboflow Universe.

Both datasets have only one class for detection, the former having the class "tennis ball" and the latter having the
class "people".

The datasets were standardized and the images had a size of 640 pixels in width and height. The object detection
labels were in the YOLO format, stored in .txt files sharing the same name as the images they belonged to.

The datasets were re-structured according to the YOLO format, having two folders "images" and "labels", both of which
had sub-directories "train", "test", and "val".
""")

st.write("##### Models Used")
st.write("""
As mentioned above, two models have been used in this project, a **yolov8x** model for tennis ball object detection and 
a **yolov10x** model for player object detection. Both models are the largest variants of their kind (as inferred by the
"x" in the names of the models), and thus have the highest mAP (Mean Average Precision) values, as benchmarked on the
COCO dataset.

In the case of this project, pre-trained models (yolov8x.pt and yolov10x.pt) were not used, rather new models 
(yolov8x.yaml and yolov10x.yaml) were trained from scratch (this resulted in better precision and recall).

Both the models were accessed and trained using the Ultralytics YOLO API, as provided in the **"ultralytics"** Python
package.

For more information regarding the YOLO models, refer to the
[Ultralytics YOLO docs](https://docs.ultralytics.com/)
""")

st.write("##### Training Analysis")
st.write("""
Both models were trained using the ".train" function in the ultralytics YOLO class.

The maximum amount of epochs that the models could train for was set to 600, with the patience of 50 epochs
(patience is a hyper-parameter that defines how many epochs will pass before EarlyStopping stops the training due to no
improvement). The models were trained using Kaggle cloud computing; thus, two Nvidia Tesla T4 GPUs were used for
training the models. The optimizer was set to "auto", due to which "Stochastic Gradient Descent (aka SGD)" was used as 
the optimizer. Default values were used for all other hyper-parameters.

The models were trained for several hours before EarlyStopping terminated the trainings.
""")

st.write("##### Model Links")
st.write("""
The trained models for this project can be 
found on my [Kaggle](https://www.kaggle.com/models/pythonistasamurai/yolov8x_v10x_tennis_analysis_models).
""")

st.subheader("Prediction Pipeline", divider=True)

st.info(
    body="**For detailed in-line comment explanation of the prediction pipeline visit my "
         "[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/object-detection/yolov8x-v10x-supervision-advance-tennis-analysis).**",
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
from collections import defaultdict, deque

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
""", language="Python")

st.write("##### Specifying Paths")
st.write("""
In this part of the prediction pipeline, the absolute paths to the trained tennis ball detection model, player
detection model, input video, output video export directory, and the radar court image to be annotated is specified.
""")

st.write("##### Loading and Manipulating Data")
st.write("""
In this part of the prediction pipeline, the radar court image to be annotated is loaded into memory. In addition to 
that, the input video is also loaded using the cv2.VideoCapture function, and the width, height, and FPS information 
of the input video are extracted for later use. Afterward, the output video writer is initialized using the 
cv2.VideoWriter class. The parameters used for writing the output video are listed below.
""")

st.code("""
FOUR_CC = cv2.VideoWriter.fourcc(*"mp4v")
OUTPUT_VIDEO_FPS = 20.0
VIDEO_RESOLUTION = (2372, HEIGHT)
""", language="Python")

st.write("##### Perspective Transformations")
st.write("""
In this part of the prediction pipeline, the matrices for perspective transformations are defined. To accurately
estimate the speed of the ball and its location on the court in real life, we must remove the visual distortion caused 
by the single camera recording the match. This is because in the video the farther part of the court occupies fewer 
pixels in width as compared to the closer part of the court, the same goes for the length of the court which cannot be 
correctly estimated using the pixel values of the input video frames. Therefore, we must transform the pixel values 
occupied by the court in the video frames to a court having pixel width and height same as the court's in real life. In 
this way, we can use a single pixel of the transformed court as 1 meter in length. Ultimately, we can use the transformed 
court to correctly estimate the speeds of the ball and its location on the court.

We define three matrices for perspective transformations.

1) The source matrix has the coordinates of each corner of the court in the video.
2) The target real view matrix has the coordinates of each corner of the court in real life (where 1 meter = 1 pixel).
3) The target radar view matrix has the coordinates of each corner of the radar court image loaded above, this image
   will be used to show a top-down view of the court along with the positions the ball hit on the court.
   
Afterward, the cv2.getPerspectiveTransform function is used to transform the source matrix to the corresponding output
matrices.

Later we define functions to transform the coordinates travelled by the ball on the source matrix to the output 
matrices.
""")

st.write("##### Defining and Initializing Polygon Zones for Filtering Detections")
st.write("""
In this part of the prediction pipeline, the polygon zones that the court will be divided into to filter detections, are 
defined. The court polygons will be used to define the regions where the ball can be present in the 
court. The player polygons are used to define the regions the players move in, this is to filter the players from other
people in the video.

After the polygon zones are defined, sv.PolygonZone objects are initialized to convert the polygon coordinates defined 
above into supervision format and to specify the triggering anchors (the xy-coordinates) that will decide if an object 
is inside the zones or not. Player one polygon zone is the region in the video in which player one can move in, the same 
goes for the player two polygon zone. Along with the player polygon zones, the court polygon zones are also initialized,
these zones will be used to decide the xy-coordinates of the ball in the court.
""")

st.write("##### Initializing Annotators")
st.write("""
In this part of the prediction pipeline, supervision annotators, for annotating the input video frames with the desired 
information, are initialized.

1) LabelAnnotator is used to annotate the frames with the player names (in this case "Djokovic" and "Sonego")
2) RoundBoxAnnotator is used to annotate the frames with boxes having rounded corners around the ball and the players.
3) PolygonZoneAnnotators are used to annotate the frames with the polygon zones defined above.
""")

st.write("##### Initializing Trackers")
st.write("""
In this part of the prediction pipeline, trackers (sv.ByteTrack) to track the ball and the players throughout the video, 
are initialized. Tracker IDs received by the ball tracker are used to estimate the speeds of the ball, whereas the 
information from the player tracker is used to keep track of the players for correct name annotation.
""")

st.write("##### Instantiating Trained Models")
st.write("""
In this part of the prediction pipeline, both of my trained models (yolov8x model for tennis ball detection and yolov10x
model for player detection). are instantiated, and the confidence scores for the models are specified to be 0.2.
""")

st.write("##### Specifying Parameters and Instantiating Objects")
st.write("""
In this part of the prediction pipeline, various parameters are initialized for annotating the frames with textual
information representing tennis analytics and for drawing circles on the radar court image corresponding to the
points at which the tennis ball hits the court.

Afterwards, three defaultdict objects for storing and processing information are initialized. Dictionaries are not used 
because they give a key error if a specific key is not present in the dictionary, however defaultdict fixes this by 
adding a non-present key into the dictionary and assigning it an empty value of the variable type.
Another point to be noted is that in the "coordinates" defaultdict, deque is used in place of a normal list. This is 
because deque (doubly ended queue) allows faster access to the first and last index, which will serve beneficial
later on in speed estimation.
""")

st.write("##### Defining Various Functions")
st.write("""
In this part of the prediction pipeline, two functions are defined for transforming the xy-coordinates travelled by the
ball on the source matrix (as defined before), to the target matrices.

Moreover, functions for annotating the radar view image with circles, annotating frames with tennis analytics, and
the main function that deals with all frame processing is defined.
""")

st.write("##### Main Loop")
st.write("""
Lastly, the main loop that breaks the input video into frames and calls relevant functions is started. This loop
gets the model inferences on the frames and then calls the frameProcessor() function for inference handling and 
frame processing.
""")

st.write("#### Code Link")
st.write("""
The code for this project can be found on my
[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/object-detection/yolov8x-v10x-supervision-advance-tennis-analysis).
""")
