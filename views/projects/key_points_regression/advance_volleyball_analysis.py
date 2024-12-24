import streamlit as st

st.title("Advance Volleyball Analysis")
st.write("\n")

with st.container(height=400, border=True):
    st.video("assets/key_points_regression/advance_volleyball_analysis/processed_court.mp4", muted=True)

with st.container(height=400, border=True):
    st.video("assets/key_points_regression/advance_volleyball_analysis/processed_radar.mp4", muted=True)

st.write("\n")
st.subheader("Project Overview", divider=True)
st.write("""
In this project I have utilized advance computer vision skills, including but not limited to, object detection, key 
points regression, perspective transformations, and homography calculations, for carrying out an advance volleyball
match analysis.

I have precisely annotated three datasets, trained three yolov8 models, and have produced a radar view of the volleyball
match!
""")

st.write("\n")
st.subheader("Dataset Pipeline", divider=True)

st.write("##### Images Collection")
st.write("""
The images comprising the datasets were extracted from a short 22-second clip of a Volleyball match uploaded on 
[YouTube](https://www.youtube.com/watch?v=BfcL_cxB-9o&t=10s).

A total of 1320 frames were made available for annotation by extracting 60 frames per second.
""")

st.write("##### Images Annotation")
st.write("""
A total of three datasets were annotated for this project, those being:
    
1. Volleyball Court Key Points Regression Dataset (862 images).
2. Volleyball Ball Object Detection Dataset (548 images).
3. Volleyball Players And Referee Object Detection Dataset (66 images).

Roboflow Workspace was chosen as the platform of choice for images annotation. All datasets were annotated precisely
over the course of several months. All datasets can be found on my 
[Kaggle](https://www.kaggle.com/pythonistasamurai/datasets) and [Roboflow](https://universe.roboflow.com/primaryws/).
""")

st.write("##### Pre-Processing And Augmentations")
st.write("""
Following pre-processing and augmentations were applied to the datasets:

1. Volleyball Court Key Points Regression Dataset 
    - All versions were stretched to 640 pixels in width and height.
    - Two versions of this dataset were subjected to grayscale.
    - No augmentations applied.
2. Volleyball Ball Object Detection Dataset
    - Both versions were stretched to 640 pixels in width and height.
    - One version was subjected to grayscale.
    - No augmentations applied.
3. Volleyball Players And Referee Object Detection Dataset
    - Version was stretched to 640 pixels in width and height.
    - Version was augmented by flipping the images horizontally between -15° and +15°.
""")

st.write("##### Datasets Exporting")
st.write("""
The datasets were exported in the following manner:

1. Volleyball Court Key Points Regression Dataset 
    - Four versions were exported.
    - All versions can be downloaded in multiple formats through my 
    [Roboflow](https://universe.roboflow.com/primaryws/volleyball_court_key_points_regression_dataset).
    - The version used for training the best model can be found on my 
    [Kaggle](https://www.kaggle.com/datasets/pythonistasamurai/volleyball-court-key-points-regression-dataset).
2. Volleyball Ball Object Detection Dataset
    - Two versions were exported.
    - Both versions can be downloaded in multiple formats through my 
    [Roboflow](https://universe.roboflow.com/primaryws/volleyball_ball_object_detection_dataset).
    - The version used for training the best model can be found on my 
    [Kaggle](https://www.kaggle.com/datasets/pythonistasamurai/volleyball-ball-object-detection-dataset).
3. Volleyball Players And Referee Object Detection Dataset
    - One version was exported.
    - The version can be downloaded in multiple formats through my 
    [Roboflow](https://universe.roboflow.com/primaryws/volleyball_players_and_referee_object_detection_dataset), or
    in the YOLO format through my 
    [Kaggle](https://www.kaggle.com/datasets/pythonistasamurai/volleyball-players-and-referee-object-detection).
""")

st.write("\n")
st.subheader("Training Pipeline", divider=True)

st.write("##### Models Used")
st.write("""
Three YOLOv8 models were trained for this project:

1. A YOLOv8x-pose model for Volleyball Court Key Points Regression (trained from scratch using yolov8x-pose.yaml). 
2. A YOLOv8x model for Volleyball Ball Object Detection (trained from scratch using yolov8x.yaml).
3. A YOLOv8x model for Volleyball Players and Referee Object Detection (trained from pre-trained weights using 
yolov8x.pt).

All models were accessed and trained using the Ultralytics YOLO API, as provided in the **"ultralytics"** Python
package.

For more information regarding the YOLO models, refer to the
[Ultralytics YOLO docs](https://docs.ultralytics.com/)
""")

st.write("##### Training Analysis")
st.write("""
All models were trained using the ".train" function present in the ultralytics YOLO class.

The models were trained for hundreds of epochs. Kaggle cloud computing was used for model training and thus two Nvidia 
Tesla T4 GPUs were used for training the models. The optimizer was set to "auto", due to which "Stochastic Gradient 
Descent (aka SGD)" was used as the optimizer for all trainings. Default values were used for all other hyper-parameters.

The models were trained for several hours before EarlyStopping terminated the trainings.
""")

st.write("##### Model Links")
st.write("""
The trained models for this project can be 
found on my [Kaggle](https://www.kaggle.com/models/pythonistasamurai/yolov8x_volleyball_analysis_models).
""")

st.subheader("Prediction Pipeline", divider=True)

st.info(
    body="**For detailed in-line comment explanation of the prediction pipeline visit my "
         "[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/key-points-regression/yolov8x-supervision-advance-volleyball-analysis).**",
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
import numpy as np
import supervision as sv
from skimage.color import gray2rgb
from ultralytics import YOLO
""", language="Python")

st.write("##### Specifying Paths")
st.write("""
In this part of the prediction pipeline, the absolute paths to the input video, output court video directory, output 
radar video directory, models, and the radar image to be annotated are specified.
""")

st.write("##### Loading and Manipulating Data")
st.write("""
In this part of the prediction pipeline, the radar court image to be annotated is loaded into memory and it's height 
and width are extracted for later use. In addition to that, the input video is also loaded using the cv2.VideoCapture 
function, and the width, height, and FPS information of the input video is extracted for later use. Afterwards, two 
output video writers, one for writing the annotated court video and the other for writing the annotated radar video are 
initialized using the cv2.VideoWriter class.
""")

st.write("##### Instantiating Trained Models")
st.write("""
In this part of the prediction pipeline, all three trained models for volleyball analysis are loaded into memory.
""")

st.write("##### Perspective And Homography Transformations")
st.write("""
In this part of the prediction pipeline, the selected key points present on the radar image being used, are defined. 
These key points will be used to transform the perspective of the court in the input video frames to the radar court 
image via homography. This is necessary for accurately estimating the locations of the players on the court in real 
life. Later we use the key points regressed by the key points regression model as the source matrix and the defined 
points list as the target matrix for homography calculation.

Later we define a class for calculating homography and carrying perspective transfromations.
""")

st.write("##### Initializing Annotators")
st.write("""
In this part of the prediction pipeline, supervision annotators, for annotating the input video frames with the desired 
information, are initialized.

1) EllipseAnnotator is used to annotate the frames with the ellipses surrounding the feet of the players and referee.
2) BoundingBoxAnnotator is used to annotate the frames with a box surrounding the detected ball.
3) VertexAnnotator is used to annotate the frames with the regressed court key points.
""")

st.write("##### Defining A Class For Homography And Perspective Transformations")
st.write("""
In this part of the prediction pipeline, a class for calculating the homography between the input video court 
and the radar court image using the key points regressed by the key points regression model as the source matrix and 
the radar court key points defined above as the target matrix, is defined. 
This class also provides the functionality to transform any cartesian coordinates from the input video court to the 
radar court image, which will be used to estimate the position of the players on the radar court image, using the 
points predicted by the player detection model on the input video court.
""")

st.write("##### Defining Functions")
st.write("""
In this part of the prediction pipeline, a function is defined for drawing the transformed points (those returned by
the class defined above) on the radar court image.
In addition to that, a frame processor function is defined that deals with all inference handling, logic implementation,
annotations, and video writing.
""")

st.write("##### Main Loop")
st.write("""
Lastly, the main loop that breaks the input video into frames and calls relevant functions is started. This loop
gets the models inference on the frames and then calls the frameProcessor() function for inference handling and 
frame processing.
""")

st.write("#### Code Link")
st.write("""
The code for this project can be found on my
[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/key-points-regression/yolov8x-supervision-advance-volleyball-analysis).
""")
