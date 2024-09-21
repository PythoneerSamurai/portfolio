import streamlit as st

st.title("Leaf Disease Analysis")
st.write("\n")

with st.container(height=380, border=True):
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.image(
            image="assets/segmentation/leaf_disease_analysis/disease_seg_one.jpg",
            caption="Disease Segmentation One"
        )
    with col2:
        st.image(
            image="assets/segmentation/leaf_disease_analysis/disease_seg_two.jpg",
            caption="Disease Segmentation Two"
        )
    st.image(
        image="assets/segmentation/leaf_disease_analysis/disease_seg_three.jpg",
        caption="Disease Segmentation Three"
    )

st.write("\n")
st.subheader("Project Overview", divider=True)
st.write("""
In this project I have utilized semantic segmentation for the purpose of segmenting leaf diseases. I have successfully
trained a model which can perform the aforementioned task with high precision and accuracy.
""")

st.write("\n")
st.subheader("Training Pipeline", divider=True)
st.write("##### Training Data")
st.write("""
In the case of this project, only one yolov8x model has been utilized, that for the purpose of leaf disease 
segmentation.

The dataset used to train the model is the
[Leaf disease segmentation dataset](https://www.kaggle.com/datasets/fakhrealam9537/leaf-disease-segmentation-dataset) 
dataset provided by "**Fakhre Alam**" on Kaggle.

The dataset has only one class for segmentation, that is "disease".

The datasets were preprocessed and the images had a size of 640 pixels in width and height. The datasets did not have
YOLO format .txt segmentation labels, rather they came with binary masks, therefore I used an open-source script to
convert those binary masks into YOLO .txt files, having the same name as the images they belonged to.

The dataset was re-structured according to the YOLO format, having two folders "images" and "labels", both of which
had sub-directories "train", "test", and "val".
""")

st.write("##### Model Used")
st.write("""
As mentioned above, only one model has been used in this project, that is a **yolov8x** model for leaf disease
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
found on my [Kaggle](https://www.kaggle.com/models/pythonistasamurai/yolov8x-supervision-leaf-disease-analysis-model).
""")

st.subheader("Prediction Pipeline", divider=True)

st.info(
    body="**For detailed in-line comment explanation of the prediction pipeline visit my "
         "[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/segmentation/yolov8x-supervision-leaf-disease-analysis).**",
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
import supervision as sv
from ultralytics import YOLO
""", language="Python")

st.write("##### Specifying Paths")
st.write("""
In this part of the prediction pipeline, three paths are specified for the proper functioning of the code, those being,
the paths to the model, the input image, and the output folder.
""")

st.write("##### Instantiating the Model")
st.write("""
In this part of the prediction pipeline, the trained yolov8x model for leaf disease segmentation is instantiated into
memory using the ultralytics YOLO class, provided in the "**ultralytics**" Python package.
""")

st.write("##### Initializing an Annotator and Loading the Input Image")
st.write("""
In this part of the prediction pipeline, a supervision MaskAnnotator object is initialized into memory, that for the 
purpose of annotating the input image with the leaf disease segmentations.

The input image is also loaded into memory using the cv2.imread() function.
""")

st.write("##### Getting Model's Inference and Defining the Frame Processor Function")
st.write("""
Lastly in the prediction pipeline, the trained model's inference is carried out on the input image. Later on, a function
is defined for the purpose of handling all functionality, ranging from inference conversion (from YOLO format to
supervision format) to the saving of the annotated image.
""")

st.write("#### Code Link")
st.write("""
The code for this project can be found on my
[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/segmentation/yolov8x-supervision-leaf-disease-analysis).
""")
