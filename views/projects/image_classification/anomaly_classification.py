import streamlit as st

st.title("Anomaly Classification")

st.write("\n")
st.subheader("Project Overview", divider=True)
st.write("""
Anomaly can be defined as,

> *Deviation or departure from the normal or common order, form, or rule.*

Anomaly classification is an important task in the field of computer vision. Places that require or follow a strict
disciplinary pattern for their subjects can use anomaly classification models to keep a check on the flow of things.
Anomaly classification models can also be used for security surveillance, for the purpose of maintaining public 
security, and to classify unwanted events, such as crime, robbery, or fights.

In this project I have trained a model that can classify various anomalous events, majorly those that lie within
the boundaries of crime. **This project has been implemented in both Keras and PyTorch**.
""")

st.write("\n")
st.subheader("Training Pipeline", divider=True)
st.write("#### Training Data")
st.write("""
The dataset used for training the model is the 
[UCF Crime Dataset](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset),
provided by **"Sanskar Hasija"** on Kaggle. 

This dataset has a total of fourteen classes, eleven of which represent criminal acts, two need not always represent 
intentional crime (explosion and road accidents), and one class represent normal flow of things (normal videos).

1. Abuse 
2. Arrest
3. Arson
4. Assault
5. Burglary
6. Explosion
7. Fighting
8. Normal Videos
9. RoadAccidents
10. Robbery
11. Shooting
12. Shoplifting
13. Stealing
14. Vandalism

This dataset was prepared by extracting the tenth frame from every video in the UCF Crime Dataset.

The total image count for the train subset is 1,266,345, and for the test subset is 111,308.

Since the dataset was not structured to have a validation and training split, therefore 70% of the total images were
used for training and 30% were used for validation. A batch size of 32 was used for training.

The dataset was standardized and the images came with a size of 64 pixels in width and height. The images were then 
rescaled to grayscale, and in case of the PyTorch implementation of this project, converted to torch tensors.
""")

st.write("#### Model Implementation")
st.write("""
The architecture for this model can be found on my 
[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/image-classification/anomaly-classification-ucf-crime-dataset).

The model is a convolutional neural network, consisting of 2 2D-Convolutional layers, 2 2D-MaxPooling layers,
2 2D-Dropout layers, 3 linear layers, and 2 2D-BatchNorm layers.
""")

st.write("#### Training Analysis")
st.write("""
The model was trained for a total of 50 epochs, with **Sparse Categorical CrossEntropy** being used as the loss
function in the keras implementation and **CrossEntropy** being used as the loss function in the PyTorch
implementation (both losses are essentially the same). Adam was used as the optimizer, with default 
hyper-parameter values, in both implementations.

No callbacks were used in this projects.
""")

st.write("#### Model Links")
st.write("""
I could not find the trained models for this project, however one can run either of the implementations
(Keras or PyTorch) belonging to this project, present on my GitHub, to train the model (it won't take much time
because image classification is far less resource intensive).
""")
