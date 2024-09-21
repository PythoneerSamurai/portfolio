import streamlit as st

st.title("Colon Histopathological Images Classification")

st.write("\n")
st.subheader("Project Overview", divider=True)
st.write("""
According to **verywellhealth.com**:
> *Histopathology involves using a microscope to look at human tissue to check for signs of disease. The term is 
derived from “histology” (meaning the study of tissues), and “pathology” (meaning the study of disease).*

Histopathological images can be used for various types of tissue analysis, such as checking for accumulation of white 
blood cells, crystallized deposits, and abnormal cell formations. Histopathological images are RGB images, having
a large amount of fine details, which makes it a lot more difficult for AI models to understand the intricate
relationships between image features, but at the same time, it makes it a lot more fun for computer vision engineers
to design the best architecture to classify these images.

In this project I have trained a model for the accurate classification of colon diseases using histopathological
images of colon slices. **This project has been implemented in both Keras and PyTorch**.
""")

st.write("\n")
st.subheader("Training Pipeline", divider=True)
st.write("#### Training Data")
st.write("""
The dataset used for training the model is the 
[Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) dataset,
provided by **"Larxel"** on Kaggle. 

This dataset has a total of five classes for classification, however only two of the classes represent colon diseases,
those being:

1. Colon adenocarcinoma
2. Colon benign tissue
""")

with st.container(height=300, border=True):
    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.image(
            "assets/image_classification/colon_images/colon_adenocarcinoma.jpeg",
            width=250,
            caption="Colon Adenocarcinoma Sample"
        )
    with col2:
        st.image(
            "assets/image_classification/colon_images/colon_benign.jpeg",
            width=250,
            caption="Colon Benign Sample"
        )

st.write("""
Since the dataset was not structured to have a validation and training split, therefore 80% of the total images were
used for training and 20% were used for validation. A batch size of 32 was used for training.

Before being fed to the model for training, the training split was preprocessed to have a size of 128 pixels in
width and height, rescaled to grayscale, and in case of the PyTorch implementation of this project, converted to
torch tensors.
""")

st.write("#### Model Implementation")
st.write("""
The architecture for this model can be found on my 
[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/image-classification/colon-histopathological-images-classification).

The model is a deep convolutional neural network, consisting of 8 2D-Convolutional layers, 4 2D-MaxPooling layers,
4 2D-Dropout layers, 11 Rectified Linear Unit layers, 4 linear layers, and 4 2D-BatchNorm layers.
""")

with st.container(height=400, border=True):
    with st.columns(3)[1]:
        st.image(
            "assets/image_classification/colon_images/colonHisto.keras.svg",
            caption="Model Architecture",
            width=200
        )

st.write("#### Training Analysis")
st.write("""
The model was trained for at most 3 epochs (three epochs were enough for training this architecture, further training
would result in over-fitting), with **BinaryCrossEntropy** being used as the loss function in both Keras and PyTorch
implementations. Adam was used as the optimizer, with default hyper-parameter values, in both implementations.

ModelCheckpoint was the only callback used in this project (in Keras implementation only). After each epoch, the model 
with the best training loss (lowest training loss) was saved.

The trained model gave an accuracy higher than 96% (in case of PyTorch implementation the accuracy was calculated 
using the **Accuracy class** provided in the **torchmetrics** package).
""")

st.write("#### Model Links")
st.write("""
The trained models (plural because the same model was trained in both Keras and PyTorch) for this project can be on my 
[Kaggle](https://www.kaggle.com/models/pythonistasamurai/colon-histopathological-images-classification).
""")
