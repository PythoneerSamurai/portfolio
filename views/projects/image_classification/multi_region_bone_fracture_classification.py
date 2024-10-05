import streamlit as st

st.title("Multi-Region Bone Fracture Classification")

st.write("\n")
st.subheader("Project Overview", divider=True)
st.write("""
In this project I have trained a model for classifying x-ray images of bones, covering all anatomical body regions,
including, but not limited to, lower limb, upper limb, lumbar, hips, and knees, to find out any fractures present
in them. **This project has been implemented in both Keras and PyTorch**.
""")

st.write("\n")
st.subheader("Training Pipeline", divider=True)
st.write("#### Training Data")
st.write("""
The dataset used for training the model is the 
[Bone Fracture Multi-Region X-ray Data](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data),
provided by **"Madushani Rodrigo"** on Kaggle. 

This dataset has two classes for classification, those being:

1. Fractured (contains the images from all anatomical body regions)
2. Non-Fractured

The training subset had 9246 images and the validation subset had 828 images.
""")

with st.container(height=400, border=True):
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.image(
            "assets/image_classification/bonefracture_images/fractured.jpeg",
            width=250,
            caption="Fractured Image Sample"
        )
    with col2:
        st.image(
            "assets/image_classification/bonefracture_images/non_fractured.png",
            width=250,
            caption="Non-Fractured Image Sample"
        )

st.write("""
The dataset was already structured to have both training and validation splits, and thus both splits were used for their
purpose.

Before being fed to the model for training, the training split was preprocessed to have a size of 128 pixels in
width and height rescaled to grayscale, and in the case of the PyTorch implementation of this project, converted to
torch tensors.
""")

st.write("#### Model Implementation")
st.write("""
The architecture for this model can be found on my 
[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/image-classification/multi-region-bone-fracture-classification).

The model is a convolutional neural network, consisting of 3 2D-Convolutional layers, 3 2D-MaxPooling layers,
3 2D-Dropout layers, 3 Rectified Linear Unit layers, 3 linear layers, and 3 2D-BatchNorm layers.
""")

with st.container(height=400, border=True):
    with st.columns(3)[1]:
        st.image(
            "assets/image_classification/bonefracture_images/boneFrac.keras.svg",
            caption="Model Architecture",
            width=200
        )

st.write("#### Training Analysis")
st.write("""
The model was trained for at most 30 epochs, with **BinaryCrossEntropy** being used as the loss function for both Keras
and PyTorch implementations. Adam was used as the optimizer, with default hyper-parameter values, in both 
implementations.

Only one callback was used for training (only in the Keras implementation for this project), that being EarlyStopping. 
The criterion for early stopping was set to training loss. If the training loss did not decrease for 5 consecutive 
epochs, the training would be terminated.

The trained model gave an accuracy higher than 96% (in the case of PyTorch implementation the accuracy was calculated 
using the **Accuracy class** provided in the **torchmetrics** package).
""")

st.write("#### Model Links")
st.write("""
The trained models (plural because the same model was trained in both Keras and PyTorch) for this project can be 
found on my [Kaggle](https://www.kaggle.com/models/pythonistasamurai/multi-region-bone-fracture-classification-models).
""")
