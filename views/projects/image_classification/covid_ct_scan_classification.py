import streamlit as st

st.title("COVID CT Scans Classification")

st.write("\n")
st.subheader("Project Overview", divider=True)
st.write("""
COVID is a pandemic, the effects of which are hidden from none. For almost 4 years COVID has continuously infected and
destroyed entire populations, and even today COVID is far from gone.

The rate at which COVID spread, and the forms it evolved into, made it very difficult for medicinal companies to produce
a vaccine for the virus, and it took years for the first vaccine to enter the market. At the same time, what made COVID
even worse was that it shared it's symptoms with common diseases like flu and chest infections, due to which it
became difficult for medical practitioners and the common person, to classify himself to be infected by COVID or not.

Soon enough datasets which could be used to train AI models for the diagnosis of COVID started to surface on the
internet, and were gratefully welcomed by AI engineers and researchers around the world.

In this project I have trained a model for the accurate classification of the presence or non-presence of COVID, based 
upon CT scans of the lungs of COVID infected patients and normal people. **This project has been implemented in both
Keras and PyTorch**.
""")

st.write("\n")
st.subheader("Training Pipeline", divider=True)
st.write("#### Training Data")
st.write("""
The dataset used for training the model is the 
[SARS-COV-2 Ct-Scan Dataset](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset),
provided by **"PlamenEduardo"** on Kaggle. 

This dataset has two classes for classification, those being:

1. COVID-Infected (1252 training images)
2. Non-Infected (1230 training images)
""")

with st.container(height=300, border=True):
    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.image(
            "assets/image_classification/covid_images/covid_infected.png",
            width=250,
            caption="COVID-Infected Sample"
        )
    with col2:
        st.image(
            "assets/image_classification/covid_images/non_infected.png",
            width=250,
            caption="Non-Infected Sample"
        )

st.write("""
This dataset was not structured to have a training and validation split. In case of this project, the entire dataset
was used for training and no images were reserved for validation.

Before being fed to the model for training, the training split was preprocessed to have a size of 128 pixels in
width and height, rescaled to grayscale, and in case of the PyTorch implementation of this project, converted to
torch tensors.
""")

st.write("#### Model Implementation")
st.write("""
The architecture for this model can be found on my 
[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/image-classification/sars-covid-two-ct-scan-classification).

The model is a convolutional neural network, consisting of 3 2D-Convolutional layers, 3 2D-MaxPooling layers,
3 2D-Dropout layers, 5 Rectified Linear Unit layers, and 3 linear layers, 3 2D-BatchNorm layers, and 1 sigmoid layer as 
the final layer.
""")

with st.container(height=400, border=True):
    with st.columns(3)[1]:
        st.image(
            "assets/image_classification/covid_images/sarsCov.keras.svg",
            caption="Model Architecture",
            width=200
        )

st.write("#### Training Analysis")
st.write("""
The model was trained for at most 30 epochs, with **BinaryCrossEntropy** being used as the loss function in both the
Keras and PyTorch implementations. Adam was used as the optimizer, with default hyper-parameter values, in both 
implementations.

Only EarlyStopping was used as a callback for this project. The criterion for early stopping was set to training loss 
(training accuracy was used as the criterion for the PyTorch implementation). If the training loss did not decrease for 
5 consecutive epochs (or the training accuracy did not increase for 5 consecutive epochs, in the case of PyTorch 
implementation), the training would be terminated.

The trained model gave an accuracy higher than 96% (in case of PyTorch implementation the accuracy was calculated 
using the **Accuracy class** provided in the **torchmetrics** package).
""")

st.write("#### Model Links")
st.write("""
The trained models (plural because the same model was trained in both Keras and PyTorch) for this project can be 
found on my [Kaggle](https://www.kaggle.com/models/pythonistasamurai/sars-covid-two-models).
""")
