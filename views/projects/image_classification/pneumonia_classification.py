import streamlit as st

st.title("Pneumonia Classification")

st.write("\n")
st.subheader("Project Overview", divider=True)
st.write("""
In this project I have trained a model for classifying x-ray images of lungs in order to identify three types of
pneumonia, those being viral, bacterial, and COVID-19 variant. **This project has been implemented in both Keras and
PyTorch**.
""")

st.write("\n")
st.subheader("Training Pipeline", divider=True)
st.write("#### Training Data")
st.write("""
The dataset used for training the model is the 
[3 kinds of Pneumonia](https://www.kaggle.com/datasets/artyomkolas/3-kinds-of-pneumonia) dataset,
provided by **"Artsiom Kolas"** on Kaggle. 

This dataset has four classes, representing three types of pneumonia and normal lungs, for classification. The classes
are listed below.

1. Normal (3270 images)
2. Pneumonia-Bacterial (3001 images)
3. Pneumonia-Viral (1656 images)
4. Pneumonia-COVID-19 (1281 images)
""")

with st.container(height=500, border=True):
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.image(
            "assets/image_classification/pneumonia_images/normal.jpg",
            width=250,
            caption="Normal Lungs Image Sample"
        )
        st.image(
            "assets/image_classification/pneumonia_images/pneumonia_bacterial.jpg",
            width=250,
            caption="Bacterial Image Sample"
        )
    with col2:
        st.image(
            "assets/image_classification/pneumonia_images/pneumonia_viral.jpg",
            width=250,
            caption="Viral Image Sample"
        )
        st.image(
            "assets/image_classification/pneumonia_images/pneumonia_covid.jpg",
            width=250,
            caption="COVID Image Sample"
        )

st.write("""
Since the dataset was not structured to have a validation and training split, therefore 80% of the total images were
used for training and 20% were used for validation. A batch size of 32 was used for training.

Before being fed to the model for training, the training split was preprocessed to have a size of 128 pixels in
width and height, rescaled to grayscale, and in case of the PyTorch implementation of this project, converted to
torch tensors, and then assigned the data type torch.float32.
""")

st.write("#### Model Implementation")
st.write("""
The architecture for this model can be found on my 
[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/image-classification/pneumonia-classification).

The model is a deep convolutional neural network, consisting of 8 2D-Convolutional layers, 4 2D-MaxPooling layers,
4 2D-Dropout layers, 11 Rectified Linear Unit layers, 4 linear layers, and 4 2D-BatchNorm layers.
""")

with st.container(height=400, border=True):
    with st.columns(3)[1]:
        st.image(
            "assets/image_classification/pneumonia_images/pneumonia.keras.svg",
            caption="Model Architecture",
            width=200
        )

st.write("#### Training Analysis")
st.write("""
The model was trained for at most 30 epochs, with **Sparse Categorical CrossEntropy** being used as the loss
function in the keras implementation and **CrossEntropy** being used as the loss function in the PyTorch
implementation (both losses are essentially the same). Adam was used as the optimizer, with default 
hyper-parameter values, in both implementations.

Two callbacks were used for training (only in the Keras implementation for this project), those being EarlyStopping 
and ModelCheckpoint. The criterion for early stopping was set to training loss. If the training loss did not 
decrease for 5 consecutive epochs, the training would be terminated. As far as model checkpointing is concerned, 
after each epoch, the model with the best training loss (lowest training loss) was saved.

The trained model gave an accuracy higher than 96% (in case of PyTorch implementation the accuracy was calculated 
using the **Accuracy class** provided in the **torchmetrics** package).
""")

st.write("#### Model Links")
st.write("""
The trained models (plural because the same model was trained in both Keras and PyTorch) for this project can be 
found on my [Kaggle](https://www.kaggle.com/models/pythonistasamurai/pneumonia-classification-models).
""")
