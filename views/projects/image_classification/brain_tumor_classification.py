import streamlit as st

st.title("Brain Tumor Classification")

st.write("\n")
st.subheader("Project Overview", divider=True)
st.write("""
According to the **National Institute of Neurological Disorders and Strokes**:
> *A tumor is a mass of abnormal cells that either form into a new growth or the growth was there when you were born 
(congenital). Tumors occur when something goes wrong with genes that regulate cell growth, allowing cells to grow and 
divide out of control. Tumors can form anywhere in your body. Brain and spinal cord tumors form in the tissue inside 
your brain or spinal cord, which make up the central nervous system (CNS).*

also,
> *Depending on its type, a growing tumor may not cause any symptoms or can kill or displace healthy cells or disrupt 
their function. A tumor can move or press on sensitive tissue and block the flow of blood and other fluid, causing pain
and inflammation. A tumor can also block the normal flow of activity in the brain or signaling to and from the brain.*

In this project I have trained a model for the accurate classification of three types of brain tumors, which can be
used by medical practitioners for the diagnosis of brain tumors. **This project has been implemented in both Keras
and PyTorch**.
""")

st.write("\n")
st.subheader("Training Pipeline", divider=True)
st.write("#### Training Data")
st.write("""
The dataset used for training the model is the 
[Brain Tumor Image Dataset](https://www.kaggle.com/datasets/denizkavi1/brain-tumor?select=1),
provided by **"Deniz Kavi"** on Kaggle. 

This dataset has a total of three classes, each of which represents a type of brain tumor. The classes are listed
below.

1. Meningioma samples
2. Glioma samples
3. Pituitary tumor samples
""")

with st.container(height=600, border=True):
    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.image(
            "assets/image_classification/brain_tumor_images/meningioma.png",
            width=250,
            caption="Meningioma Sample"
        )
    with col2:
        st.image(
            "assets/image_classification/brain_tumor_images/glioma.png",
            width=250,
            caption="Glioma Sample "
        )
    st.image(
        "assets/image_classification/brain_tumor_images/pituitary_tumor.png",
        width=250,
        caption="Pituitary Tumor Sample"
    )

st.write("""
Since the dataset was not structured to have a validation and training split, therefore 80% of the total images were
used for training and 20% were used for validation. A batch size of 64 was used for training.

Before being fed to the model for training, the training split was preprocessed to have a size of 128 pixels in
width and height rescaled to grayscale, and in the case of the PyTorch implementation of this project, converted to
torch tensors.
""")

st.write("#### Model Implementation")
st.write("""
The architecture for this model can be found on my 
[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/image-classification/brain-tumor-classification).

The model is a deep convolutional neural network, consisting of 8 2D-Convolutional layers, 4 2D-MaxPooling layers,
4 2D-Dropout layers, 11 Rectified Linear Unit layers, 4 linear layers, and 4 2D-BatchNorm layers.
""")

with st.container(height=400, border=True):
    with st.columns(3)[1]:
        st.image(
            "assets/image_classification/brain_tumor_images/brainTumor.keras.svg",
            caption="Model Architecture",
            width=200
        )

st.write("#### Training Analysis")
st.write("""
The model was trained for at most 20 epochs, with **Sparse Categorical CrossEntropy** being used as the loss
function in the Keras implementation and **CrossEntropy** being used as the loss function in the PyTorch
implementation (both losses are essentially the same). Adam was used as the optimizer, with default 
hyper-parameter values, in both implementations.

Two callbacks were used for training (only in the Keras implementation for this project), those being EarlyStopping 
and ModelCheckpoint. The criterion for early stopping was set to training loss. If the training loss did not 
decrease for 5 consecutive epochs, the training would be terminated. As far as model checkpointing is concerned, 
after each epoch, the model with the best training loss (lowest training loss) was saved.

The trained model gave an accuracy higher than 96% (in the case of PyTorch implementation the accuracy was calculated 
using the **Accuracy class** provided in the **torchmetrics** package).
""")

st.write("#### Model Links")
st.write("""
The trained models (plural because the same model was trained in both Keras and PyTorch) for this project can be 
found on my [Kaggle](https://www.kaggle.com/models/pythonistasamurai/brain-tumor-classification-models).
""")
