import streamlit as st

st.title("Blood Cancer Classification")

st.write("\n")
st.subheader("Project Overview", divider=True)
st.write("""
There can be no doubt that blood cancer is one of the worst and most life-threatening health conditions faced by
populations worldwide. One major factor that contributes to the worsening of this condition is the late diagnosis
of the disease. With every passing second cancer gradually spreads throughout the body, therefore it is of utmost
importance that upon being exposed to its symptoms, one must get diagnosed.

Though a lot of progress has been made in recent times for fast, efficient, and accurate diagnosis of blood cancer,
still, the affected has to go through multiple tests, including but not limited to, bone marrow analysis and
hematology tests, needless to say, these tests are quite painful.

Now with the inception of Artificial Intelligence, researchers worldwide have been conducting research to discover,
utilize, and push the limits of what AI can offer in the field of medical diagnosis. Without any doubt, AI models have 
astonished minds with their highly accurate and precise disease diagnosis capabilities.

In this project, I have trained a model that consumes images of blood cells and is capable of classifying four types of
ALL (Acute Lymphoblastic Leukemia), if present, with a high accuracy. **This project has been implemented in both
Keras and PyTorch**.
""")

st.write("\n")
st.subheader("Training Pipeline", divider=True)
st.write("#### Training Data")
st.write("""
The dataset used for training the model is the 
[Blood Cells Cancer (ALL) dataset](https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class),
provided by **"M Amir Eshraghi and Mustafa Ghaderzadeh"** on Kaggle. 

This dataset has a total of four classes, three of which represent malignant ALL, and one represents benign (harmless)
ALL. The classes are listed below.

1. Benign
2. [Malignant] Pre-B
3. [Malignant] Pro-B
4. [Malignant] Early Pre-B
""")

with st.container(height=500, border=True):
    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.image("assets/image_classification/blood_cancer_images/benign.jpg", width=250, caption="Benign")
        st.image("assets/image_classification/blood_cancer_images/pre_b.jpg", width=250, caption="Pre-B")
    with col2:
        st.image("assets/image_classification/blood_cancer_images/pro_b.jpg", width=250, caption="Pro-B")
        st.image("assets/image_classification/blood_cancer_images/early_pre_b.jpg", width=250, caption="Early Pre-B")

st.write("""
Since the dataset was not structured to have a validation and training split, therefore 80% of the total images were
used for training and 20% were used for validation. A batch size of 32 was used for training.

Before being fed to the model for training, the training split was preprocessed to have a size of 128 pixels in
width and height, rescaled to grayscale, and in the case of the PyTorch implementation of this project, converted to
torch tensors.
""")

st.write("#### Model Implementation")
st.write("""
The architecture for this model can be found on my 
[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/image-classification/blood-cancer-cells-classification).

The model is a deep convolutional neural network, consisting of 8 2D-Convolutional layers, 4 2D-MaxPooling layers,
4 2D-Dropout layers, 11 Rectified Linear Unit layers, 4 linear layers, and 4 2D-BatchNorm layers.
""")

with st.container(height=400, border=True):
    with st.columns(3)[1]:
        st.image(
            "assets/image_classification/blood_cancer_images/bloodCancer.keras.svg",
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
The trained models (plural because the same model was trained in both Keras and PyTorch) for this project can be can be 
found on my [Kaggle](https://www.kaggle.com/models/pythonistasamurai/blood-cancer-cells-classification-models).
""")
