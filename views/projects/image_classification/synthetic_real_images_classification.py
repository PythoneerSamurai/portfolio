import streamlit as st

st.title("Synthetic and Real Images Classification")

st.write("\n")
st.subheader("Project Overview", divider=True)
st.write("""
The 21st century has marked great advancements in the fields of technology, most notably in the field of AI. AI has 
become an almost integral part of human life. Though AI has bore great fruits, still it has also resulted in the 
spreading of a lot of misinformation, primarily in the form of fake images or videos which are produced using 
generative AI.

In this project, I have trained a model that can accurately classify images to be either real or fake. **This project
has been implemented in both Keras and PyTorch**.
""")

st.write("\n")
st.subheader("Training Pipeline", divider=True)
st.write("#### Training Data")
st.write("""
The dataset used for training the model is the 
[CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images),
dataset provided by **"Jordan J. Bird"** on Kaggle. 

This dataset has two classes for classification, those being:

1. Fake
2. Real

The training subset had a total of 100k images, 50k for each class, whereas the test subset had 20k images, 10k for each 
class.

The dataset was already structured to have a train and test split. In this project, I have used the test split for validation. A batch size of 32 was used for training.

Before being fed to the model for training, both the training and validation splits were preprocessed to have a size of 
32 pixels in width and height, rescaled to grayscale, and in the case of the PyTorch implementation of this project, 
converted to torch tensors.
""")

st.write("#### Model Implementation")
st.write("""
The architecture for this model can be found on my 
[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/image-classification/cifake-synthetic-ai-and-real-images-discrimination-classification).

The model is a convolutional neural network, consisting of 2 2D-Convolutional layers, 2 2D-MaxPooling layers,
2 2D-Dropout layers, 2 Rectified Linear Unit layers, 3 linear layers, and 2 2D-BatchNorm layers.
""")

with st.container(height=400, border=True):
    with st.columns(3)[1]:
        st.image("assets/image_classification/cifake_images/cifake.keras.svg", caption="Model Architecture", width=200)

st.write("#### Training Analysis")
st.write("""
The model was trained for at most 30 epochs, with **BinaryCrossEntropy** being used as the loss function in both the
Keras and PyTorch implementations. Adam was used as the optimizer, with default hyper-parameter values, in both 
implementations.

Two callbacks were used for training, those being EarlyStopping and ModelCheckpoint (ModelCheckpoint was only used on 
the Keras implementation). The criterion for early stopping was set to training loss (training accuracy was used as the
criterion for the PyTorch implementation). If the training loss did not decrease for 5 consecutive epochs 
(or the training accuracy did not increase for 5 consecutive epochs, in the case of PyTorch implementation), the 
training would be terminated. As far as model checkpointing is concerned, after each epoch, the model with the best 
training loss (lowest training loss) was saved.

The trained model gave an accuracy higher than 96% (in the case of PyTorch implementation the accuracy was calculated 
using the **Accuracy class** provided in the **torchmetrics** package).
""")

st.write("#### Model Links")
st.write("""
The trained models (plural because the same model was trained in both Keras and PyTorch) for this project can be 
found on my [Kaggle](https://www.kaggle.com/models/pythonistasamurai/cifake-ai-and-real-images-classification).
""")
