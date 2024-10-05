import streamlit as st

st.title("Alzheimer Classification")

st.write("\n")
st.subheader("Project Overview", divider=True)
st.write("""
According to the **National Institute of Aging**:
> *Alzheimer’s disease is a brain disorder that slowly destroys memory and thinking skills, and eventually, the ability 
to carry out the simplest tasks. In most people with Alzheimer’s, symptoms first appear later in life. Estimates vary, 
but experts suggest that more than 6 million Americans, most of them age 65 or older, may have Alzheimer’s.*

also,
> *Alzheimer’s is currently ranked as the seventh leading cause of death in the United States and is the most common 
cause of dementia among older adults.*

Keeping in view the adverse effects of Alzheimer's disease on the well-being of mankind, scientists have been working on
newer, more accurate, and more efficient ways for the diagnosis of this disease. Now with the inception of Artificial
Intelligence, a new era of early disease diagnosis has begun. Though in infancy, AI has been shown to have promising
diagnosis capabilities in the field of healthcare.

In this project, I have trained a model that can accurately classify the existence or non-existence of Alzheimer's
disease, by consuming MRI images of the human brain. The model can be used by medical practitioners for the diagnosis 
of the disease. **This project has been implemented in both Keras and PyTorch**.
""")

st.write("\n")
st.subheader("Training Pipeline", divider=True)
st.write("#### Training Data")
st.write("""
The dataset used for training the model is the 
[Augmented Alzheimer MRI dataset](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset),
provided by **"URANINJO"** on Kaggle. 

This dataset has four classes for classification, those being:

1. Mild Demented (8960 training images)
2. Moderate Demented (6464 training images)
3. Non Demented (9600 training images)
4. Very Mild Demented (8960 training images)
""")

with st.container(height=600, border=True):
    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.image(
            "assets/image_classification/alzheimer_images/mild_demented.jpg",
            width=250,
            caption="Mild Demented"
        )
        st.image(
            "assets/image_classification/alzheimer_images/moderate_demented.jpg",
            width=250,
            caption="Moderate Demented"
        )
    with col2:
        st.image(
            "assets/image_classification/alzheimer_images/non_demented.jpg",
            width=250,
            caption="Non Demented"
        )
        st.image(
            "assets/image_classification/alzheimer_images/very_mild_demented.jpg",
            width=250,
            caption="Very Mild Demented"
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
I had designed various model architectures for this project, which were then tested. The finalized architecure can be 
found on my 
[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/image-classification/alzheimer-mri-classification).

The finalized model is a deep convolutional neural network, consisting of 8 2D-Convolutional layers, 4 2D-MaxPooling layers,
4 2D-Dropout layers, 8 Rectified Linear Unit layers, 4 linear layers,  and 4 2D-BatchNorm layers.
""")

with st.container(height=400, border=True):
    with st.columns(3)[1]:
        st.image(
            "assets/image_classification/alzheimer_images/alzheimer.keras.svg",
            caption="Model Architecture",
            width=200
        )

st.write("#### Training Analysis")
st.write("""
The model was trained for at most 30 epochs, with **Sparse Categorical CrossEntropy** being used as the loss
function in the Keras implementation and **CrossEntropy** being used as the loss function in the PyTorch
implementation (both losses are essentially the same). Adam was used as the optimizer, with default 
hyper-parameter values, in both implementations.

Two callbacks were used for training, those being EarlyStopping and ModelCheckpoint. The criterion for early stopping 
was set to training loss (training accuracy was used as the criterion for the PyTorch implementation). If the training 
loss did not decrease for 7 consecutive epochs (or the training accuracy did not increase for 5 consecutive epochs, in 
the case of PyTorch implementation), the training would be terminated. As far as model checkpointing is concerned, 
after each epoch, the model with the best training loss (lowest training loss) was saved. ModelCheckpoint callback
was only used in the Keras implementation.

The trained model gave an accuracy higher than 96% (in the case of PyTorch implementation the accuracy was calculated 
using the **Accuracy class** provided in the **torchmetrics** package).
""")

st.write("#### Model Links")
st.write("""
The trained models (plural because the same model was trained in both Keras and PyTorch) for this project can be 
found on my [Kaggle](https://www.kaggle.com/models/pythonistasamurai/alzheimer-mri-classification-models).
""")
