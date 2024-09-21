import streamlit as st

st.title("GeoSpatial Images Classification")

st.write("\n")
st.subheader("Project Overview", divider=True)
st.write("""
In this project I have trained a model for classifying the use of land based upon the satellite images received from
the Sentinal-2 satellite. **This project has been implemented in both Keras and PyTorch**.
""")

st.write("\n")
st.subheader("Training Pipeline", divider=True)
st.write("#### Training Data")
st.write("""
The dataset used for training the model is the 
[EuroSat Dataset](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset),
provided by **"Gotam Dahiya and ArjunDebnath"** on Kaggle. 

This dataset has a total of ten classes, representing various land features and water bodies, for classification.
The classes are listed below.

1. AnnualCrop
2. Forest
3. Herbaceous Vegetation
4. Highway
5. Industrial
6. Pasture
7. PermanentCrop
8. Residential
9. River
10. Sea or Lake
""")

with st.container(height=400, border=True):
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.image(
            "assets/image_classification/geospatial_images/annual_crop.jpg",
            width=200,
            caption="Annual Crop Sample"
        )
        st.image("assets/image_classification/geospatial_images/forest.jpg", width=200, caption="Forest Sample")
        st.image(
            "assets/image_classification/geospatial_images/herbaceous_vegetation.jpg",
            width=200,
            caption="Herbaceous Vegetation Sample")
        st.image("assets/image_classification/geospatial_images/highway.jpg", width=200, caption="Highway Sample")
        st.image("assets/image_classification/geospatial_images/industrial.jpg", width=200, caption="Industrial Sample")
    with col2:
        st.image("assets/image_classification/geospatial_images/pasture.jpg", width=200, caption="Pasture Sample")
        st.image(
            "assets/image_classification/geospatial_images/permanent_crop.jpg",
            width=200,
            caption="Permanent Crop Sample"
        )
        st.image(
            "assets/image_classification/geospatial_images/residential.jpg",
            width=200,
            caption="Residential Sample"
        )
        st.image("assets/image_classification/geospatial_images/river.jpg", width=200, caption="River Sample")
        st.image("assets/image_classification/geospatial_images/sealake.jpg", width=200, caption="Sea or Lake Sample")

st.write("""
Since the dataset was not structured to have a validation and training split, therefore 80% of the total images were
used for training and 20% were used for validation. A batch size of 64 was used for training.

The dataset was standardized and came with images having width and height of 64 pixels, therefore image resizing was 
not carried out in this project. The images were rescaled to grayscale, and in case of the PyTorch implementation of 
this project, converted to torch tensors.
""")

st.write("#### Model Implementation")
st.write("""
The architecture for this model can be found on my 
[GitHub](https://github.com/PythoneerSamurai/computer-vision-projects/tree/master/image-classification/eurosat-geospatial-images-classification).

The model is a deep convolutional neural network, consisting of 8 2D-Convolutional layers, 4 2D-MaxPooling layers,
4 2D-Dropout layers, 8 Rectified Linear Unit layers, 4 linear layers, and 4 2D-BatchNorm layers.
""")

with st.container(height=400, border=True):
    with st.columns(3)[1]:
        st.image(
            "assets/image_classification/geospatial_images/eurosat.keras.svg",
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
decrease for 7 consecutive epochs, the training would be terminated. As far as model checkpointing is concerned, 
after each epoch, the model with the best training loss (lowest training loss) was saved.

The trained model gave an accuracy higher than 96% (in case of PyTorch implementation the accuracy was calculated 
using the **Accuracy class** provided in the **torchmetrics** package).
""")

st.write("#### Model Links")
st.write("""
The trained models (plural because the same model was trained in both Keras and PyTorch) for this project can be 
found on my [Kaggle](https://www.kaggle.com/models/pythonistasamurai/eurosat-geospatial-images-classification-models).
""")
