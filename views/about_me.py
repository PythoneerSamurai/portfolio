import streamlit as st

from dialogs.dialog import links

RESUME_FILE = "assets/resume.pdf"

col1, col2 = st.columns(2, gap="small", vertical_alignment="center")

with col1:
    st.image(
        image="https://i.postimg.cc/HWr88hPR/profile-pic.png",
        width=250,
    )

with col2:
    st.title("Haroon Rashid", anchor=False)
    st.write(
        "Computer Scientist + A student of Computer Vision."
    )
    subCol1, subCol2 = st.columns(2, gap="small")
    with subCol1:
        with open(RESUME_FILE, "rb") as pdf_file:
            btn=st.download_button(
            label="Download Resume",
            data=pdf_file,
            file_name="resume.pdf",
            mime="application/octet-stream"
        )
    with subCol2:
        if st.button("Links"):
            links()

st.write("\n")
st.subheader("Experience and Qualifications", anchor=False)
st.write("""
   - 5th semester Bachelor's of Computer Science student at the National University of Modern Languages, Islamabad, 
     Pakistan.
   - Close to 3 years of experience in Python and 2D Computer Vision.
   - 20+ high-quality projects made, 150+ models trained, and 3 high-quality datasets annotated.
   - Deep understanding of Computer Vision algorithms and mathematics.
   - Currently, a research scientist in the field of 3D Computer Vision.
   - Excellent solo-developer skills. 
""")

st.write("\n")
st.subheader("Hard Skills", anchor=False)
st.write("""
   - Programming: Python, Java, C++.
   - Development:
        - Python Desktop App Development (TKinter +TTKBootstrap + CustomTkinter)
        - Streamlit Web App Development
   - Computer Vision:
        - Image Classification
        - Object Detection (Simple + OBB)
        - Segmentation (Semantic + Instance)
        - Key Points Regression
        - Model Implementations (i.e, GANs)
   - Advance Computer Vision Skills:
        - Object Tracking
        - Region-Based Object Detection
        - Perspective transformations
        - *and more*
    - Computer Vision Tools and Frameworks:
        - PyTorch3D
        - Keras
        - PyTorch
        - YOLO
        - OpenCV
        - NumPy
        - Supervision
    - Supporting Skills:
        - Mathematics
        - Problem Solving
        - Algorithm Designing
    -  Communication:
        - Fluent in English and Urdu
""")
