import streamlit as st

from dialogs.dialog import links

RESUME_FILE = "assets/resume.pdf"
with open(RESUME_FILE, "rb") as pdf_file:
    PDFByte = pdf_file.read()

col1, col2 = st.columns(2, gap="small", vertical_alignment="center")

with col1:
    st.image(
        image="./assets/profile-pic.png",
        width=250,
    )

with col2:
    st.title("Haroon Rashid", anchor=False)
    st.write(
        "Computer Vision Engineer, providing advanced visual perception to your cameras."
    )
    subCol1, subCol2 = st.columns(2, gap="small")
    with subCol1:
        st.download_button(
            label="Download Resume",
            data=PDFByte,
            file_name="resume.pdf",
            mime="application/octet-stream",
        )
    with subCol2:
        if st.button("Links"):
            links()

st.write("\n")
st.subheader("Experience and Qualifications", anchor=False)
st.write("""
   - 4th semester Bachelor's of Computer Science student at the National University of Modern Languages, Islamabad, 
     Pakistan.
   - Almost 2 years of self-gained experience in Python and Computer Vision.
   - Deep understanding of Computer Vision algorithms and mathematics.
   - Excellent solo-developer skills. 
""")

st.write("\n")
st.subheader("Hard Skills", anchor=False)
st.write("""
   - Programming: Python, Java, C++.
   - Development:
        - Python Desktop App Development (ttkbootstrap + customtkinter libraries)
        - Streamlit Web App Development
   - Computer Vision:
        - Image Classification
        - Object Detection (Simple + OBB)
        - Segmentation (Semantic + Instance)
        - Keypoints detection
        - Model Implementations (i.e, GANs)
   - Advance Computer Vision Skills:
        - Object Tracking
        - Region-Based Object Detection
        - Perspective transformations
        - *and more*
    - Computer Vision Tools and Frameworks:
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
