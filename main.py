import streamlit as st

IMAGE_CLASSIFICATION_DIRECTORY_PATH = "views/projects/image_classification"
OBJECT_DETECTION_DIRECTORY_PATH = "views/projects/object detection"
SEGMENTATION_DIRECTORY_PATH = "views/projects/segmentation"
KEYPOINTS_DETECTION_PATH = "views/projects/keypoints_detection"

about_page = st.Page(
    page="views/about_me.py",
    title="about me",
    icon=":material/account_circle:",
    default=True,
)
# ----- IMAGE CLASSIFICATION PROJECT PAGES -----
alzheimer_classification_page = st.Page(
    page=f"{IMAGE_CLASSIFICATION_DIRECTORY_PATH}/alzheimer_classification.py",
    title="alzheimer classification",
    icon=":material/medication:",
)
anomaly_classification_page = st.Page(
    page=f"{IMAGE_CLASSIFICATION_DIRECTORY_PATH}/anomaly_classification.py",
    title="anomaly classification",
    icon=":material/policy:",
)
blood_cancer_classification_page = st.Page(
    page=f"{IMAGE_CLASSIFICATION_DIRECTORY_PATH}/blood_cancer_classification.py",
    title="blood cancer classification",
    icon=":material/hematology:",
)
brain_tumor_classification_page = st.Page(
    page=f"{IMAGE_CLASSIFICATION_DIRECTORY_PATH}/brain_tumor_classification.py",
    title="brain tumor classification",
    icon=":material/neurology:",
)
colon_histopathological_images_classification_page = st.Page(
    page=f"{IMAGE_CLASSIFICATION_DIRECTORY_PATH}/colon_histopathological_images_classification.py",
    title="colon histopathological images classification",
    icon=":material/hotel:",
)
covid_ct_scan_classification_page = st.Page(
    page=f"{IMAGE_CLASSIFICATION_DIRECTORY_PATH}/covid_ct_scan_classification.py",
    title="covid ct scans classification",
    icon=":material/coronavirus:",
)
geospatial_images_classification_page = st.Page(
    page=f"{IMAGE_CLASSIFICATION_DIRECTORY_PATH}/geospatial_images_classification.py",
    title="geospatial images classification",
    icon=":material/satellite_alt:",
)
multi_region_bone_fracture_classification_page = st.Page(
    page=f"{IMAGE_CLASSIFICATION_DIRECTORY_PATH}/multi_region_bone_fracture_classification.py",
    title="multi-region bone fracture classification",
    icon=":material/rheumatology:",
)
pneumonia_classification_page = st.Page(
    page=f"{IMAGE_CLASSIFICATION_DIRECTORY_PATH}/pneumonia_classification.py",
    title="pnuemonia classification",
    icon=":material/pulmonology:",
)
synthetic_real_images_classification_page = st.Page(
    page=f"{IMAGE_CLASSIFICATION_DIRECTORY_PATH}/synthetic_real_images_classification.py",
    title="synthetic and real images classification",
    icon=":material/imagesmode:",
)
# ----- OBJECT DETECTION PROJECT PAGES -----
advance_tennis_analysis_page = st.Page(
    page=f"{OBJECT_DETECTION_DIRECTORY_PATH}/advance_tennis_analysis.py",
    title="advance tennis analysis",
    icon=":material/sports_tennis:",
)
dwell_time_analysis_page = st.Page(
    page=f"{OBJECT_DETECTION_DIRECTORY_PATH}/dwell_time_analysis.py",
    title="dwell time analysis",
    icon=":material/timer:",
)
license_plate_recognition_page = st.Page(
    page=f"{OBJECT_DETECTION_DIRECTORY_PATH}/license_plate_recognition.py",
    title="license plate recognition",
    icon=":material/directions_car:",
)
traffic_flow_analysis_page = st.Page(
    page=f"{OBJECT_DETECTION_DIRECTORY_PATH}/traffic_flow_analysis.py",
    title="traffic flow analysis",
    icon=":material/route:",
)
vehicle_speed_estimation_page = st.Page(
    page=f"{OBJECT_DETECTION_DIRECTORY_PATH}/vehicle_speed_estimation.py",
    title="vehicle speed estimation",
    icon=":material/speed:",
)
# ----- SEGMENTATION PROJECT PAGES -----
background_editor_page = st.Page(
    page=f"{SEGMENTATION_DIRECTORY_PATH}/background_editor.py",
    title="background editor",
    icon=":material/background_replace:",
)
destruction_estimation_page = st.Page(
    page=f"{SEGMENTATION_DIRECTORY_PATH}/destruction_estimation.py",
    title="destruction estimation",
    icon=":material/destruction:",
)
leaf_disease_analysis_page = st.Page(
    page=f"{SEGMENTATION_DIRECTORY_PATH}/leaf_disease_analysis.py",
    title="leaf disease analysis",
    icon=":material/psychiatry:",
)
road_area_estimation_page = st.Page(
    page=f"{SEGMENTATION_DIRECTORY_PATH}/road_area_estimation.py",
    title="road area estimation",
    icon=":material/road:",
)
skin_cancer_analysis_page = st.Page(
    page=f"{SEGMENTATION_DIRECTORY_PATH}/skin_cancer_analysis.py",
    title="skin cancer analysis",
    icon=":material/microbiology:",
)
# -------- Navigation --------
pg = st.navigation({
    "Info": [about_page],
    "Projects": [],
    "Image Classification": [
        alzheimer_classification_page,
        anomaly_classification_page,
        blood_cancer_classification_page,
        brain_tumor_classification_page,
        colon_histopathological_images_classification_page,
        covid_ct_scan_classification_page,
        geospatial_images_classification_page,
        multi_region_bone_fracture_classification_page,
        pneumonia_classification_page,
        synthetic_real_images_classification_page
    ],
    "Object Detection": [
        advance_tennis_analysis_page,
        dwell_time_analysis_page,
        license_plate_recognition_page,
        traffic_flow_analysis_page,
        vehicle_speed_estimation_page
    ],
    "Segmentation": [
        background_editor_page,
        destruction_estimation_page,
        leaf_disease_analysis_page,
        road_area_estimation_page,
        skin_cancer_analysis_page
    ]
})

st.logo(image="assets/logo.png")
st.sidebar.text(body="Made with ❤️ by Haroon")

pg.run()
