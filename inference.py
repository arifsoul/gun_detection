import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
from pathlib import Path

# Page Config
st.set_page_config(
    page_title="Gun Detection System",
    page_icon="🔫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
RUNS_DIR = NOTEBOOKS_DIR / "runs" / "detect"
TEST_VIDEOS_DIR = (
    PROJECT_ROOT / "BS Interview CV Test_ Weapon Object Detection" / "test videos"
)

# CSS for styling
st.markdown(
    """
<style>
    .reportview-container {
        background: #000000;
    }
    .sidebar .sidebar-content {
        background: #111111;
    }
    .stButton>button {
        color: white;
        background-color: #ff4b4b;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #ffcccc;
        color: #cc0000;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #cc0000;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
        margin-bottom: 10px;
    }
    .safe-box {
        background-color: #ccffcc;
        color: #006600;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #006600;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
        margin-bottom: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🔫 Automatic Gun Detection System")
st.markdown("### Interactive Inference Interface")

# Sidebar - Configuration
st.sidebar.header("Configuration")

# 1. Model Selection
model_options = {
    "Combined (Syn + Real)": RUNS_DIR / "train_combined" / "weights" / "best.pt",
    "Real Data Only": RUNS_DIR / "train_real" / "weights" / "best.pt",
    "Synthetic Data Only": RUNS_DIR / "train_syn" / "weights" / "best.pt",
    "Pre-trained YOLOv8n (COCO)": "yolov8n.pt",  # Fallback
}

selected_model_name = st.sidebar.selectbox("Select Model", list(model_options.keys()))
model_path = model_options[selected_model_name]

# Check if model exists
if isinstance(model_path, Path) and not model_path.exists():
    st.sidebar.warning(
        f"⚠️ Model weights not found at {model_path}.\nUsing default YOLOv8n implementation."
    )
    model_path = "yolov8n.pt"

# 2. Confidence Threshold
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.40, 0.05)

# 3. Input Source
input_source = st.sidebar.radio("Input Source", ["Upload Video", "Select Test Video"])

video_path = None

if input_source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader(
        "Upload a video...", type=["mp4", "avi", "mov"]
    )
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

elif input_source == "Select Test Video":
    if TEST_VIDEOS_DIR.exists():
        videos = list(TEST_VIDEOS_DIR.glob("*.*"))
        video_names = [
            v.name for v in videos if v.suffix.lower() in [".mp4", ".avi", ".mov"]
        ]
        selected_video = st.sidebar.selectbox("Choose a test video", video_names)
        if selected_video:
            video_path = str(TEST_VIDEOS_DIR / selected_video)
    else:
        st.sidebar.error("Test videos directory not found!")


# Main Inference Loop
if video_path:
    st.sidebar.success(f"Loaded: {os.path.basename(video_path)}")

    col1, col2 = st.columns([2, 1])

    with col1:
        st_frame = st.empty()

    with col2:
        st_status = st.empty()
        st_metrics = st.empty()

    if st.sidebar.button("Start Detection"):
        try:
            model = YOLO(str(model_path))
            cap = cv2.VideoCapture(video_path)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Inference
                # Ensure class 0 is gun if using our custom models.
                # If using COCO pretrained, we might detect everything.
                # Our custom models strictly have class 0 = Black Gun.

                results = model.predict(frame, conf=conf_threshold, verbose=False)
                res = results[0]

                # Visualize
                annotated_frame = res.plot()

                # Check for detection
                gun_detected = False
                if len(res.boxes) > 0:
                    gun_detected = True

                # Display Status
                if gun_detected:
                    st_status.markdown(
                        '<div class="warning-box">⚠️ GUN DETECTED!</div>',
                        unsafe_allow_html=True,
                    )
                    # Play alarm sound? (Streamlit doesn't support easy audio trigger without rerun, maybe just visual)
                else:
                    st_status.markdown(
                        '<div class="safe-box">✅ SAFE</div>', unsafe_allow_html=True
                    )

                # Metrics
                st_metrics.json(
                    {
                        "Detections": len(res.boxes),
                        "Max Confidence": f"{res.boxes.conf.max().item():.2f}"
                        if len(res.boxes) > 0
                        else "0.00",
                        "Device": str(res.device),
                    }
                )

                # Streamlit Image
                st_frame.image(
                    annotated_frame, channels="BGR", use_container_width=True
                )

                # Sleep slightly to match simpler playback or just run as fast as possible
                # time.sleep(0.01)

            cap.release()
            st.success("Finished processing video.")

        except Exception as e:
            st.error(f"Error during inference: {e}")

else:
    st.info("Please select or upload a video from the sidebar to begin.")

st.markdown("---")
st.markdown("Developed for Computer Vision Assessment Task - Gun Detection")
