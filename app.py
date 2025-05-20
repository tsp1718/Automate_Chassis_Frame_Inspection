import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# Corrected part nomenclature mapping
NOMENCLATURE = {
    "07": "IA348549", "08": "IA349043", "09": "IA353945", "25": "IC318330", "34": "IC379896", "35": "IC382160", "40": "IC391070",
    "41": "IC391071", "42": "IC392312", "43": "IC392313", "47": "IC399170", "64": "IC411673", "74": "IC518851", "102": "IC800958",
    "108": "IC801489", "117": "ID366902", "118": "ID369862", "120": "ID602785", "121": "ID602786", "122": "ID603820", "123": "ID606124",
    "130": "IE303129"
}

# Fixed list of 12 distinct colors in BGR format (excluding red)
DISTINCT_COLORS = [
    (0, 255, 0),       # Green
    (255, 0, 0),       # Blue
    (0, 255, 255),     # Yellow
    (255, 0, 255),     # Magenta
    (0, 128, 255),     # Orange
    (255, 255, 0),     # Cyan
    (128, 128, 0),     # bluish-green
    (0, 255, 128),     # Light green
    (128, 255, 0),     # Turquoise
    (255, 128, 0),     # Light blue
    (0, 128, 128),     # Olive
    (128, 128, 255),   # Light purple
]

# Create color map using the fixed colors cyclically
COLOR_MAP = {}
color_idx = 0
for part in NOMENCLATURE.keys():
    COLOR_MAP[part] = DISTINCT_COLORS[color_idx % len(DISTINCT_COLORS)]
    color_idx += 1

EXPECTED = {
    "left": ["43", "64", "130", "41", "25", "108", "35", "117", "102", "123", "09"],
    "right": ["74", "34", "35", "122", "08", "47", "118", "121", "120", "40", "64", "130", "07"]
}

EXPECTED = {side: [str(int(p)).zfill(2) for p in parts] for side, parts in EXPECTED.items()}

REFERENCE_BBOXES = {
    "left": {
        "35": (656, 371, 700, 426),
        "64": (151, 304, 209, 460),
        "108": (487, 376, 529, 431),
        "123": (940, 522, 973, 653),
        "09": (997, 481, 1060, 630),
        "43": (90, 446, 146, 545),
        "130": (144, 482, 218, 623),
        "102": (847, 498, 889, 651),
        "117": (641, 455, 706, 621),
        "25": (417, 406, 466, 433),
        "41": (243, 371, 371, 461),
        # If you want to handle duplicates, you can keep only one or average them
    },
    "right": {
        "40": (959, 356, 1128, 505),
        "74": (151, 370, 379, 519),
        "130": (1153, 502, 1265, 680),
        "118": (791, 472, 855, 563),
        "08": (716, 500, 763, 571),
        "64": (1154, 266, 1241, 504),
        "35": (559, 375, 608, 454),
        "34": (388, 490, 464, 655),
        "07": (804, 564, 869, 653),
        "47": (765, 371, 815, 450),
        "122": (603, 512, 640, 664),
        "121": (731, 553, 797, 675),
        "120": (870, 558, 900, 636),
    }
}

def normalize_part_number(part):
    return f"{int(part):02d}" if part.isdigit() else part

def load_model():
    # Use st.cache_resource to load the model only once
    @st.cache_resource
    def get_model():
        return YOLO("./Nano_1k.pt")
    return get_model()

def preprocess_image(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def detect_side(detected_classes):
    detected_classes = [normalize_part_number(p) for p in detected_classes]
    left_markers, right_markers = {"09", "123", "102"}, {"07", "08", "121"}
    left_score = sum(cls in left_markers for cls in detected_classes)
    right_score = sum(cls in right_markers for cls in detected_classes)
    return "LEFT" if left_score > right_score else "RIGHT"

def detect_objects(model, image):
    results, detected = model(image), []
    detected_parts = []
    for result in results:
        for box in result.boxes:
            cls = normalize_part_number(model.names[int(box.cls.item())])
            detected_parts.append(cls)
    detected_parts = list(set(detected_parts)) # Unique detected parts

    # Assign a unique color to each detected part for this image
    local_color_map = {}
    for idx, part in enumerate(detected_parts):
        local_color_map[part] = DISTINCT_COLORS[idx % len(DISTINCT_COLORS)]

    # Draw boxes for detected parts
    for result in results:
        for box in result.boxes:
            cls = normalize_part_number(model.names[int(box.cls.item())])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = local_color_map.get(cls, (0, 255, 0))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            cv2.putText(image, f"{cls} ({NOMENCLATURE.get(cls, 'Unknown')})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            detected.append(cls)
    return image, list(set(detected))


def show_chassis_inspector():
    st.title("Automotive Chassis Component Inspector")
    st.markdown("<p style='font-size:18px; font-weight:bold;'>FRAME PHANTOM PRO6048T 6785 WB CBC PRM BS6 (FR6319N)</p>", unsafe_allow_html=True)

    # Add instructions
    st.markdown("""
    ### 📋 Instructions:
    1. Upload a clear image of the chassis side (left or right)
    2. Ensure the image is well-lit and shows the complete chassis side
    3. The system will automatically:
        - Detect the chassis side (left/right)
        - Identify all visible components
        - Highlight any missing components in red
        - Show component details in the tables below
    """)

    uploaded_file = st.file_uploader("Upload Chassis Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # Load model only when a file is uploaded
        model = load_model()
        img = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Original Image")
            st.image(img, width=400, use_container_width=True)

        img_cv = preprocess_image(img)
        processed_img, detected = detect_objects(model, img_cv)
        detected = [normalize_part_number(p) for p in detected]

        if len(detected) < 2: # Adjusted condition for a potentially invalid image
            st.error("Could not detect enough key components to determine chassis side. Please upload a clearer image.")
        else:
            chassis_side = detect_side(detected)
            expected, missing = EXPECTED[chassis_side.lower()], list(set(EXPECTED[chassis_side.lower()]) - set(detected))

            # Draw bounding boxes for missing parts
            for miss in missing:
                bbox = REFERENCE_BBOXES.get(chassis_side.lower(), {}).get(miss)
                if bbox:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(processed_img, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red box for missing
                    cv2.putText(processed_img, f"Missing: {miss} ({NOMENCLATURE.get(miss, 'Unknown')})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            # Also display all missing items as a white rectangle with red text at the top left, like main.py
            text_x, text_y = 20, 50
            for i, miss in enumerate(missing):
                (w, h), _ = cv2.getTextSize(f"Missing: {miss} ({NOMENCLATURE.get(miss, 'Unknown')})", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                # Ensure rectangle and text stay within image bounds (simple check)
                rect_x2 = min(text_x + w + 5, processed_img.shape[1])
                rect_y2 = min(text_y + i * 30 + 5, processed_img.shape[0])
                if rect_x2 > text_x and rect_y2 > text_y + i * 30 - 20: # Only draw if valid size
                     cv2.rectangle(processed_img, (text_x - 5, text_y + i * 30 - 20), (rect_x2, rect_y2), (255, 255, 255), -1)
                     cv2.putText(processed_img, f"Missing: {miss} ({NOMENCLATURE.get(miss, 'Unknown')})", (text_x, text_y + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)


            reference_image = f"./reference images/{chassis_side.lower()}_reference.jpg"
            if os.path.exists(reference_image):
                 with col2:
                     st.subheader("Reference Image")
                     st.image(reference_image, width=400, use_container_width=True)
            else:
                 with col2:
                      st.subheader("Reference Image")
                      st.warning(f"Reference image not found: {reference_image}")


            # Display the detected chassis side here
            st.markdown(f"""<h3 style='text-align:center; font-weight:bold;'>IDENTIFIED CHASSIS SIDE: {chassis_side}</h3>""", unsafe_allow_html=True)

            st.subheader("Processed Image")
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)

            col3, col4 = st.columns(2)

            with col3:
                st.error("### Missing Components")
                if missing:
                    missing_data = [(m, NOMENCLATURE.get(m, "Unknown")) for m in missing]
                    st.table({"Part Number": [m[0] for m in missing_data], "Component ID": [m[1] for m in missing_data]})
                else:
                    st.markdown("**No missing components**")

            with col4:
                st.success("### Detected Components")
                if detected:
                    detected_data = [(d, NOMENCLATURE.get(d, "Unknown")) for d in detected]
                    st.table({"Part Number": [d[0] for d in detected_data], "Component ID": [d[1] for d in detected_data]})
                else:
                    st.markdown("**No detected components**")

def main():
    st.set_page_config(page_title="Chassis Inspector", layout="wide", page_icon="🔧")
    with st.sidebar:
        # Only the Chassis Inspector is available now
        st.title("Chassis Inspector")
        st.markdown("""
        ## 🛠️ About
        - Automatic side detection
        - Missing parts identification
        - Detailed component nomenclature

        ## 📸 How to Use
        1. Upload a chassis image
        2. View results
        """)
    # Directly show the Chassis Inspector page
    show_chassis_inspector()

if __name__ == "__main__":
    main()