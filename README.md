# AI-Powered Workstation PPE & Occupancy Monitoring System



A comprehensive, real-time Computer Vision pipeline designed to monitor industrial workstations. This system utilizes YOLOv8 to track worker occupancy, ensure compliance with Personal Protective Equipment (PPE) guidelines (helmets and gloves), and log violations automatically.

## 🚀 Features

* **Intelligent Occupancy Tracking:** Uses YOLOv8 and BoT-SORT to track individuals across predefined Regions of Interest (ROIs). Alerts if a station is unattended or over-occupied based on configurable time thresholds.
* **Helmet Detection:** Integrates a custom-trained YOLOv8 model to verify if workers are wearing safety helmets within their designated workstations.
* **Glove Detection (Pose-Based):** Utilizes YOLOv8-pose to track skeletal landmarks (specifically the elbow and wrist). It projects a sampling region over the hand area and uses a configurable brightness heuristic to determine glove presence.
* **Automated Violation Logging:** Generates comprehensive CSV reports detailing compliance intervals, unattended stations, and specific PPE violations over time.
* **Dynamic Visual Feedback:** Renders on-screen bounding boxes, ROI polygons, and color-coded status texts (e.g., warnings for missing equipment or over-occupancy) directly onto the output video.

---

## 🧠 Models Used

This repository relies on three specific YOLOv8 models to function. **Note: These model weights are not included in this repository and must be downloaded or trained separately.**

1.  **Person Tracking (`yolov8s.pt`)**
    * **Type:** Standard YOLOv8 Small model.
    * **Purpose:** Used by the `PersonOccupancyModule` to detect individuals (Class 0) and track them across frames to calculate station occupancy duration.
    * **Source:** Automatically downloaded via the `ultralytics` package upon first run, or available at the [Ultralytics GitHub](https://github.com/ultralytics/ultralytics).

2.  **Glove Detection via Pose (`yolov8s-pose.pt`)**
    * **Type:** YOLOv8 Small Pose Estimation model.
    * **Purpose:** Used by the `GloveDetector`. It identifies the worker's arm keypoints to map out the physical hand area, evaluating local pixel intensity to deduce if a glove is being worn.
    * **Source:** Available via Ultralytics.

3.  **Helmet Detection (`hemletYOLOV8_100epochs.pt`)**
    * **Type:** Custom-trained YOLOv8 model.
    * **Purpose:** Used by the `HelmetDetector` to identify safety helmets. The system computes the Intersection over Union (IoU) between detected helmets and the upper 30% of a tracked worker's bounding box.
    * **Source:** Download this specific custom weights file from: [meryemsakin/helmet-detection-yolov8](https://github.com/meryemsakin/helmet-detection-yolov8/tree/main/models).

---

## 🛠️ Project Structure

* `main.py`: The core execution script that orchestrates video capture, module initialization, frame processing, and output rendering.
* `ROI.py`: A visual utility script to manually draw and save polygonal workstation bounds to a `.pkl` file.
* `person_occupancy_module.py`: Handles individual tracking, ROI intersection math, and occupancy interval logic.
* `helmet_module.py`: Manages the logic for helmet validation and temporal buffering (requiring continuous positive frames to clear a violation).
* `glove_module.py`: Calculates arm keypoint geometry and samples hand regions for glove detection.
* `violations_logger.py`: Periodically dumps violation states into a structured CSV file.
* `text_enhancement.py`: Custom OpenCV rendering functions for high-visibility UI overlays.

---

## ⚙️ Installation & Prerequisites

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/ppe-monitoring-system.git](https://github.com/yourusername/ppe-monitoring-system.git)
    cd ppe-monitoring-system
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    Ensure you have Python 3.8+ installed.
    ```bash
    pip install ultralytics opencv-python numpy pandas shapely
    ```

4.  **Download Models:**
    Place `yolov8s.pt`, `yolov8s-pose.pt`, and `hemletYOLOV8_100epochs.pt` in the root directory of the project (or update the paths in the `CONFIG` dictionary in `main.py`). *Note: Please update the model paths in `main.py` from `yolov8m` to `yolov8s` if you are using the small variants.*

---

## 🚀 Usage Guide

### Step 1: Define Regions of Interest (ROIs)
Before running the main pipeline, you must define the workstations in your video.

1.  Extract a single representative frame from your target video and save it as `worker.png` in the root directory.
2.  Run the ROI script:
    ```bash
    python ROI.py
    ```
3.  **Left-click** to draw polygon points around a workstation.
4.  **Right-click** to close and save the polygon. Enter the workstation name in the terminal when prompted.
5.  Press `q` to quit. This generates a `rois.pkl` file.

### Step 2: Configure Parameters
Open `main.py` and review the `CONFIG` dictionary. Update the following as needed:
* `video_input_path`: Path to your source video.
* `model_path` (for all three modules): Ensure these point to your downloaded `.pt` files.
* **Thresholds:** Adjust `violation_time_sec`, `confidence_threshold`, and `brightness_threshold` to match your environment's lighting and operational rules.

### Step 3: Run the System
Execute the main pipeline:
```bash
python main.py

```

Press `q` at any time while the video is playing to terminate the process safely and flush data to the logs.

---

## 📊 Outputs

Upon completion, the system generates several artifacts:

1. **`output/final_output.mp4`**: The rendered video with bounding boxes, ROI polygons, and text overlays.
2. **`violations_log.csv`**: A periodic dump of active violations per workstation.
3. **`occupancy_intervals_[video_name].csv`**: Logs detailing how many workers were present at specific time intervals.
4. **`workstation_cumulative_time.csv`**: A summary report of the total time each station was occupied.
