import pickle
import cv2
import datetime
import os

from person_occupancy_module import PersonOccupancyModule
from helmet_module import HelmetDetector
from glove_module import GloveDetector
from violations_logger import ViolationsLogger

CONFIG = {
    'video_input_path': 'Compilation of activity - Made with Clipchamp_1753253111265.mp4',
    'video_output_path': 'output/final_output.mp4',
    'rois_path': 'rois.pkl',

    # logging intervals (in seconds)
    'violations_logging_interval_seconds': 10,
    'occupancy_logging_interval_seconds': 10, 
    'occupancy_recent_window_seconds': 3,       # Worker can be present in last N seconds to count for interval

    'person_occupancy': {
        'model_path': 'yolov8m.pt',
        'tracker_config': 'botsort.yaml',
        'confidence_threshold': 0.3,
        'max_age_frames': 15,
        'vacant_threshold_sec': 2,
        'unattended_violation_sec': 5,
        'over_occupancy_violation_sec': 10,
        'iou_threshold': 0.30,
    },

    'helmet': {
        'model_path': 'hemletYoloV8_100epochs.pt',
        'confidence_threshold': 0.75,
        'iou_threshold': 0.5,
        'violation_time_sec': 5,
        'frames_for_positive': 10,  # Number of frames helmet must be detected continuously
        'frames_for_negative': 3,  # Number of frames helmet must be missing to start violation
    },

    'glove': {
        'model_path': 'yolov8m-pose.pt',
        'confidence_threshold': 0.5,
        'brightness_threshold': 150,
        'frames_for_positive': 3,
        'frames_for_negative': 15,
        'hold_duration_frames': 90,
        'violation_time_sec': 5,
    }
}

def main():
    try:
        with open(CONFIG['rois_path'], "rb") as f:
            rois_data = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ Error: ROI file not found at {CONFIG['rois_path']}")
        return

    person_config = {
        **CONFIG['person_occupancy'],
        'rois_data': rois_data,
        'video_output_path': CONFIG['video_output_path'],
        'occupancy_logging_interval_seconds': CONFIG['occupancy_logging_interval_seconds'],  
        'occupancy_recent_window_seconds': CONFIG['occupancy_recent_window_seconds'],       
    }
    person_occupancy_detector = PersonOccupancyModule(person_config)

    helmet_config = {**CONFIG['helmet'], 'rois_data': rois_data}
    helmet_detector = HelmetDetector(helmet_config)

    glove_config = {**CONFIG['glove'], 'rois_data': rois_data}
    glove_detector = GloveDetector(glove_config)

    violations_logger_config = {
        'interval_seconds': CONFIG['violations_logging_interval_seconds'],
        'video_filename': CONFIG['video_input_path'],
        'start_datetime': datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0), 
        'output_path': 'violations_log.csv',
        'unattended_threshold': CONFIG['person_occupancy']['unattended_violation_sec'],
        'max_workers_threshold': CONFIG['person_occupancy']['over_occupancy_violation_sec']
    }
    violations_logger = ViolationsLogger(violations_logger_config)

    cap = cv2.VideoCapture(CONFIG['video_input_path'])
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {CONFIG['video_input_path']}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(CONFIG['video_output_path'], cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print("✅ All modules initialized. Starting video processing... Press 'q' to quit.")

    last_frame_time = 0.0
    video_filename = CONFIG['video_input_path']
    start_datetime = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)  
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("🏁 End of video stream.")
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                last_frame_time = total_frames / fps
            else:
                last_frame_time = 0.0
            break

        last_frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Process frame for person tracking, occupancy, and violations
        processed_frame, persons_data = person_occupancy_detector.process_frame(
            frame, last_frame_time, video_filename=video_filename, start_datetime=start_datetime)

        # Get helmet and glove violation status dicts
        helmet_status = helmet_detector.process_frame(frame, persons_data, last_frame_time)
        glove_status = glove_detector.process_frame(frame, persons_data, last_frame_time)

        # Update violations logger with helmet and glove violations
        for station in person_occupancy_detector.station_states:
            station_violations = {
                'helmet': False,
                'glove': False
            }
            
            # Check for helmet violations in this station
            for track_id in station.current_workers:
                if track_id in helmet_status and helmet_status[track_id].get('violation', False):
                    station_violations['helmet'] = True
                if track_id in glove_status and glove_status[track_id].get('violation', False):
                    station_violations['glove'] = True
            
            violations_logger.update_violations(station.name, station_violations)

        # Process violations logging at intervals
        violations_logger.process_frame(last_frame_time, person_occupancy_detector.station_states)

        # Update the processed frame with helmet and glove overlays
        processed_frame = person_occupancy_detector._draw_overlays(
            processed_frame, helmet_status, glove_status, last_frame_time)

        cv2.imshow('Combined PPE Detection System', processed_frame)
        out.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    person_occupancy_detector.cleanup(last_frame_time)
    glove_detector.cleanup()
    violations_logger.cleanup()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Processing complete. Output saved to {CONFIG['video_output_path']}")
    print(f"✅ Violations log saved to violations_log.csv")
    print(f"✅ Occupancy intervals log saved to occupancy_intervals_{os.path.splitext(os.path.basename(CONFIG['video_input_path']))[0]}.csv")
    print(f"✅ Cumulative time summary saved to workstation_cumulative_time.csv")

if __name__ == "__main__":
    main()