from typing import Dict, Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO
import csv
import os

class GloveDetector:

    def __init__(self, config: Dict):
        self.model = YOLO(config['model_path'])
        self.conf_threshold = config['confidence_threshold']
        self.rois = [np.array(roi, dtype=np.int32) for roi in config['rois_data']['rois']]
        self.roi_names = config['rois_data']['names']
        
        self.brightness_threshold = config['brightness_threshold']
        self.frames_for_positive = config['frames_for_positive'] 
        self.frames_for_negative = config['frames_for_negative'] 
        self.hold_duration_frames = config['hold_duration_frames']
        self.violation_time_sec = config.get('violation_time_sec', 10) 
        
        self.region_length = 40
        self.region_width = 30
        self.left_elbow_idx, self.left_wrist_idx = 7, 9
        
        self.person_status: Dict[int, Dict] = {}
        self.no_glove_timers: Dict[int, float] = {}  # {track_id: start_time}
        self.last_logged_time: Dict[int, float] = {} # {track_id: timestamp_of_last_log}

    def _get_initial_status(self) -> Dict:
        return {'status': 'Checking', 'pos_count': 0, 'neg_count': 0, 'hold_count': 0}

    def _sample_hand_area(self, image: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float:
        """Calculates average brightness in a rectangular area extending from the wrist."""
        direction = wrist - elbow
        if np.linalg.norm(direction) == 0: return 0.0
        unit_vec = direction / np.linalg.norm(direction)

        start = wrist
        end = wrist + unit_vec * self.region_length
        perp = np.array([-unit_vec[1], unit_vec[0]])
        p1, p2 = start + perp * (self.region_width / 2), start - perp * (self.region_width / 2)
        p3, p4 = end - perp * (self.region_width / 2), end + perp * (self.region_width / 2)
        polygon = np.array([p1, p2, p3, p4], dtype=np.int32)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray[mask == 255]) if np.any(mask) else 0.0
        return brightness

    def process_frame(self, frame: np.ndarray, persons_data: Dict[int, np.ndarray], current_time: float = None) -> dict:
        active_person_ids = set(persons_data.keys())
        for track_id in list(self.person_status.keys()):
            if track_id not in active_person_ids:
                del self.person_status[track_id]
                self.no_glove_timers.pop(track_id, None)
                self.last_logged_time.pop(track_id, None)
        if not persons_data:
            return {}
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        keypoints_data = results[0].keypoints.data.cpu().numpy()
        roi_polygon = self.rois[0]  # Only monitor work station 1
        station_name_to_log = self.roi_names[0] if self.roi_names else "WORK STATION 1"
        def bbox_roi_intersection_area(bbox, roi_poly):
            x1, y1, x2, y2 = map(int, bbox)
            bbox_poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            roi_cnt = roi_poly.reshape(-1, 2)
            inter = cv2.intersectConvexConvex(bbox_poly.astype(np.float32), roi_cnt.astype(np.float32))
            return inter[0] if inter[0] > 0 else 0
        max_area = 0
        selected_track_id = None
        selected_bbox = None
        for track_id, bbox in persons_data.items():
            area = bbox_roi_intersection_area(bbox, roi_polygon)
            if area > max_area:
                max_area = area
                selected_track_id = track_id
                selected_bbox = bbox
        glove_status = {}
        if selected_track_id is None:
            return glove_status
        for pose in keypoints_data:
            wrist_pt = pose[self.left_wrist_idx][:2]
            wrist_conf = pose[self.left_wrist_idx][2]
            track_id = selected_track_id
            bbox = selected_bbox
            x1, y1, x2, y2 = map(int, bbox)
            if wrist_conf > self.conf_threshold and (x1 < wrist_pt[0] < x2 and y1 < wrist_pt[1] < y2):
                if track_id not in self.person_status:
                    self.person_status[track_id] = self._get_initial_status()
                status_info = self.person_status[track_id]
                status = status_info['status']
                if status == 'Glove Detected':
                    status_info['hold_count'] += 1
                    if status_info['hold_count'] >= self.hold_duration_frames:
                        status_info['status'] = 'Checking'
                        status_info['pos_count'] = 0
                        status_info['neg_count'] = 0
                        status_info['hold_count'] = 0
                    # Clear violation states while in 'Glove Detected' status
                    status_info['violation'] = False
                    status_info['checking'] = False
                    status_info['violation_time'] = 0.0
                    self.no_glove_timers.pop(track_id, None)
                    self.last_logged_time.pop(track_id, None)
                else:
                    elbow_pt = pose[self.left_elbow_idx][:2]
                    brightness = self._sample_hand_area(frame, elbow_pt, wrist_pt)
                    glove_detected = brightness > self.brightness_threshold
                            
                    if glove_detected:
                        status_info['violation'] = False
                        status_info['checking'] = False
                        status_info['violation_time'] = 0.0
                        self.no_glove_timers.pop(track_id, None)
                        self.last_logged_time.pop(track_id, None)
                                
                        status_info['pos_count'] += 1
                        status_info['neg_count'] = 0  
                                
                        if status_info['pos_count'] >= self.frames_for_positive:
                            status_info['status'] = 'Glove Detected'
                            status_info['hold_count'] = 0
                            
                    else: # Glove not detected
                        status_info['neg_count'] += 1
                        status_info['pos_count'] = 0 

                        if status_info['neg_count'] >= self.frames_for_negative:
                            status_info['status'] = 'Not Detected'
                                    
                            if current_time is not None:
                                if track_id not in self.no_glove_timers:
                                    self.no_glove_timers[track_id] = current_time
                                        
                                time_since_start = current_time - self.no_glove_timers[track_id]
                                status_info['violation_time'] = time_since_start
                                        
                                # We must also check last_logged_time for subsequent violations
                                last_log_time = self.last_logged_time.get(track_id)
                                time_since_last_event = current_time - (last_log_time or self.no_glove_timers[track_id])

                                if time_since_last_event >= self.violation_time_sec:
                                    status_info['violation'] = True
                                    status_info['checking'] = False
                                else:
                                    status_info['violation'] = False
                                    status_info['checking'] = True
                is_glove_detected = (status_info.get('status') == 'Glove Detected' or 
                                   status_info.get('pos_count', 0) > 0)
                
                if is_glove_detected:
                    status_info['violation'] = False
                    status_info['checking'] = False
                    status_info['violation_time'] = 0.0
                
                glove_status[track_id] = {
                    'violation': status_info.get('violation', False),
                    'checking': status_info.get('checking', False),
                    'violation_time': status_info.get('violation_time', 0.0),
                    'glove_detected': is_glove_detected
                }
                break  # Only check one person
        return glove_status

    def _seconds_to_hms(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _log_violation(self, timestamp: float, track_id: int, workstation: str):
        pass

    def cleanup(self):
        pass