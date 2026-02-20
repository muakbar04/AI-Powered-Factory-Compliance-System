from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
import os
import csv

def is_person_in_roi(person_bbox: np.ndarray, rois: List[np.ndarray]) -> bool:
    """Check if the bottom-center of a person's bbox is inside any ROI."""
    px = int(person_bbox[0] + (person_bbox[2] - person_bbox[0]) * 0.5)
    py = int(person_bbox[3])
    for roi in rois:
        if cv2.pointPolygonTest(roi, (px, py), False) >= 0:
            return True
    return False

def check_helmet_on_person(person_bbox: np.ndarray, helmet_bboxes: List[np.ndarray]) -> bool:
    """Check if any helmet bbox significantly overlaps with the person's head area."""
    px1, py1, px2, p_y2 = person_bbox
    # Define the head region as the top 30% of the person's bounding box.
    head_region_y2 = py1 + int((p_y2 - py1) * 0.3)

    for h_bbox in helmet_bboxes:
        hx1, hy1, hx2, hy2 = h_bbox
        # Calculate intersection area between the helmet and the person's head region.
        inter_x1 = max(px1, hx1)
        inter_y1 = max(py1, hy1)
        inter_x2 = min(px2, hx2)
        inter_y2 = min(head_region_y2, hy2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        helmet_area = (hx2 - hx1) * (hy2 - hy1)

        if helmet_area > 0 and (inter_area / helmet_area) > 0.5:
            return True
    return False

class HelmetDetector:
    """Detects helmets, associates them with tracked persons, and flags violations."""
    def __init__(self, config: Dict):
        self.model = YOLO(config['model_path'])
        self.model.fuse() 
        
        self.conf_threshold = config['confidence_threshold']
        self.iou_threshold = config['iou_threshold']
        self.violation_time_sec = config['violation_time_sec']
        self.frames_for_positive = config.get('frames_for_positive', 5)
        self.frames_for_negative = config.get('frames_for_negative', 3)
        
        self.rois = [np.array(roi, dtype=np.int32) for roi in config['rois_data']['rois']]
        self.roi_names = config['rois_data']['names']
        
        self.no_helmet_timers: Dict[int, float] = {}
        self.last_logged_time: Dict[int, float] = {}
        self.person_status: Dict[int, Dict] = {}  # Track person status for frame counting

    def _get_initial_status(self) -> Dict:
        return {
            'status': 'Not Detected', 
            'pos_count': 0, 
            'neg_count': 0,
            'violation': False,
            'checking': True,
            'violation_time': 0.0
        }

    def _bbox_roi_intersection_area(self, bbox, roi_poly):
        x1, y1, x2, y2 = map(int, bbox)
        bbox_poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        roi_cnt = roi_poly.reshape(-1, 2)

        intersection_area, _ = cv2.intersectConvexConvex(bbox_poly.astype(np.float32), roi_cnt.astype(np.float32))
        return intersection_area

    def process_frame(self, frame: np.ndarray, persons_data: Dict[int, np.ndarray], current_time: float) -> dict:
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)[0]
        
        helmet_bboxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]

            if cls_id != 1:
                continue
            helmet_bboxes.append([x1, y1, x2, y2])

        active_person_ids = set(persons_data.keys())
        
        # Clean up inactive persons
        for track_id in list(self.person_status.keys()):
            if track_id not in active_person_ids:
                del self.person_status[track_id]
                self.no_helmet_timers.pop(track_id, None)
                self.last_logged_time.pop(track_id, None)

        # Find the person to check in each ROI
        persons_to_check_ids = set()
        for roi_poly in self.rois:
            max_area = 0
            main_person_track_id = None
            for track_id, person_bbox in persons_data.items():
                area = self._bbox_roi_intersection_area(person_bbox, roi_poly)
                if area > max_area:
                    max_area = area
                    main_person_track_id = track_id
            
            if main_person_track_id is not None:
                persons_to_check_ids.add(main_person_track_id)

        helmet_status = {}
        for track_id, person_bbox in persons_data.items():
            if track_id not in self.person_status:
                self.person_status[track_id] = self._get_initial_status()
            
            status_info = self.person_status[track_id]
            status = status_info['status']
            
            if track_id in persons_to_check_ids:
                has_helmet = check_helmet_on_person(person_bbox, helmet_bboxes)
                
                if has_helmet:
                    status_info['pos_count'] += 1
                    status_info['neg_count'] = 0
                    if status_info['pos_count'] >= self.frames_for_positive:
                        # Helmet confirmed for required frames
                        status_info['status'] = 'Helmet Detected'
                        status_info['violation'] = False
                        status_info['checking'] = False
                        status_info['violation_time'] = 0.0
                        self.no_helmet_timers.pop(track_id, None)
                        self.last_logged_time.pop(track_id, None)
                    else:
                        # Brief helmet detection - don't change state or reset timer
                        # Keep current violation/checking state and timer running
                        # This ensures that brief helmet detections don't prematurely clear violations
                        pass
                else:
                    # No helmet detected
                    status_info['neg_count'] += 1
                    status_info['pos_count'] = 0
                    
                    # Only change states if we've been counting negative frames for the required duration
                    if status_info['neg_count'] >= self.frames_for_negative:
                        status_info['status'] = 'Not Detected'
                        if current_time is not None:
                            if track_id not in self.no_helmet_timers:
                                self.no_helmet_timers[track_id] = current_time
                            time_since_start = current_time - self.no_helmet_timers[track_id]
                            status_info['violation_time'] = time_since_start
                            if time_since_start >= self.violation_time_sec:
                                status_info['violation'] = True
                                status_info['checking'] = False
                            else:
                                status_info['violation'] = False
                                status_info['checking'] = True
                    else:
                        # Still counting negative frames - don't change any states
                        # Keep current violation/checking state and timer running
                        pass
            else:
                # Not the main person in any ROI - don't change states, just clear timers
                self.no_helmet_timers.pop(track_id, None)
                self.last_logged_time.pop(track_id, None)
                # Keep current violation/checking states to avoid state changes
                pass
            
            # Only consider helmet detected if status is confirmed, not during brief detection
            is_helmet_detected = (status_info.get('status') == 'Helmet Detected')
            
            # If helmet is confirmed detected, ensure no violation states
            if status_info.get('status') == 'Helmet Detected':
                status_info['violation'] = False
                status_info['checking'] = False
                status_info['violation_time'] = 0.0
            
            helmet_status[track_id] = {
                'violation': status_info.get('violation', False),
                'checking': status_info.get('checking', False),
                'violation_time': status_info.get('violation_time', 0.0),
                'helmet_detected': is_helmet_detected
            }
        
        return helmet_status

    def _get_station_name(self, person_bbox: np.ndarray) -> str:
        """Finds the name of the ROI the person is in."""
        for i, roi in enumerate(self.rois):
            if is_person_in_roi(person_bbox, [roi]):
                return self.roi_names[i]
        return "Unknown Station"

    def _draw_violation(self, frame: np.ndarray, station_name: str):
        pass

    def _seconds_to_hms(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _log_violation(self, timestamp: float, track_id: int, workstation: str):
        pass