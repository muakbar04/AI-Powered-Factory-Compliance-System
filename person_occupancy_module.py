import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, box
from ultralytics import YOLO
import os
from text_enhancement import draw_violation_text, draw_status_text

PERSON_CLASS_ID = 0

STATION_WORKER_CONFIG = {
    # Example: "Station Name": {"min": 1, "max": 1}
}

class StationState:
    """A data class to encapsulate all state information for a single workstation."""
    def __init__(self, name: str, polygon: np.ndarray, min_workers: int = 1, max_workers: int = 1):
        self.name = name
        self.polygon = polygon
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers: Set[int] = set()
        self.status: str = "Unoccupied"
        self.status_color: Tuple[int, int, int] = (0, 255, 255)  # Yellow
        self.last_entry_time: Optional[float] = None
        self.absence_start_time: Optional[float] = None
        self.over_occupancy_start_time: Optional[float] = None
        self.violation_logged_time: Optional[float] = None # Tracks when the last violation was logged
        self.cumulative_seconds: float = 0.0
        
        # Timers for state changes 
        self.last_entry_time: Optional[float] = None
        self.absence_start_time: Optional[float] = None
        self.over_occupancy_start_time: Optional[float] = None
        
        self.occupancy_history: List[bool] = []  # Track occupancy for last 5 frames
        self.last_entry_time: Optional[float] = None  # When current occupancy period started

class PersonOccupancyModule:
    def __init__(self, config: Dict):
        """Initializes the monitor with a configuration dictionary."""
        self.config = config
        self._setup_paths()
        self._setup_logging()

        self.model = YOLO(config['model_path'])
        self.station_states = self._load_and_initialize_stations()
        
        # Timed Violation Thresholds
        self.vacant_threshold_sec = config['vacant_threshold_sec']
        self.unattended_violation_sec = config['unattended_violation_sec']
        self.over_occupancy_violation_sec = config['over_occupancy_violation_sec']
        self.iou_threshold = config['iou_threshold']
        self.max_age_frames = config['max_age_frames']

        # State Management
        self.frame_number = 0
        self.track_history: Dict[int, int] = {}
        self.track_bboxes: Dict[int, np.ndarray] = {}
        self.track_last_bbox: Dict[int, np.ndarray] = {}  # track_id -> last known bounding box
        self.track_last_seen: Dict[int, float] = {}  # track_id -> last seen time (seconds)
        self.station_worker_history: Dict[str, Dict[int, float]] = {}  # station_name -> {track_id -> last_seen_in_station_time}
        logging.info("PersonOccupancyModule initialized successfully with timed violation logic.")

        self.summary_path = 'workstation_cumulative_time.csv'
        self.occupancy_interval_sec = config.get('occupancy_logging_interval_seconds', 10)
        self.occupancy_recent_window_sec = config.get('occupancy_recent_window_seconds', 3)
        self.cumulative_window_frames = 20  # Sliding window size for cumulative time calculation
        self.last_occupancy_log_time = 0.0
        self.debug_cumulative_time = False  # enable or disable debug logging for cumulative time
        self.occupancy_log_path = None  
        self.occupancy_log_writer = None
        self.occupancy_log_file = None
        self.occupancy_log_initialized = False
        self.occupancy_log_interval_counter = 1

    def process_frame(self, frame: np.ndarray, current_time: float, helmet_status=None, glove_status=None, video_filename=None, start_datetime=None) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Processes a single video frame for person tracking and occupancy analysis.
        Args:
            frame: The input video frame.
            current_time: The current timestamp of the frame in seconds.
        Returns:
            A tuple containing the annotated frame and a dictionary of active tracks {track_id: bbox}.
        """
        self.frame_number += 1
        
        results = self.model.track(
            source=frame, tracker=self.config['tracker_config'], persist=True,
            classes=[PERSON_CLASS_ID], conf=self.config['confidence_threshold'], 
            iou=self.config.get('nms_iou_threshold', 0.5), # NMS IoU, not for ROI check
            verbose=False
        )

        active_track_ids = set()
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            active_track_ids.update(track_ids)
            for box, track_id in zip(boxes, track_ids):
                self.track_bboxes[track_id] = box
                self.track_last_bbox[track_id] = box  # Store the last known bounding box
                self.track_history[track_id] = self.frame_number

        # Prune old tracks
        stale_tracks = [
            tid for tid, last_seen in self.track_history.items()
            if self.frame_number - last_seen > self.max_age_frames
        ]
        for tid in stale_tracks:
            self.track_history.pop(tid, None)
            self.track_bboxes.pop(tid, None)
            self.track_last_bbox.pop(tid, None)  # Also remove from last bbox history
        
        # Include all tracks not older than max_age_frames
        valid_tracks = {
            tid: bbox for tid, bbox in self.track_bboxes.items()
            if self.frame_number - self.track_history.get(tid, -9999) <= self.max_age_frames
        }
        self._update_station_states(valid_tracks, current_time)

        # Update track_last_seen for all active tracks
        for tid in valid_tracks:
            self.track_last_seen[tid] = current_time

        annotated_frame = self._draw_overlays(frame.copy(), helmet_status or {}, glove_status or {}, current_time)

        # Log occupancy at configured interval
        if not self.occupancy_log_initialized and video_filename and start_datetime:
            self._init_occupancy_log(video_filename, start_datetime)
        if self.occupancy_log_initialized:
            if current_time - self.last_occupancy_log_time >= self.occupancy_interval_sec:
                for station in self.station_states:
                    timestamp = self.occupancy_log_start_datetime.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Check if any worker was present in this station in the last N seconds
                    # Count as same person if track IDs change, only count multiple when actually present at interval
                    recent_workers = 0
                    
                    # First, count currently present workers
                    current_worker_count = len(station.current_workers)
                    recent_workers = current_worker_count
                    
                    # If no workers currently present, check recent history
                    if current_worker_count == 0:
                        # Check if any worker was in this station within the recent window
                        for tid in self.station_worker_history.get(station.name, {}):
                            last_seen_in_station = self.station_worker_history[station.name][tid]
                            if (current_time - last_seen_in_station) <= self.occupancy_recent_window_sec:
                                recent_workers = 1  # Count as 1 person (not multiple track IDs)
                                break  # Only need to find one recent worker to count the station as occupied
                    
                    self.occupancy_log_writer.writerow([
                        video_filename,
                        timestamp,
                        station.name,
                        recent_workers,
                        self.occupancy_log_interval_counter
                    ])
                self.last_occupancy_log_time = current_time
                self.occupancy_log_interval_counter += 1

        return annotated_frame, self.track_bboxes

    def _update_station_states(self, track_bboxes: Dict[int, np.ndarray], current_time: float):
        for station in self.station_states:
            roi_poly = Polygon(station.polygon.reshape(-1, 2))
            
            # Check for currently detected workers in station
            worker_ids_in_station = {
                tid for tid, bbox in track_bboxes.items()
                if self._compute_iou(bbox, roi_poly) > self.iou_threshold
            }
            
            # Check for active tracks that are not currently detected but still have valid bounding boxes
            # (track frame is still above them, even if not detected in current frame)
            active_tracks_in_station = {
                tid for tid in self.track_history.keys()
                if (self.frame_number - self.track_history[tid]) <= self.max_age_frames  # Track is still active
                and tid in self.track_last_bbox  # Track has a last known bounding box
                and self._compute_iou(self.track_last_bbox[tid], roi_poly) > self.iou_threshold  # Track is in station
            }
            
            # Consider station occupied if either currently detected OR track is still active
            is_occupied_now = len(worker_ids_in_station) > 0 or len(active_tracks_in_station) > 0
            
            # Debug logging for track-based occupancy
            if self.debug_cumulative_time:
                detected_str = f"Detected: {list(worker_ids_in_station)}" if worker_ids_in_station else "Detected: None"
                active_str = f"Active tracks: {list(active_tracks_in_station)}" if active_tracks_in_station else "Active tracks: None"
                logging.info(f"[{station.name}] {detected_str}, {active_str}, Occupied: {is_occupied_now}")
                
                # Debug IoU for all tracks in this station
                if len(track_bboxes) > 0:
                    for tid, bbox in track_bboxes.items():
                        iou = self._compute_iou(bbox, roi_poly)
                        if iou > 0.1:  # Only log if there's some overlap
                            logging.info(f"[{station.name}] Track {tid} IoU: {iou:.3f} (threshold: {self.iou_threshold})")

            # Update sliding n-frame occupancy history
            station.occupancy_history.append(is_occupied_now)
            if len(station.occupancy_history) > self.cumulative_window_frames:
                station.occupancy_history.pop(0)  # Keep only last n frames
            
            # Debug logging for occupancy history
            if self.debug_cumulative_time:
                logging.info(f"[{station.name}] Occupancy history: {station.occupancy_history}")
            
            # If currently occupied, start or continue tracking
            if is_occupied_now:
                if station.last_entry_time is None:
                    # Start tracking a new occupancy period
                    station.last_entry_time = current_time
                    if self.debug_cumulative_time:
                        logging.info(f"[{station.name}] Started cumulative time tracking at {current_time:.2f}s")
                # Continue accumulating time (will be calculated at cleanup)
            else:
                # Not currently occupied, check if we need to end a tracking period
                if station.last_entry_time is not None:
                    # Check if we should end tracking based on sliding window
                    if len(station.occupancy_history) >= self.cumulative_window_frames:
                        # Check if worker was absent for ALL of the last n frames
                        recent_frames = station.occupancy_history[-self.cumulative_window_frames:]
                        worker_absent_for_all = not any(recent_frames)
                        
                        if worker_absent_for_all:
                            # End current occupancy period and add to cumulative time
                            duration = current_time - station.last_entry_time
                            station.cumulative_seconds += duration
                            if self.debug_cumulative_time:
                                logging.info(f"[{station.name}] Added {duration:.2f}s to cumulative time (total: {station.cumulative_seconds:.2f}s). History: {recent_frames}")
                            station.last_entry_time = None
            
            # Update station worker history
            if station.name not in self.station_worker_history:
                self.station_worker_history[station.name] = {}
            
            # Update history for workers currently in this station (both detected and active tracks)
            all_workers_in_station = worker_ids_in_station.union(active_tracks_in_station)
            for tid in all_workers_in_station:
                self.station_worker_history[station.name][tid] = current_time
            
            station.current_workers = all_workers_in_station
            num_workers = len(station.current_workers)

            # Timed Violation State Machine
            if num_workers > station.max_workers:
                station.absence_start_time = None  # Reset absence timer
                if station.over_occupancy_start_time is None:
                    station.over_occupancy_start_time = current_time
                
                duration = current_time - station.over_occupancy_start_time
                if duration > self.over_occupancy_violation_sec:
                    station.status = f"VIOLATION: Over-Occupied > {self.over_occupancy_violation_sec}s"
                    station.status_color = (0, 0, 255)  # Red
                    if not station.violation_logged_time or (current_time - station.violation_logged_time) >= self.over_occupancy_violation_sec:
                        station.violation_logged_time = current_time
                else:
                    station.status = f"Warning: {num_workers} Workers (Max: {station.max_workers})"
                    station.status_color = (203, 192, 255)  # Purple
                    station.violation_logged_time = None # Reset when not in violation
            
            elif num_workers < station.min_workers:
                station.over_occupancy_start_time = None # Reset over-occupancy timer
                if station.absence_start_time is None:
                    station.absence_start_time = current_time

                absence_duration = current_time - station.absence_start_time
                if absence_duration >= self.unattended_violation_sec:
                    station.status = f"VIOLATION: Unattended for {int(absence_duration)} sec"
                    station.status_color = (0, 0, 255)  # Red
                    if not station.violation_logged_time or (current_time - station.violation_logged_time) >= self.unattended_violation_sec:
                        station.violation_logged_time = current_time
                else:
                    station.status = ""
                    station.status_color = (0, 255, 0)  # Green
                    station.violation_logged_time = None # Reset when not in violation
            
            else: # Correct occupancy
                station.absence_start_time = None
                station.over_occupancy_start_time = None
                station.status = f"Occupied by ID(s): {', '.join(map(str, sorted(list(station.current_workers))))}"
                station.status_color = (0, 255, 0)  # Green
                station.violation_logged_time = None # Reset when not in violation

    def _draw_overlays(self, frame: np.ndarray, helmet_status: dict, glove_status: dict, current_time: float) -> np.ndarray:
        """Draws bounding boxes, track IDs, and ROI overlays with enhanced text visibility."""
        for track_id, last_seen_frame in list(self.track_history.items()):
            # Only draw if track is not stale
            if self.frame_number - last_seen_frame > self.max_age_frames:
                continue
            if track_id in self.track_bboxes:
                box = self.track_bboxes[track_id]
                x1, y1, x2, y2 = box
                violations = []
                color = (0, 255, 0)  # Default green
                # Check helmet violation
                hstat = helmet_status.get(track_id, {})
                gstat = glove_status.get(track_id, {})
                
                # Check for violations first
                helmet_violation = hstat.get('violation', False)
                glove_violation = gstat.get('violation', False)
                
                if hstat.get('helmet_detected', False):
                    # Helmet is detected, show green (no violation)
                    color = (0, 255, 0)  # Green
                elif helmet_violation:
                    color = (0, 0, 255)
                    violations.append(f"No helmet for {int(hstat.get('violation_time', 0))} sec")
                
                if gstat.get('glove_detected', False):
                    # Gloves are detected, show green (no violation)
                    color = (0, 255, 0)  # Green
                elif glove_violation:
                    color = (0, 0, 255)
                    violations.append(f"No glove for {int(gstat.get('violation_time', 0))} sec")
                
                # Only show checking message if no violations are active
                if not helmet_violation and not glove_violation:
                    helmet_checking = hstat.get('checking', False)
                    glove_checking = gstat.get('checking', False)
                    
                    if helmet_checking or glove_checking:
                        color = (0, 165, 255)
                        violations.append("checking")
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                y_text = y2 + 30  # Start below the bounding box with more spacing
                for vmsg in violations:
                    if "No helmet" in vmsg:
                        draw_violation_text(frame, vmsg, (x1, y_text), "helmet")
                    elif "No glove" in vmsg:
                        draw_violation_text(frame, vmsg, (x1, y_text), "glove")
                    elif "checking" in vmsg:
                        draw_violation_text(frame, vmsg, (x1, y_text), "checking")
                    else:
                        draw_violation_text(frame, vmsg, (x1, y_text), "general")
                    y_text += 60  # Increased spacing between text lines for better separation

        y_offset = 40
        for station in self.station_states:
            # Default ROI color is green
            roi_color = (0, 255, 0)
            status_text = ""
            if "VIOLATION: Over-Occupied" in station.status:
                roi_color = (0, 0, 255)
                status_text = f"Maximum workers exceeded: {len(station.current_workers)} people"
            elif "VIOLATION: Unattended" in station.status:
                roi_color = (0, 0, 255)
                status_text = station.status
            cv2.polylines(frame, [station.polygon], isClosed=True, color=roi_color, thickness=3)
            if status_text:
                M = cv2.moments(station.polygon)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                else:
                    cx, cy = station.polygon[0][0][0], station.polygon[0][0][1]
                # Use enhanced text rendering for station status
                if "VIOLATION" in status_text:
                    draw_violation_text(frame, status_text, (cx - 150, cy), "occupancy")
                else:
                    draw_status_text(frame, status_text, (cx - 150, cy), "normal")
        return frame

    def cleanup(self, last_timestamp: float):
        logging.info("Running cleanup for PersonOccupancyModule...")
        self._write_summary_report(last_timestamp)
        if self.occupancy_log_file:
            self.occupancy_log_file.close()
        logging.info("✅ Occupancy reports generated.")

    def _setup_paths(self):
        output_path = Path(self.config.get('video_output_path', 'output/default.mp4'))
        output_path.parent.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [Occupancy] %(message)s")

    def _load_and_initialize_stations(self) -> List[StationState]:
        rois_data = self.config['rois_data']
        rois = dict(zip(rois_data['names'], rois_data['rois']))
        
        states = []
        for name, polygon in rois.items():
            polygon_np = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            config = STATION_WORKER_CONFIG.get(name, {"min": 1, "max": 1})
            states.append(StationState(name, polygon_np, config["min"], config["max"]))
            logging.info(f"Loaded station '{name}' with polygon: {polygon}")
        logging.info(f"Initialized {len(states)} station states.")
        return states

    def _seconds_to_hms(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _init_occupancy_log(self, video_filename, start_datetime):
        base_name = os.path.splitext(os.path.basename(video_filename))[0]
        self.occupancy_log_path = f"occupancy_intervals_{base_name}.csv"
        self.occupancy_log_file = open(self.occupancy_log_path, 'w', newline='')
        self.occupancy_log_writer = csv.writer(self.occupancy_log_file)
        
        if self.occupancy_interval_sec >= 60:
            interval_text = f"Interval ({self.occupancy_interval_sec // 60} min)"
        else:
            interval_text = f"Interval ({self.occupancy_interval_sec} s)"
            
        self.occupancy_log_writer.writerow([
            'video filename', 'timestamp', 'station', 'totalWorkers', interval_text
        ])
        self.occupancy_log_initialized = True
        self.occupancy_log_interval_counter = 1
        self.occupancy_log_start_datetime = start_datetime

    @staticmethod
    def _compute_iou(bbox: np.ndarray, roi_poly: Polygon) -> float:
        """Calculates IoU between a bounding box and a Shapely polygon."""
        try:
            person_poly = box(*bbox)
            if not person_poly.is_valid or not roi_poly.is_valid:
                return 0.0
            inter_area = person_poly.intersection(roi_poly).area
            union_area = person_poly.union(roi_poly).area
            return inter_area / union_area if union_area > 0 else 0.0
        except Exception:
            return 0.0

    def _write_summary_report(self, last_timestamp: float):
        summary_data = []
        for station in self.station_states:
            # Handle final cumulative time calculation with sliding window logic
            if station.last_entry_time is not None:
                # Check if worker was present in the last n frames
                if len(station.occupancy_history) >= self.cumulative_window_frames:
                    worker_present_in_recent_window = any(station.occupancy_history[-self.cumulative_window_frames:])
                    if worker_present_in_recent_window:
                        # Worker was present in recent frames, count the remaining time
                        station.cumulative_seconds += (last_timestamp - station.last_entry_time)
                    # If worker was not present in recent frames, don't count the time
                else:
                    # Not enough frames in history, check if currently occupied
                    if station.occupancy_history and station.occupancy_history[-1]:
                        # Currently occupied, count the time
                        station.cumulative_seconds += (last_timestamp - station.last_entry_time)
                
            summary_data.append({
                "station_name": station.name,
                "total_occupancy_time_seconds": round(station.cumulative_seconds, 2)
            })
        try:
            with open(self.summary_path, 'w', newline='') as f:
                import csv
                writer = csv.DictWriter(f, fieldnames=["station_name", "total_occupancy_time_seconds"])
                writer.writeheader()
                writer.writerows(summary_data)
            import logging
            logging.info(f"Cumulative time summary saved to '{self.summary_path}'.")
        except IOError as e:
            import logging
            logging.error(f"Could not write summary report: {e}")