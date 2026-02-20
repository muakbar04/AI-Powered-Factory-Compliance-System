import csv
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

class ViolationsLogger:
    """
    Configurable violations logger that logs violations at specified intervals.
    
    Format:
    - video_filename: Name of the video being processed
    - timestamp: Date from processing + video start time + interval increments
    - interval: Configurable interval number
    - station: Work station name
    - Unattended Station Violation: 1 if true, empty if false
    - Helmet Violation: 1 if true, 0 if false
    - Glove Violation: 1 if true, 0 if false
    - Maximum Workers Violation: 1 if true, 0 if false
    - details: Text description of violations
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the violations logger.
        
        Args:
            config: Dictionary containing:
                - interval_seconds: Interval duration in seconds
                - video_filename: Name of the video file
                - start_datetime: Processing start datetime
                - output_path: Path for the violations log file
                - unattended_threshold: Seconds threshold for unattended station violation (default: 5)
                - max_workers_threshold: Seconds threshold for maximum workers violation (default: 10)
        """
        self.interval_seconds = config.get('interval_seconds', 60) 
        self.video_filename = config.get('video_filename', 'video_1.mp4')
        self.start_datetime = config.get('start_datetime', datetime.now())
        self.output_path = config.get('output_path', 'violations_log.csv')
        
        self.unattended_threshold = config.get('unattended_threshold', 5)  
        self.max_workers_threshold = config.get('max_workers_threshold', 10) 
        
        # State tracking
        self.current_interval = 1
        self.last_log_time = 0.0
        self.violations_buffer: Dict[str, Dict] = {} 
        
        self._init_log_file()
        
        logging.info(f"ViolationsLogger initialized with {self.interval_seconds}s intervals, "
                    f"unattended threshold: {self.unattended_threshold}s, "
                    f"max workers threshold: {self.max_workers_threshold}s")
    
    def _init_log_file(self):
        """Initialize the violations log CSV file with headers."""
        if not os.path.exists(self.output_path):
            with open(self.output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                interval_text = f"interval ({self.interval_seconds}s)"
                writer.writerow([
                    'video filename',
                    'timestamp',
                    interval_text,
                    'station',
                    'Unattended Station Violation',
                    'Helmet Violation',
                    'Glove Violation',
                    'Maximum Workers Violation',
                    'details'
                ])
    
    def _get_timestamp_for_interval(self, video_time: float) -> str:
        """
        Generate timestamp for the current interval.
        
        Args:
            video_time: Current time in video (seconds)
            
        Returns:
            Formatted timestamp string (always shows video start time)
        """
        return self.start_datetime.strftime('%Y-%m-%d %H:%M:%S')
    
    def update_violations(self, station: str, violations: Dict[str, bool], details: str = ""):
        """
        Args:
            station: Station name
            violations: Dictionary with violation types as keys and boolean values
            details: Optional details about the violations
        """
        self.violations_buffer[station] = {
            'violations': violations,
            'details': details
        }
    
    def process_frame(self, current_time: float, station_states: List) -> bool:
        """
        Process current frame and log violations if interval has passed.
        
        Args:
            current_time: Current time in video (seconds)
            station_states: List of station state objects
            
        Returns:
            True if violations were logged, False otherwise
        """
        if current_time - self.last_log_time >= self.interval_seconds:
            self._log_interval_violations(current_time, station_states)
            self.current_interval += 1
            self.last_log_time = current_time
            return True
        return False
    
    def _log_interval_violations(self, current_time: float, station_states: List):
        """Log violations for all stations in the current interval."""
        timestamp = self._get_timestamp_for_interval(current_time)
        
        for station in station_states:
            violations = self._determine_violations(station, current_time)
            
            if violations.get('unattended_station', False):
                helmet_violation = ""
                glove_violation = ""
                max_workers_violation = ""
                details = violations.get('details', 'No Worker')
            else:
                helmet_violation = "1" if violations.get('helmet', False) else "0"
                glove_violation = "1" if violations.get('glove', False) else "0"
                max_workers_violation = "1" if violations.get('max_workers', False) else "0"
                details = violations.get('details', "")
            
            with open(self.output_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.video_filename,
                    timestamp,
                    self.current_interval,
                    station.name,
                    "1" if violations.get('unattended_station', False) else "0",
                    helmet_violation,
                    glove_violation,
                    max_workers_violation,
                    details
                ])
            
            logging.info(f"Logged violations for {station.name} at interval {self.current_interval}")
    
    def _determine_violations(self, station, current_time: float) -> Dict[str, bool]:
        """
        Determine violations for a station based on its current state.
        
        Args:
            station: Station state object
            current_time: Current time in video (seconds)
            
        Returns:
            Dictionary with violation types and their boolean values
        """
        violations = {
            'unattended_station': False,
            'helmet': False,
            'glove': False,
            'max_workers': False,
            'details': ""
        }
        
        num_workers = len(station.current_workers)
        detail_parts = []
        
        # Check unattended station violation
        if num_workers < station.min_workers:
            if station.absence_start_time is not None:
                absence_duration = current_time - station.absence_start_time
                if absence_duration >= self.unattended_threshold:
                    violations['unattended_station'] = True
                    violations['details'] = f"No Worker for {int(absence_duration)}s"
        
        # Check maximum workers violation
        if num_workers > station.max_workers:
            if station.over_occupancy_start_time is not None:
                over_duration = current_time - station.over_occupancy_start_time
                if over_duration >= self.max_workers_threshold:
                    violations['max_workers'] = True
                    detail_parts.append(f"{num_workers} workers at {station.name} for {int(over_duration)}s")
        
        # For helmet and glove violations, get from buffer
        if station.name in self.violations_buffer:
            buffer_violations = self.violations_buffer[station.name]['violations']
            violations['helmet'] = buffer_violations.get('helmet', False)
            violations['glove'] = buffer_violations.get('glove', False)
            
            if violations['helmet']:
                detail_parts.append("No Helmet")
            if violations['glove']:
                detail_parts.append("No Gloves")
        
        # Combine all details
        if detail_parts:
            violations['details'] = " | ".join(detail_parts)
        
        return violations
    
    def cleanup(self):
        logging.info("ViolationsLogger cleanup completed") 