"""
Advanced Computer Vision Processor
Implements object tracking, pose estimation, semantic segmentation, and more
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass, field
from enum import Enum
import torch
from collections import defaultdict, deque
import time
from abc import ABC, abstractmethod

from ..models.model_manager import Detection, get_model_manager
from ..core.config import get_config

logger = logging.getLogger(__name__)


class TrackerType(Enum):
    """Supported tracker types"""
    BYTETRACK = "ByteTrack"
    DEEPSORT = "DeepSORT"
    STRONGSORT = "StrongSORT"
    BOTSORT = "BotSORT"


@dataclass
class TrackedObject:
    """Tracked object with history"""
    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]
    confidence: float
    center_history: deque = field(default_factory=lambda: deque(maxlen=30))
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=10))
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    total_frames: int = 0
    disappeared_frames: int = 0
    velocity: Tuple[float, float] = (0.0, 0.0)
    is_active: bool = True


@dataclass
class PoseKeypoint:
    """Pose estimation keypoint"""
    x: float
    y: float
    confidence: float
    visible: bool = True


@dataclass
class PoseEstimation:
    """Pose estimation result"""
    keypoints: List[PoseKeypoint]
    bbox: Tuple[float, float, float, float]
    confidence: float
    pose_id: Optional[int] = None


class BaseTracker(ABC):
    """Base class for object trackers"""
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 50):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_id = 0
        self.tracked_objects: Dict[int, TrackedObject] = {}
    
    @abstractmethod
    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """Update tracker with new detections"""
        pass
    
    def _calculate_distance(self, bbox1: Tuple[float, float, float, float], 
                          bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate distance between two bounding boxes"""
        x1_center = (bbox1[0] + bbox1[2]) / 2
        y1_center = (bbox1[1] + bbox1[3]) / 2
        x2_center = (bbox2[0] + bbox2[2]) / 2
        y2_center = (bbox2[1] + bbox2[3]) / 2
        
        return np.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)


class SimpleTracker(BaseTracker):
    """Simple centroid-based tracker"""
    
    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """Update tracker with simple centroid-based matching"""
        current_time = time.time()
        
        # If no detections, mark all as disappeared
        if not detections:
            for track_id in list(self.tracked_objects.keys()):
                self.tracked_objects[track_id].disappeared_frames += 1
                if self.tracked_objects[track_id].disappeared_frames > self.max_disappeared:
                    del self.tracked_objects[track_id]
            return list(self.tracked_objects.values())
        
        # If no existing objects, create new tracks
        if not self.tracked_objects:
            for detection in detections:
                self._create_new_track(detection, current_time)
        else:
            # Match detections to existing tracks
            self._match_detections_to_tracks(detections, current_time)
        
        # Remove tracks that have disappeared for too long
        to_remove = []
        for track_id, tracked_obj in self.tracked_objects.items():
            if tracked_obj.disappeared_frames > self.max_disappeared:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracked_objects[track_id]
        
        return list(self.tracked_objects.values())
    
    def _create_new_track(self, detection: Detection, current_time: float) -> None:
        """Create a new tracked object"""
        center = self._get_bbox_center(detection.bbox)
        
        tracked_obj = TrackedObject(
            track_id=self.next_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            bbox=detection.bbox,
            confidence=detection.confidence,
            first_seen=current_time,
            last_seen=current_time,
            total_frames=1,
            disappeared_frames=0
        )
        
        tracked_obj.center_history.append(center)
        tracked_obj.bbox_history.append(detection.bbox)
        
        self.tracked_objects[self.next_id] = tracked_obj
        self.next_id += 1
    
    def _match_detections_to_tracks(self, detections: List[Detection], current_time: float) -> None:
        """Match detections to existing tracks"""
        if not self.tracked_objects:
            return
        
        # Calculate distance matrix
        track_ids = list(self.tracked_objects.keys())
        distances = np.full((len(detections), len(track_ids)), np.inf)
        
        for i, detection in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                tracked_obj = self.tracked_objects[track_id]
                distance = self._calculate_distance(detection.bbox, tracked_obj.bbox)
                
                # Only consider matches of the same class and within distance threshold
                if (detection.class_id == tracked_obj.class_id and 
                    distance <= self.max_distance):
                    distances[i, j] = distance
        
        # Simple greedy matching (can be improved with Hungarian algorithm)
        used_detections = set()
        used_tracks = set()
        
        for _ in range(min(len(detections), len(track_ids))):
            min_idx = np.unravel_index(np.argmin(distances), distances.shape)
            det_idx, track_idx = min_idx
            
            if distances[det_idx, track_idx] != np.inf:
                # Match found
                detection = detections[det_idx]
                track_id = track_ids[track_idx]
                
                self._update_track(track_id, detection, current_time)
                
                used_detections.add(det_idx)
                used_tracks.add(track_idx)
                
                # Remove this match from future consideration
                distances[det_idx, :] = np.inf
                distances[:, track_idx] = np.inf
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in used_detections:
                self._create_new_track(detection, current_time)
        
        # Mark unmatched tracks as disappeared
        for j, track_id in enumerate(track_ids):
            if j not in used_tracks:
                self.tracked_objects[track_id].disappeared_frames += 1
    
    def _update_track(self, track_id: int, detection: Detection, current_time: float) -> None:
        """Update an existing track with new detection"""
        tracked_obj = self.tracked_objects[track_id]
        
        # Update basic properties
        old_center = self._get_bbox_center(tracked_obj.bbox)
        new_center = self._get_bbox_center(detection.bbox)
        
        tracked_obj.bbox = detection.bbox
        tracked_obj.confidence = detection.confidence
        tracked_obj.last_seen = current_time
        tracked_obj.total_frames += 1
        tracked_obj.disappeared_frames = 0
        
        # Update history
        tracked_obj.center_history.append(new_center)
        tracked_obj.bbox_history.append(detection.bbox)
        
        # Calculate velocity
        if len(tracked_obj.center_history) >= 2:
            prev_center = tracked_obj.center_history[-2]
            time_diff = 1/30  # Assume 30 FPS
            tracked_obj.velocity = (
                (new_center[0] - prev_center[0]) / time_diff,
                (new_center[1] - prev_center[1]) / time_diff
            )
    
    def _get_bbox_center(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Get center point of bounding box"""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


class AdvancedCVProcessor:
    """
    Advanced computer vision processor with multiple features:
    - Object detection and tracking
    - Pose estimation
    - Semantic segmentation
    - Depth estimation
    - Face recognition
    """
    
    def __init__(self):
        self.config = get_config()
        self.tracker: Optional[BaseTracker] = None
        self.pose_model = None
        self.segmentation_model = None
        self.depth_model = None
        self.face_recognition_model = None
        
        # Feature flags
        self.cv_config = self.config.get_cv_features_config()
        self.tracking_enabled = self.cv_config.get('object_tracking', {}).get('enabled', False)
        self.pose_enabled = self.cv_config.get('pose_estimation', {}).get('enabled', False)
        self.segmentation_enabled = self.cv_config.get('semantic_segmentation', {}).get('enabled', False)
        self.depth_enabled = self.cv_config.get('depth_estimation', {}).get('enabled', False)
        self.face_recognition_enabled = self.cv_config.get('face_recognition', {}).get('enabled', False)
        
        # Initialize components
        self._initialize_tracker()
        self._initialize_pose_estimation()
        self._initialize_segmentation()
    
    def _initialize_tracker(self) -> None:
        """Initialize object tracker"""
        if not self.tracking_enabled:
            return
        
        tracker_config = self.cv_config.get('object_tracking', {})
        tracker_type = tracker_config.get('tracker_type', 'SimpleTracker')
        
        # For now, use simple tracker (can be extended with ByteTrack, DeepSORT)
        self.tracker = SimpleTracker(
            max_disappeared=tracker_config.get('max_disappeared', 30),
            max_distance=tracker_config.get('max_distance', 50)
        )
        
        logger.info(f"Initialized {tracker_type} tracker")
    
    def _initialize_pose_estimation(self) -> None:
        """Initialize pose estimation model"""
        if not self.pose_enabled:
            return
        
        try:
            from ultralytics import YOLO
            pose_config = self.cv_config.get('pose_estimation', {})
            model_path = pose_config.get('model', 'yolov8n-pose.pt')
            self.pose_model = YOLO(model_path)
            logger.info("Pose estimation model initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize pose estimation: {e}")
            self.pose_enabled = False
    
    def _initialize_segmentation(self) -> None:
        """Initialize semantic segmentation model"""
        if not self.segmentation_enabled:
            return
        
        try:
            from ultralytics import YOLO
            seg_config = self.cv_config.get('semantic_segmentation', {})
            model_path = seg_config.get('model', 'yolov8n-seg.pt')
            self.segmentation_model = YOLO(model_path)
            logger.info("Segmentation model initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize segmentation: {e}")
            self.segmentation_enabled = False
    
    async def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame with all enabled CV features
        
        Returns:
            Dictionary containing all processing results
        """
        results = {
            'frame': frame.copy(),
            'detections': [],
            'tracked_objects': [],
            'poses': [],
            'segmentation_masks': None,
            'depth_map': None,
            'faces': [],
            'processing_time': {}
        }
        
        # Object Detection
        start_time = time.time()
        model_manager = await get_model_manager()
        detections = await model_manager.predict(frame)
        results['detections'] = detections
        results['processing_time']['detection'] = time.time() - start_time
        
        # Object Tracking
        if self.tracking_enabled and self.tracker:
            start_time = time.time()
            tracked_objects = self.tracker.update(detections)
            results['tracked_objects'] = tracked_objects
            results['processing_time']['tracking'] = time.time() - start_time
            
            # Update detections with track IDs
            for detection in detections:
                for tracked_obj in tracked_objects:
                    if (self._calculate_iou(detection.bbox, tracked_obj.bbox) > 0.5 and
                        detection.class_id == tracked_obj.class_id):
                        detection.track_id = tracked_obj.track_id
                        break
        
        # Pose Estimation
        if self.pose_enabled and self.pose_model:
            start_time = time.time()
            poses = await self._estimate_poses(frame)
            results['poses'] = poses
            results['processing_time']['pose_estimation'] = time.time() - start_time
        
        # Semantic Segmentation
        if self.segmentation_enabled and self.segmentation_model:
            start_time = time.time()
            masks = await self._segment_objects(frame)
            results['segmentation_masks'] = masks
            results['processing_time']['segmentation'] = time.time() - start_time
        
        # Depth Estimation (placeholder for future implementation)
        if self.depth_enabled:
            start_time = time.time()
            depth_map = await self._estimate_depth(frame)
            results['depth_map'] = depth_map
            results['processing_time']['depth_estimation'] = time.time() - start_time
        
        # Face Recognition (placeholder for future implementation)
        if self.face_recognition_enabled:
            start_time = time.time()
            faces = await self._recognize_faces(frame)
            results['faces'] = faces
            results['processing_time']['face_recognition'] = time.time() - start_time
        
        return results
    
    async def _estimate_poses(self, frame: np.ndarray) -> List[PoseEstimation]:
        """Estimate human poses in the frame"""
        if not self.pose_model:
            return []
        
        try:
            results = self.pose_model(frame)
            poses = []
            
            for result in results:
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints_data = result.keypoints.data
                    boxes_data = result.boxes.xyxy if result.boxes is not None else []
                    confs_data = result.boxes.conf if result.boxes is not None else []
                    
                    for i, keypoints_tensor in enumerate(keypoints_data):
                        keypoints = []
                        for kp in keypoints_tensor:
                            x, y, conf = kp.cpu().numpy()
                            keypoints.append(PoseKeypoint(
                                x=float(x),
                                y=float(y),
                                confidence=float(conf),
                                visible=conf > 0.5
                            ))
                        
                        bbox = tuple(boxes_data[i].cpu().numpy()) if i < len(boxes_data) else (0, 0, 0, 0)
                        confidence = float(confs_data[i].cpu().numpy()) if i < len(confs_data) else 0.0
                        
                        poses.append(PoseEstimation(
                            keypoints=keypoints,
                            bbox=bbox,
                            confidence=confidence
                        ))
            
            return poses
            
        except Exception as e:
            logger.error(f"Pose estimation failed: {e}")
            return []
    
    async def _segment_objects(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Perform semantic segmentation"""
        if not self.segmentation_model:
            return None
        
        try:
            results = self.segmentation_model(frame)
            
            if results and hasattr(results[0], 'masks') and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                return masks
            
            return None
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            return None
    
    async def _estimate_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Estimate depth map (placeholder for MiDaS or similar)"""
        # Placeholder for depth estimation implementation
        # Would integrate with MiDaS, DPT, or similar models
        return None
    
    async def _recognize_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Recognize faces in the frame (placeholder)"""
        # Placeholder for face recognition implementation
        # Would integrate with FaceNet, ArcFace, or similar models
        return []
    
    def _calculate_iou(self, bbox1: Tuple[float, float, float, float], 
                      bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate Intersection over Union for two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detection bounding boxes on frame"""
        result_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.bbox)
            
            # Choose color based on class
            color = self._get_class_color(detection.class_id)
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            if detection.track_id is not None:
                label += f" (ID: {detection.track_id})"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(result_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_frame
    
    def draw_tracking_trails(self, frame: np.ndarray, tracked_objects: List[TrackedObject]) -> np.ndarray:
        """Draw tracking trails for tracked objects"""
        result_frame = frame.copy()
        
        for tracked_obj in tracked_objects:
            if len(tracked_obj.center_history) < 2:
                continue
            
            color = self._get_class_color(tracked_obj.class_id)
            
            # Draw trail
            points = list(tracked_obj.center_history)
            for i in range(1, len(points)):
                # Fade trail over time
                alpha = i / len(points)
                thickness = max(1, int(3 * alpha))
                
                cv2.line(result_frame, 
                        (int(points[i-1][0]), int(points[i-1][1])),
                        (int(points[i][0]), int(points[i][1])),
                        color, thickness)
        
        return result_frame
    
    def draw_poses(self, frame: np.ndarray, poses: List[PoseEstimation]) -> np.ndarray:
        """Draw pose keypoints and skeleton"""
        result_frame = frame.copy()
        
        # COCO pose skeleton connections
        skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
        
        for pose in poses:
            keypoints = pose.keypoints
            
            # Draw keypoints
            for kp in keypoints:
                if kp.visible and kp.confidence > 0.5:
                    cv2.circle(result_frame, (int(kp.x), int(kp.y)), 3, (0, 255, 0), -1)
            
            # Draw skeleton
            for connection in skeleton:
                kp1_idx, kp2_idx = connection[0] - 1, connection[1] - 1
                if (kp1_idx < len(keypoints) and kp2_idx < len(keypoints) and
                    keypoints[kp1_idx].visible and keypoints[kp2_idx].visible and
                    keypoints[kp1_idx].confidence > 0.5 and keypoints[kp2_idx].confidence > 0.5):
                    
                    pt1 = (int(keypoints[kp1_idx].x), int(keypoints[kp1_idx].y))
                    pt2 = (int(keypoints[kp2_idx].x), int(keypoints[kp2_idx].y))
                    cv2.line(result_frame, pt1, pt2, (255, 0, 0), 2)
        
        return result_frame
    
    def draw_segmentation_masks(self, frame: np.ndarray, masks: Optional[np.ndarray]) -> np.ndarray:
        """Overlay segmentation masks on frame"""
        if masks is None:
            return frame
        
        result_frame = frame.copy()
        
        for i, mask in enumerate(masks):
            # Resize mask to frame size if needed
            if mask.shape != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            
            # Create colored mask
            color = self._get_class_color(i)
            colored_mask = np.zeros_like(frame)
            colored_mask[mask > 0.5] = color
            
            # Overlay with transparency
            result_frame = cv2.addWeighted(result_frame, 0.8, colored_mask, 0.2, 0)
        
        return result_frame
    
    def _get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get color for a specific class ID"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
            (192, 192, 192), (128, 128, 128), (255, 165, 0), (255, 20, 147),
            (0, 191, 255), (255, 140, 0), (34, 139, 34), (255, 69, 0)
        ]
        return colors[class_id % len(colors)]
    
    def get_analytics(self, tracked_objects: List[TrackedObject]) -> Dict[str, Any]:
        """Generate analytics from tracked objects"""
        analytics = {
            'total_objects': len(tracked_objects),
            'active_objects': len([obj for obj in tracked_objects if obj.is_active]),
            'class_distribution': defaultdict(int),
            'average_confidence': 0.0,
            'tracking_duration': {},
            'object_speeds': {}
        }
        
        if not tracked_objects:
            return analytics
        
        total_confidence = 0.0
        
        for obj in tracked_objects:
            analytics['class_distribution'][obj.class_name] += 1
            total_confidence += obj.confidence
            
            # Calculate tracking duration
            duration = obj.last_seen - obj.first_seen
            analytics['tracking_duration'][obj.track_id] = duration
            
            # Calculate speed
            if len(obj.center_history) >= 2:
                velocity_magnitude = np.sqrt(obj.velocity[0]**2 + obj.velocity[1]**2)
                analytics['object_speeds'][obj.track_id] = velocity_magnitude
        
        analytics['average_confidence'] = total_confidence / len(tracked_objects)
        
        return analytics
    
    def create_heatmap(self, frame_shape: Tuple[int, int], tracked_objects: List[TrackedObject]) -> np.ndarray:
        """Create heatmap showing object density"""
        heatmap = np.zeros(frame_shape[:2], dtype=np.float32)
        
        for obj in tracked_objects:
            for center in obj.center_history:
                x, y = int(center[0]), int(center[1])
                if 0 <= x < frame_shape[1] and 0 <= y < frame_shape[0]:
                    # Add gaussian blob around center
                    cv2.circle(heatmap, (x, y), 20, 1.0, -1)
        
        # Normalize and apply colormap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return heatmap_colored
    
    def get_object_statistics(self, tracked_objects: List[TrackedObject]) -> Dict[str, Any]:
        """Get detailed object statistics"""
        stats = {
            'total_unique_objects': len(tracked_objects),
            'currently_visible': len([obj for obj in tracked_objects if obj.disappeared_frames == 0]),
            'class_counts': defaultdict(int),
            'average_tracking_duration': 0.0,
            'longest_tracked_object': None,
            'fastest_object': None,
            'detection_confidence_stats': {
                'mean': 0.0,
                'min': 1.0,
                'max': 0.0,
                'std': 0.0
            }
        }
        
        if not tracked_objects:
            return stats
        
        durations = []
        speeds = []
        confidences = []
        
        for obj in tracked_objects:
            stats['class_counts'][obj.class_name] += 1
            
            duration = obj.last_seen - obj.first_seen
            durations.append(duration)
            
            velocity_magnitude = np.sqrt(obj.velocity[0]**2 + obj.velocity[1]**2)
            speeds.append(velocity_magnitude)
            
            confidences.append(obj.confidence)
        
        if durations:
            stats['average_tracking_duration'] = np.mean(durations)
            longest_idx = np.argmax(durations)
            stats['longest_tracked_object'] = {
                'track_id': tracked_objects[longest_idx].track_id,
                'duration': durations[longest_idx],
                'class_name': tracked_objects[longest_idx].class_name
            }
        
        if speeds:
            fastest_idx = np.argmax(speeds)
            stats['fastest_object'] = {
                'track_id': tracked_objects[fastest_idx].track_id,
                'speed': speeds[fastest_idx],
                'class_name': tracked_objects[fastest_idx].class_name
            }
        
        if confidences:
            stats['detection_confidence_stats'] = {
                'mean': np.mean(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences),
                'std': np.std(confidences)
            }
        
        return stats


# Global CV processor instance
cv_processor = AdvancedCVProcessor()


def get_cv_processor() -> AdvancedCVProcessor:
    """Get global CV processor instance"""
    return cv_processor
