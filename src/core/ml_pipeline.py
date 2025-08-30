"""
AI-Powered Machine Learning Pipeline
Implements active learning, automated annotation, model ensemble, and continuous learning
"""

import asyncio
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pickle
from abc import ABC, abstractmethod

from ..models.model_manager import Detection, get_model_manager
from ..core.config import get_config
from ..database.manager import get_database_manager

logger = logging.getLogger(__name__)


class SamplingStrategy(Enum):
    """Active learning sampling strategies"""
    UNCERTAINTY = "uncertainty"
    DIVERSITY = "diversity"
    HYBRID = "hybrid"
    RANDOM = "random"


class LearningMode(Enum):
    """Learning modes"""
    SUPERVISED = "supervised"
    SEMI_SUPERVISED = "semi_supervised"
    ACTIVE = "active"
    CONTINUOUS = "continuous"


@dataclass
class AnnotationCandidate:
    """Candidate for annotation in active learning"""
    image_id: str
    image_path: str
    detections: List[Detection]
    uncertainty_score: float
    diversity_score: float
    composite_score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    is_annotated: bool = False
    annotation_quality: Optional[float] = None


@dataclass
class TrainingJob:
    """Training job information"""
    job_id: str
    model_name: str
    dataset_name: str
    status: str  # pending, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    map_50: float
    map_95: float
    precision: float
    recall: float
    f1_score: float
    inference_time: float
    memory_usage: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class UncertaintyEstimator:
    """Estimates prediction uncertainty for active learning"""
    
    def __init__(self, method: str = "entropy"):
        self.method = method
    
    def calculate_uncertainty(self, detections: List[Detection]) -> float:
        """Calculate uncertainty score for a set of detections"""
        if not detections:
            return 1.0  # High uncertainty for no detections
        
        if self.method == "entropy":
            return self._entropy_uncertainty(detections)
        elif self.method == "variance":
            return self._variance_uncertainty(detections)
        elif self.method == "confidence":
            return self._confidence_uncertainty(detections)
        else:
            return 0.5
    
    def _entropy_uncertainty(self, detections: List[Detection]) -> float:
        """Calculate entropy-based uncertainty"""
        confidences = [det.confidence for det in detections]
        
        if not confidences:
            return 1.0
        
        # Calculate entropy
        entropy = 0.0
        for conf in confidences:
            if conf > 0:
                entropy -= conf * np.log2(conf)
            if (1 - conf) > 0:
                entropy -= (1 - conf) * np.log2(1 - conf)
        
        return entropy / len(confidences)
    
    def _variance_uncertainty(self, detections: List[Detection]) -> float:
        """Calculate variance-based uncertainty"""
        confidences = [det.confidence for det in detections]
        
        if len(confidences) < 2:
            return 1.0
        
        return float(np.var(confidences))
    
    def _confidence_uncertainty(self, detections: List[Detection]) -> float:
        """Calculate confidence-based uncertainty (1 - max_confidence)"""
        if not detections:
            return 1.0
        
        max_confidence = max(det.confidence for det in detections)
        return 1.0 - max_confidence


class DiversityEstimator:
    """Estimates sample diversity for active learning"""
    
    def __init__(self):
        self.feature_cache: Dict[str, np.ndarray] = {}
    
    def calculate_diversity(self, image_features: np.ndarray, 
                          existing_features: List[np.ndarray]) -> float:
        """Calculate diversity score relative to existing samples"""
        if not existing_features:
            return 1.0
        
        # Calculate minimum distance to existing samples
        min_distance = float('inf')
        for existing_feature in existing_features:
            distance = np.linalg.norm(image_features - existing_feature)
            min_distance = min(min_distance, distance)
        
        # Normalize diversity score
        return min(1.0, min_distance / 1000.0)  # Adjust scaling as needed


class ActiveLearningManager:
    """
    Manages active learning pipeline:
    - Uncertainty estimation
    - Sample selection
    - Annotation candidate generation
    """
    
    def __init__(self):
        self.config = get_config()
        self.al_config = self.config.get('ml_pipeline.active_learning', {})
        self.enabled = self.al_config.get('enabled', False)
        
        self.uncertainty_threshold = self.al_config.get('uncertainty_threshold', 0.7)
        self.annotation_budget = self.al_config.get('annotation_budget', 1000)
        self.sampling_strategy = SamplingStrategy(self.al_config.get('sampling_strategy', 'uncertainty'))
        
        self.uncertainty_estimator = UncertaintyEstimator()
        self.diversity_estimator = DiversityEstimator()
        
        self.annotation_candidates: List[AnnotationCandidate] = []
        self.annotated_samples: List[AnnotationCandidate] = []
    
    async def evaluate_frame(self, image: np.ndarray, detections: List[Detection], 
                           image_id: str) -> Optional[AnnotationCandidate]:
        """Evaluate if frame should be annotated"""
        if not self.enabled:
            return None
        
        # Calculate uncertainty
        uncertainty_score = self.uncertainty_estimator.calculate_uncertainty(detections)
        
        # Skip if uncertainty is too low
        if uncertainty_score < self.uncertainty_threshold:
            return None
        
        # Calculate diversity (simplified - would use actual image features)
        diversity_score = 0.8  # Placeholder
        
        # Calculate composite score
        composite_score = self._calculate_composite_score(uncertainty_score, diversity_score)
        
        candidate = AnnotationCandidate(
            image_id=image_id,
            image_path=f"temp/{image_id}.jpg",
            detections=detections,
            uncertainty_score=uncertainty_score,
            diversity_score=diversity_score,
            composite_score=composite_score
        )
        
        # Add to candidates if budget allows
        if len(self.annotation_candidates) < self.annotation_budget:
            self.annotation_candidates.append(candidate)
            self.annotation_candidates.sort(key=lambda x: x.composite_score, reverse=True)
            return candidate
        
        return None
    
    def _calculate_composite_score(self, uncertainty: float, diversity: float) -> float:
        """Calculate composite score for sample selection"""
        if self.sampling_strategy == SamplingStrategy.UNCERTAINTY:
            return uncertainty
        elif self.sampling_strategy == SamplingStrategy.DIVERSITY:
            return diversity
        elif self.sampling_strategy == SamplingStrategy.HYBRID:
            return 0.7 * uncertainty + 0.3 * diversity
        else:  # RANDOM
            return np.random.random()
    
    def get_top_candidates(self, n: int = 10) -> List[AnnotationCandidate]:
        """Get top N annotation candidates"""
        return self.annotation_candidates[:n]
    
    async def mark_annotated(self, image_id: str, annotation_quality: float = 1.0) -> None:
        """Mark a candidate as annotated"""
        for candidate in self.annotation_candidates:
            if candidate.image_id == image_id:
                candidate.is_annotated = True
                candidate.annotation_quality = annotation_quality
                self.annotated_samples.append(candidate)
                self.annotation_candidates.remove(candidate)
                break


class AutoAnnotationManager:
    """
    Manages automated annotation with human-in-the-loop validation
    """
    
    def __init__(self):
        self.config = get_config()
        self.auto_config = self.config.get('ml_pipeline.auto_annotation', {})
        self.enabled = self.auto_config.get('enabled', False)
        
        self.confidence_threshold = self.auto_config.get('confidence_threshold', 0.9)
        self.human_review_threshold = self.auto_config.get('human_review_threshold', 0.7)
        
        self.auto_annotations: List[Dict[str, Any]] = []
        self.pending_review: List[Dict[str, Any]] = []
    
    async def process_detections(self, image_id: str, detections: List[Detection]) -> Dict[str, Any]:
        """Process detections for auto-annotation"""
        if not self.enabled:
            return {"auto_annotated": 0, "needs_review": 0}
        
        auto_annotated = 0
        needs_review = 0
        
        for detection in detections:
            if detection.confidence >= self.confidence_threshold:
                # High confidence - auto-annotate
                await self._save_auto_annotation(image_id, detection)
                auto_annotated += 1
            elif detection.confidence >= self.human_review_threshold:
                # Medium confidence - needs human review
                await self._queue_for_review(image_id, detection)
                needs_review += 1
        
        return {
            "auto_annotated": auto_annotated,
            "needs_review": needs_review
        }
    
    async def _save_auto_annotation(self, image_id: str, detection: Detection) -> None:
        """Save auto-annotation to database"""
        annotation = {
            "image_id": image_id,
            "bbox": detection.bbox,
            "class_id": detection.class_id,
            "class_name": detection.class_name,
            "confidence": detection.confidence,
            "auto_generated": True,
            "timestamp": datetime.utcnow()
        }
        
        self.auto_annotations.append(annotation)
        
        # Save to database
        db_manager = get_database_manager()
        await db_manager.save_annotation(annotation)
    
    async def _queue_for_review(self, image_id: str, detection: Detection) -> None:
        """Queue detection for human review"""
        review_item = {
            "image_id": image_id,
            "detection": detection,
            "timestamp": datetime.utcnow(),
            "reviewed": False
        }
        
        self.pending_review.append(review_item)
    
    def get_pending_review(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get items pending human review"""
        return self.pending_review[:limit]


class ContinuousLearningManager:
    """
    Manages continuous learning pipeline:
    - Performance monitoring
    - Automated retraining
    - Model validation
    """
    
    def __init__(self):
        self.config = get_config()
        self.cl_config = self.config.get('ml_pipeline.continuous_learning', {})
        self.enabled = self.cl_config.get('enabled', False)
        
        self.retrain_interval = self.cl_config.get('retrain_interval', 'weekly')
        self.min_new_samples = self.cl_config.get('min_new_samples', 100)
        self.performance_threshold = self.cl_config.get('performance_threshold', 0.05)
        
        self.performance_history: List[ModelPerformance] = []
        self.training_jobs: Dict[str, TrainingJob] = {}
        
        self.last_retrain_time = datetime.utcnow()
        self.new_samples_count = 0
    
    async def monitor_performance(self, model_name: str) -> ModelPerformance:
        """Monitor model performance and trigger retraining if needed"""
        if not self.enabled:
            return None
        
        # Get current performance metrics
        performance = await self._evaluate_model_performance(model_name)
        self.performance_history.append(performance)
        
        # Check if retraining is needed
        should_retrain = await self._should_retrain(performance)
        
        if should_retrain:
            await self._trigger_retraining(model_name)
        
        return performance
    
    async def _evaluate_model_performance(self, model_name: str) -> ModelPerformance:
        """Evaluate current model performance"""
        # Placeholder implementation
        # Would run validation on test set and calculate metrics
        
        model_manager = await get_model_manager()
        model_info = model_manager.get_model_info(model_name)
        
        # Simulate performance evaluation
        return ModelPerformance(
            model_name=model_name,
            map_50=0.85,  # Placeholder values
            map_95=0.72,
            precision=0.88,
            recall=0.82,
            f1_score=0.85,
            inference_time=25.0,  # ms
            memory_usage=model_info.memory_usage if model_info else 0.0
        )
    
    async def _should_retrain(self, current_performance: ModelPerformance) -> bool:
        """Determine if model should be retrained"""
        # Check if enough time has passed
        if self.retrain_interval == 'daily':
            time_threshold = timedelta(days=1)
        elif self.retrain_interval == 'weekly':
            time_threshold = timedelta(weeks=1)
        elif self.retrain_interval == 'monthly':
            time_threshold = timedelta(days=30)
        else:
            time_threshold = timedelta(days=7)  # Default to weekly
        
        time_passed = datetime.utcnow() - self.last_retrain_time
        
        # Check conditions
        time_condition = time_passed >= time_threshold
        samples_condition = self.new_samples_count >= self.min_new_samples
        
        # Check performance degradation
        performance_condition = False
        if len(self.performance_history) >= 2:
            prev_performance = self.performance_history[-2]
            performance_drop = prev_performance.map_50 - current_performance.map_50
            performance_condition = performance_drop > self.performance_threshold
        
        return time_condition and samples_condition or performance_condition
    
    async def _trigger_retraining(self, model_name: str) -> str:
        """Trigger model retraining"""
        job_id = f"retrain_{model_name}_{int(time.time())}"
        
        training_job = TrainingJob(
            job_id=job_id,
            model_name=model_name,
            dataset_name="continuous_learning_dataset",
            status="pending",
            start_time=datetime.utcnow()
        )
        
        self.training_jobs[job_id] = training_job
        
        # Start training in background
        asyncio.create_task(self._run_training_job(job_id))
        
        logger.info(f"Triggered retraining for model {model_name}, job ID: {job_id}")
        return job_id
    
    async def _run_training_job(self, job_id: str) -> None:
        """Execute training job"""
        job = self.training_jobs[job_id]
        
        try:
            job.status = "running"
            logger.info(f"Starting training job {job_id}")
            
            # Placeholder for actual training implementation
            # Would involve:
            # 1. Preparing dataset
            # 2. Setting up training configuration
            # 3. Running training loop
            # 4. Validating results
            # 5. Saving new model
            
            for progress in range(0, 101, 10):
                job.progress = progress
                await asyncio.sleep(0.5)  # Simulate training time
            
            # Simulate training completion
            job.status = "completed"
            job.end_time = datetime.utcnow()
            job.metrics = {
                "final_map": 0.87,
                "training_loss": 0.023,
                "validation_loss": 0.031,
                "epochs_completed": 50
            }
            
            self.last_retrain_time = datetime.utcnow()
            self.new_samples_count = 0
            
            logger.info(f"Training job {job_id} completed successfully")
            
        except Exception as e:
            job.status = "failed"
            job.end_time = datetime.utcnow()
            job.error_message = str(e)
            logger.error(f"Training job {job_id} failed: {e}")
    
    def add_new_samples(self, count: int) -> None:
        """Add count of new training samples"""
        self.new_samples_count += count
    
    def get_training_status(self, job_id: Optional[str] = None) -> Union[TrainingJob, Dict[str, TrainingJob]]:
        """Get training job status"""
        if job_id:
            return self.training_jobs.get(job_id)
        return self.training_jobs


class ModelEnsembleManager:
    """
    Manages model ensemble operations:
    - Ensemble inference
    - Model weight optimization
    - Performance comparison
    """
    
    def __init__(self):
        self.config = get_config()
        self.ensemble_weights: Dict[str, float] = {}
        self.performance_metrics: Dict[str, ModelPerformance] = {}
    
    async def optimize_ensemble_weights(self, validation_data: List[Tuple[np.ndarray, List[Detection]]]) -> Dict[str, float]:
        """Optimize ensemble weights based on validation performance"""
        model_manager = await get_model_manager()
        loaded_models = model_manager.get_loaded_models()
        
        if len(loaded_models) < 2:
            return {loaded_models[0]: 1.0} if loaded_models else {}
        
        # Evaluate each model on validation data
        model_scores = {}
        
        for model_name in loaded_models:
            total_score = 0.0
            for image, ground_truth in validation_data:
                predictions = await model_manager.predict(image, model_name)
                score = self._calculate_prediction_quality(predictions, ground_truth)
                total_score += score
            
            model_scores[model_name] = total_score / len(validation_data)
        
        # Calculate weights (softmax of scores)
        scores = np.array(list(model_scores.values()))
        weights = np.exp(scores) / np.sum(np.exp(scores))
        
        self.ensemble_weights = {
            model: weight for model, weight in zip(model_scores.keys(), weights)
        }
        
        return self.ensemble_weights
    
    def _calculate_prediction_quality(self, predictions: List[Detection], 
                                    ground_truth: List[Detection]) -> float:
        """Calculate prediction quality score (simplified mAP)"""
        if not predictions and not ground_truth:
            return 1.0
        if not predictions or not ground_truth:
            return 0.0
        
        # Simplified quality calculation
        # Would implement proper mAP calculation in production
        
        total_iou = 0.0
        matches = 0
        
        for pred in predictions:
            best_iou = 0.0
            for gt in ground_truth:
                if pred.class_id == gt.class_id:
                    iou = self._calculate_iou(pred.bbox, gt.bbox)
                    best_iou = max(best_iou, iou)
            
            if best_iou > 0.5:
                matches += 1
                total_iou += best_iou
        
        return total_iou / len(predictions) if predictions else 0.0
    
    def _calculate_iou(self, bbox1: Tuple[float, float, float, float], 
                      bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate IoU between two bounding boxes"""
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


class MLPipelineManager:
    """
    Main ML Pipeline Manager that coordinates all ML operations
    """
    
    def __init__(self):
        self.config = get_config()
        self.active_learning = ActiveLearningManager()
        self.auto_annotation = AutoAnnotationManager()
        self.continuous_learning = ContinuousLearningManager()
        self.ensemble_manager = ModelEnsembleManager()
        
        self.pipeline_stats = {
            'frames_processed': 0,
            'annotations_generated': 0,
            'models_retrained': 0,
            'start_time': datetime.utcnow()
        }
    
    async def process_inference_result(self, image: np.ndarray, detections: List[Detection], 
                                     image_id: str) -> Dict[str, Any]:
        """Process inference result through ML pipeline"""
        results = {
            'active_learning': None,
            'auto_annotation': None,
            'performance_update': None
        }
        
        self.pipeline_stats['frames_processed'] += 1
        
        # Active Learning
        if self.active_learning.enabled:
            candidate = await self.active_learning.evaluate_frame(image, detections, image_id)
            results['active_learning'] = candidate
        
        # Auto-annotation
        if self.auto_annotation.enabled:
            annotation_result = await self.auto_annotation.process_detections(image_id, detections)
            results['auto_annotation'] = annotation_result
            self.pipeline_stats['annotations_generated'] += annotation_result.get('auto_annotated', 0)
        
        # Continuous Learning (periodic check)
        if (self.continuous_learning.enabled and 
            self.pipeline_stats['frames_processed'] % 1000 == 0):  # Check every 1000 frames
            
            model_manager = await get_model_manager()
            active_model = model_manager.get_active_model_name()
            if active_model:
                performance = await self.continuous_learning.monitor_performance(active_model)
                results['performance_update'] = performance
        
        return results
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        return {
            'stats': self.pipeline_stats,
            'active_learning': {
                'enabled': self.active_learning.enabled,
                'candidates_count': len(self.active_learning.annotation_candidates),
                'annotated_count': len(self.active_learning.annotated_samples),
                'budget_remaining': self.active_learning.annotation_budget - len(self.active_learning.annotation_candidates)
            },
            'auto_annotation': {
                'enabled': self.auto_annotation.enabled,
                'auto_annotations_count': len(self.auto_annotation.auto_annotations),
                'pending_review_count': len(self.auto_annotation.pending_review)
            },
            'continuous_learning': {
                'enabled': self.continuous_learning.enabled,
                'last_retrain_time': self.continuous_learning.last_retrain_time.isoformat(),
                'new_samples_count': self.continuous_learning.new_samples_count,
                'active_jobs': len([job for job in self.continuous_learning.training_jobs.values() 
                                 if job.status in ['pending', 'running']])
            },
            'ensemble': {
                'weights': self.ensemble_manager.ensemble_weights,
                'models_count': len(self.ensemble_manager.ensemble_weights)
            }
        }
    
    async def export_training_data(self, format_type: str = "YOLO") -> str:
        """Export training data in specified format"""
        db_manager = get_database_manager()
        
        # Get all annotations
        annotations = await db_manager.get_all_annotations()
        
        # Export based on format
        if format_type.upper() == "YOLO":
            return await self._export_yolo_format(annotations)
        elif format_type.upper() == "COCO":
            return await self._export_coco_format(annotations)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    async def _export_yolo_format(self, annotations: List[Dict[str, Any]]) -> str:
        """Export annotations in YOLO format"""
        export_dir = Path("data/exports/yolo")
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Group annotations by image
        image_annotations = defaultdict(list)
        for ann in annotations:
            image_annotations[ann['image_id']].append(ann)
        
        # Create YOLO format files
        for image_id, anns in image_annotations.items():
            label_file = export_dir / f"{image_id}.txt"
            
            with open(label_file, 'w') as f:
                for ann in anns:
                    # Convert bbox to YOLO format (normalized center, width, height)
                    x1, y1, x2, y2 = ann['bbox']
                    # Assuming image size is known (would get from image metadata)
                    img_w, img_h = 640, 640  # Placeholder
                    
                    center_x = (x1 + x2) / 2 / img_w
                    center_y = (y1 + y2) / 2 / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    
                    f.write(f"{ann['class_id']} {center_x} {center_y} {width} {height}\n")
        
        return str(export_dir)
    
    async def _export_coco_format(self, annotations: List[Dict[str, Any]]) -> str:
        """Export annotations in COCO format"""
        export_dir = Path("data/exports/coco")
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Create COCO format JSON
        coco_data = {
            "info": {
                "year": datetime.utcnow().year,
                "version": "1.0",
                "description": "Auto-generated COCO dataset",
                "contributor": "Advanced Object Detection System",
                "date_created": datetime.utcnow().isoformat()
            },
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Add categories (classes)
        model_manager = await get_model_manager()
        classes = []
        if model_manager.active_model:
            model_info = model_manager.get_model_info()
            if model_info:
                classes = model_info.classes
        
        for i, class_name in enumerate(classes):
            coco_data["categories"].append({
                "id": i,
                "name": class_name,
                "supercategory": "object"
            })
        
        # Add images and annotations
        image_id_map = {}
        annotation_id = 1
        
        for ann in annotations:
            # Add image if not exists
            if ann['image_id'] not in image_id_map:
                image_id_map[ann['image_id']] = len(coco_data["images"]) + 1
                coco_data["images"].append({
                    "id": image_id_map[ann['image_id']],
                    "file_name": f"{ann['image_id']}.jpg",
                    "width": 640,  # Placeholder
                    "height": 640
                })
            
            # Add annotation
            x1, y1, x2, y2 = ann['bbox']
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id_map[ann['image_id']],
                "category_id": ann['class_id'],
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO format: [x, y, width, height]
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0
            })
            annotation_id += 1
        
        # Save COCO JSON
        coco_file = export_dir / "annotations.json"
        with open(coco_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        return str(export_dir)


# Global ML pipeline manager
ml_pipeline = MLPipelineManager()


def get_ml_pipeline() -> MLPipelineManager:
    """Get global ML pipeline manager"""
    return ml_pipeline
