"""
Advanced Model Manager
Supports multiple AI models including YOLOv8, YOLOv9, RT-DETR, and more
"""

import torch
import numpy as np
import cv2
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from ..core.config import get_config

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types"""
    YOLO = "yolo"
    RT_DETR = "rt-detr"
    DETR = "detr"
    FASTER_RCNN = "faster-rcnn"
    SSD = "ssd"


@dataclass
class Detection:
    """Detection result data class"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    mask: Optional[np.ndarray] = None
    keypoints: Optional[np.ndarray] = None
    track_id: Optional[int] = None


@dataclass
class ModelInfo:
    """Model information data class"""
    name: str
    type: ModelType
    version: str
    size: str
    path: str
    input_size: Tuple[int, int]
    classes: List[str]
    is_loaded: bool = False
    load_time: float = 0.0
    memory_usage: float = 0.0


class BaseModel(ABC):
    """Base class for all object detection models"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = self._get_device(device)
        self.model = None
        self.is_loaded = False
        self.model_info: Optional[ModelInfo] = None
    
    def _get_device(self, device: str) -> str:
        """Determine the best device for inference"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model"""
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray, **kwargs) -> List[Detection]:
        """Run inference on image"""
        pass
    
    @abstractmethod
    def get_classes(self) -> List[str]:
        """Get model class names"""
        pass
    
    def unload_model(self) -> None:
        """Unload model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
            self.is_loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Model {self.model_path} unloaded")


class YOLOModel(BaseModel):
    """YOLO model implementation (v5, v8, v9)"""
    
    def __init__(self, model_path: str, device: str = "auto", version: str = "v8"):
        super().__init__(model_path, device)
        self.version = version
    
    def load_model(self) -> None:
        """Load YOLO model"""
        try:
            start_time = time.time()
            
            if self.version in ["v8", "v9"]:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                self.model.to(self.device)
            else:  # v5
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                          path=self.model_path, force_reload=True)
                self.model.to(self.device)
            
            self.is_loaded = True
            load_time = time.time() - start_time
            
            # Create model info
            self.model_info = ModelInfo(
                name=Path(self.model_path).stem,
                type=ModelType.YOLO,
                version=self.version,
                size=self._get_model_size(),
                path=self.model_path,
                input_size=(640, 640),
                classes=self.get_classes(),
                is_loaded=True,
                load_time=load_time,
                memory_usage=self._get_memory_usage()
            )
            
            logger.info(f"YOLO {self.version} model loaded in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def predict(self, image: np.ndarray, **kwargs) -> List[Detection]:
        """Run YOLO inference"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        confidence = kwargs.get('confidence', 0.25)
        iou_threshold = kwargs.get('iou_threshold', 0.45)
        
        try:
            # Run inference
            if self.version in ["v8", "v9"]:
                results = self.model(image, conf=confidence, iou=iou_threshold)
                return self._parse_ultralytics_results(results[0])
            else:  # v5
                results = self.model(image)
                return self._parse_yolov5_results(results, confidence)
                
        except Exception as e:
            logger.error(f"YOLO inference failed: {e}")
            return []
    
    def _parse_ultralytics_results(self, result) -> List[Detection]:
        """Parse Ultralytics YOLO results"""
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().numpy()
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                # Get mask if available (segmentation)
                mask = None
                if hasattr(result, 'masks') and result.masks is not None:
                    mask = result.masks.data[i].cpu().numpy()
                
                # Get keypoints if available (pose)
                keypoints = None
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints = result.keypoints.data[i].cpu().numpy()
                
                detection = Detection(
                    bbox=tuple(bbox),
                    confidence=float(conf),
                    class_id=cls_id,
                    class_name=self.get_classes()[cls_id],
                    mask=mask,
                    keypoints=keypoints
                )
                detections.append(detection)
        
        return detections
    
    def _parse_yolov5_results(self, results, confidence: float) -> List[Detection]:
        """Parse YOLOv5 results"""
        detections = []
        
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            if conf >= confidence:
                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(conf),
                    class_id=int(cls),
                    class_name=self.get_classes()[int(cls)]
                )
                detections.append(detection)
        
        return detections
    
    def get_classes(self) -> List[str]:
        """Get YOLO class names"""
        if not self.is_loaded:
            return []
        
        if hasattr(self.model, 'names'):
            return list(self.model.names.values())
        elif hasattr(self.model, 'module') and hasattr(self.model.module, 'names'):
            return list(self.model.module.names.values())
        else:
            return [f"class_{i}" for i in range(80)]  # COCO default
    
    def _get_model_size(self) -> str:
        """Determine model size from path"""
        path_lower = self.model_path.lower()
        if 'nano' in path_lower or 'n.pt' in path_lower:
            return "nano"
        elif 'small' in path_lower or 's.pt' in path_lower:
            return "small"
        elif 'medium' in path_lower or 'm.pt' in path_lower:
            return "medium"
        elif 'large' in path_lower or 'l.pt' in path_lower:
            return "large"
        elif 'xlarge' in path_lower or 'x.pt' in path_lower:
            return "xlarge"
        else:
            return "unknown"
    
    def _get_memory_usage(self) -> float:
        """Get model memory usage in MB"""
        if self.model is None:
            return 0.0
        
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)


class RTDETRModel(BaseModel):
    """RT-DETR model implementation"""
    
    def load_model(self) -> None:
        """Load RT-DETR model"""
        try:
            start_time = time.time()
            
            # RT-DETR through Ultralytics
            from ultralytics import RTDETR
            self.model = RTDETR(self.model_path)
            self.model.to(self.device)
            
            self.is_loaded = True
            load_time = time.time() - start_time
            
            self.model_info = ModelInfo(
                name=Path(self.model_path).stem,
                type=ModelType.RT_DETR,
                version="1.0",
                size="large",
                path=self.model_path,
                input_size=(640, 640),
                classes=self.get_classes(),
                is_loaded=True,
                load_time=load_time,
                memory_usage=self._get_memory_usage()
            )
            
            logger.info(f"RT-DETR model loaded in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load RT-DETR model: {e}")
            raise
    
    def predict(self, image: np.ndarray, **kwargs) -> List[Detection]:
        """Run RT-DETR inference"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        confidence = kwargs.get('confidence', 0.25)
        
        try:
            results = self.model(image, conf=confidence)
            return self._parse_results(results[0])
        except Exception as e:
            logger.error(f"RT-DETR inference failed: {e}")
            return []
    
    def _parse_results(self, result) -> List[Detection]:
        """Parse RT-DETR results"""
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().numpy()
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                detection = Detection(
                    bbox=tuple(bbox),
                    confidence=float(conf),
                    class_id=cls_id,
                    class_name=self.get_classes()[cls_id]
                )
                detections.append(detection)
        
        return detections
    
    def get_classes(self) -> List[str]:
        """Get RT-DETR class names"""
        if not self.is_loaded:
            return []
        
        if hasattr(self.model, 'names'):
            return list(self.model.names.values())
        else:
            return [f"class_{i}" for i in range(80)]  # COCO default
    
    def _get_memory_usage(self) -> float:
        """Get model memory usage in MB"""
        if self.model is None:
            return 0.0
        
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)


class ModelManager:
    """
    Advanced model manager with support for multiple models,
    model switching, ensemble inference, and performance monitoring
    """
    
    def __init__(self):
        self.config = get_config()
        self.models: Dict[str, BaseModel] = {}
        self.active_model: Optional[str] = None
        self.model_cache: Dict[str, BaseModel] = {}
        self.inference_stats: Dict[str, Dict] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def initialize(self) -> None:
        """Initialize model manager"""
        try:
            # Load default model
            default_model = self.config.get('models.default_model', 'yolov8n')
            await self.load_model(default_model)
            self.active_model = default_model
            
            logger.info(f"Model manager initialized with {default_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model manager: {e}")
            raise
    
    async def load_model(self, model_name: str, force_reload: bool = False) -> None:
        """Load a specific model"""
        if model_name in self.models and not force_reload:
            logger.info(f"Model {model_name} already loaded")
            return
        
        try:
            model_config = self.config.get_model_config(model_name)
            model_type = ModelType(model_config['type'])
            
            # Create model instance based on type
            if model_type == ModelType.YOLO:
                version = model_config.get('version', 'v8')
                model = YOLOModel(model_config['path'], device="auto", version=version)
            elif model_type == ModelType.RT_DETR:
                model = RTDETRModel(model_config['path'], device="auto")
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, model.load_model)
            
            # Store model
            self.models[model_name] = model
            self.inference_stats[model_name] = {
                'total_inferences': 0,
                'total_time': 0.0,
                'avg_fps': 0.0,
                'last_inference_time': 0.0
            }
            
            logger.info(f"Model {model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    async def unload_model(self, model_name: str) -> None:
        """Unload a specific model"""
        if model_name in self.models:
            model = self.models[model_name]
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, model.unload_model)
            del self.models[model_name]
            logger.info(f"Model {model_name} unloaded")
    
    def switch_model(self, model_name: str) -> None:
        """Switch active model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        self.active_model = model_name
        logger.info(f"Switched to model: {model_name}")
    
    async def predict(self, image: np.ndarray, model_name: Optional[str] = None, **kwargs) -> List[Detection]:
        """Run inference with specified model or active model"""
        model_name = model_name or self.active_model
        
        if not model_name or model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        model = self.models[model_name]
        
        # Track inference time
        start_time = time.time()
        
        try:
            # Run inference in thread pool
            loop = asyncio.get_event_loop()
            detections = await loop.run_in_executor(
                self.executor, 
                model.predict, 
                image, 
                **kwargs
            )
            
            # Update stats
            inference_time = time.time() - start_time
            self._update_stats(model_name, inference_time)
            
            return detections
            
        except Exception as e:
            logger.error(f"Inference failed for model {model_name}: {e}")
            return []
    
    async def ensemble_predict(self, image: np.ndarray, models: Optional[List[str]] = None, **kwargs) -> List[Detection]:
        """Run ensemble inference across multiple models"""
        if not models:
            models = list(self.models.keys())
        
        # Run inference on all models concurrently
        tasks = []
        for model_name in models:
            if model_name in self.models:
                task = self.predict(image, model_name, **kwargs)
                tasks.append(task)
        
        if not tasks:
            return []
        
        # Wait for all predictions
        all_predictions = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and combine results
        valid_predictions = [
            pred for pred in all_predictions 
            if isinstance(pred, list)
        ]
        
        if not valid_predictions:
            return []
        
        # Apply ensemble logic (NMS across models)
        return self._ensemble_nms(valid_predictions, **kwargs)
    
    def _ensemble_nms(self, predictions_list: List[List[Detection]], **kwargs) -> List[Detection]:
        """Apply Non-Maximum Suppression across ensemble predictions"""
        all_detections = []
        for predictions in predictions_list:
            all_detections.extend(predictions)
        
        if not all_detections:
            return []
        
        # Convert to tensors for NMS
        boxes = torch.tensor([det.bbox for det in all_detections])
        scores = torch.tensor([det.confidence for det in all_detections])
        labels = torch.tensor([det.class_id for det in all_detections])
        
        # Apply NMS
        iou_threshold = kwargs.get('iou_threshold', 0.45)
        keep_indices = torch.ops.torchvision.nms(boxes, scores, iou_threshold)
        
        # Return filtered detections
        return [all_detections[i] for i in keep_indices]
    
    def _update_stats(self, model_name: str, inference_time: float) -> None:
        """Update inference statistics"""
        stats = self.inference_stats[model_name]
        stats['total_inferences'] += 1
        stats['total_time'] += inference_time
        stats['last_inference_time'] = inference_time
        
        if stats['total_inferences'] > 0:
            stats['avg_fps'] = 1.0 / (stats['total_time'] / stats['total_inferences'])
    
    def get_model_info(self, model_name: Optional[str] = None) -> Optional[ModelInfo]:
        """Get information about a model"""
        model_name = model_name or self.active_model
        if model_name and model_name in self.models:
            return self.models[model_name].model_info
        return None
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self.config.get('models.available_models', {}).keys())
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        return list(self.models.keys())
    
    def get_inference_stats(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get inference statistics"""
        if model_name:
            return self.inference_stats.get(model_name, {})
        return self.inference_stats
    
    def get_active_model_name(self) -> Optional[str]:
        """Get active model name"""
        return self.active_model
    
    async def warm_up_model(self, model_name: Optional[str] = None) -> None:
        """Warm up model with dummy inference"""
        model_name = model_name or self.active_model
        if not model_name or model_name not in self.models:
            return
        
        # Create dummy image
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        try:
            await self.predict(dummy_image, model_name)
            logger.info(f"Model {model_name} warmed up")
        except Exception as e:
            logger.warning(f"Failed to warm up model {model_name}: {e}")
    
    async def benchmark_model(self, model_name: str, num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark model performance"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        times = []
        
        # Warm up
        await self.predict(dummy_image, model_name)
        
        # Benchmark
        for _ in range(num_iterations):
            start_time = time.time()
            await self.predict(dummy_image, model_name)
            times.append(time.time() - start_time)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'mean_fps': 1.0 / np.mean(times),
            'iterations': num_iterations
        }
    
    def cleanup(self) -> None:
        """Cleanup all loaded models"""
        for model_name in list(self.models.keys()):
            asyncio.create_task(self.unload_model(model_name))
        
        self.executor.shutdown(wait=True)
        logger.info("Model manager cleaned up")


# Global model manager instance
model_manager = ModelManager()


async def get_model_manager() -> ModelManager:
    """Get global model manager instance"""
    if not model_manager.models:
        await model_manager.initialize()
    return model_manager
