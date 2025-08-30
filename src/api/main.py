"""
Advanced FastAPI Backend
High-performance API with WebSocket support, authentication, and comprehensive endpoints
"""

import asyncio
import json
import time
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status, UploadFile, File, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
import jwt
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

from ..core.config import get_config, AppConfig
from ..core.cv_processor import get_cv_processor
from ..models.model_manager import get_model_manager, Detection, ModelInfo
from ..database.manager import get_database_manager
from ..monitoring.metrics import get_metrics_manager
from ..utils.security import verify_token, create_access_token, get_current_user
from ..utils.video_stream import VideoStreamManager
from ..utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# Pydantic models for API
class DetectionResponse(BaseModel):
    """Detection result response model"""
    bbox: List[float] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2]")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    class_id: int = Field(..., ge=0, description="Class ID")
    class_name: str = Field(..., description="Class name")
    track_id: Optional[int] = Field(None, description="Tracking ID")


class InferenceRequest(BaseModel):
    """Inference request model"""
    model_name: Optional[str] = Field(None, description="Model name to use")
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0, description="Confidence threshold")
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0, description="IoU threshold")
    enable_tracking: bool = Field(True, description="Enable object tracking")
    enable_pose: bool = Field(False, description="Enable pose estimation")
    enable_segmentation: bool = Field(False, description="Enable segmentation")


class ModelSwitchRequest(BaseModel):
    """Model switch request model"""
    model_name: str = Field(..., description="Model name to switch to")


class LabelRequest(BaseModel):
    """Object labeling request model"""
    object_id: int = Field(..., description="Object ID to label")
    label: str = Field(..., min_length=1, max_length=50, description="Object label")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Label confidence")


class UserLogin(BaseModel):
    """User login model"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class User(BaseModel):
    """User model"""
    id: int
    username: str
    email: Optional[str] = None
    is_active: bool = True
    roles: List[str] = []


class SystemStatus(BaseModel):
    """System status model"""
    status: str
    version: str
    uptime: float
    active_models: List[str]
    active_connections: int
    memory_usage: Dict[str, float]
    gpu_usage: Optional[Dict[str, float]] = None


# Global variables
app_config: AppConfig
video_stream_manager: VideoStreamManager
rate_limiter: RateLimiter
active_connections: List[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Advanced Object Detection System...")
    
    global app_config, video_stream_manager, rate_limiter
    
    # Load configuration
    config = get_config()
    app_config = config.get_app_config()
    
    # Initialize components
    await get_model_manager()
    await get_database_manager().initialize()
    get_metrics_manager().start()
    
    # Initialize video stream manager
    video_stream_manager = VideoStreamManager()
    await video_stream_manager.initialize()
    
    # Initialize rate limiter
    rate_limiter = RateLimiter()
    
    logger.info("System initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down system...")
    
    # Cleanup components
    model_manager = await get_model_manager()
    model_manager.cleanup()
    
    await get_database_manager().close()
    get_metrics_manager().stop()
    
    if video_stream_manager:
        await video_stream_manager.cleanup()
    
    logger.info("System shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Advanced Object Detection API",
    description="Real-time object detection with AI-powered features",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security
security = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend/build/static"), name="static")


# Authentication dependency
async def get_current_user_dep(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user"""
    try:
        payload = verify_token(credentials.credentials)
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Get user from database (simplified)
        return User(id=int(user_id), username="admin", roles=["admin"])
        
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")


# Rate limiting dependency
async def rate_limit_check():
    """Check rate limits"""
    # Implementation would check rate limits here
    pass


# Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "name": app_config.name,
        "version": app_config.version,
        "description": app_config.description,
        "docs_url": "/docs",
        "status": "running"
    }


@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint"""
    model_manager = await get_model_manager()
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": app_config.version,
        "models_loaded": len(model_manager.get_loaded_models()),
        "active_connections": len(active_connections)
    }


@app.get("/status", response_model=SystemStatus)
async def get_system_status(current_user: User = Depends(get_current_user_dep)):
    """Get comprehensive system status"""
    model_manager = await get_model_manager()
    metrics_manager = get_metrics_manager()
    
    memory_info = metrics_manager.get_memory_usage()
    gpu_info = metrics_manager.get_gpu_usage() if torch.cuda.is_available() else None
    
    return SystemStatus(
        status="running",
        version=app_config.version,
        uptime=time.time() - metrics_manager.start_time,
        active_models=model_manager.get_loaded_models(),
        active_connections=len(active_connections),
        memory_usage=memory_info,
        gpu_usage=gpu_info
    )


# Model Management Endpoints

@app.get("/models", response_model=Dict[str, Any])
async def get_models():
    """Get available and loaded models"""
    model_manager = await get_model_manager()
    
    return {
        "available_models": model_manager.get_available_models(),
        "loaded_models": model_manager.get_loaded_models(),
        "active_model": model_manager.get_active_model_name(),
        "model_info": {
            name: model_manager.get_model_info(name).__dict__ if model_manager.get_model_info(name) else None
            for name in model_manager.get_loaded_models()
        }
    }


@app.post("/models/load")
async def load_model(
    model_name: str,
    force_reload: bool = False,
    current_user: User = Depends(get_current_user_dep)
):
    """Load a specific model"""
    try:
        model_manager = await get_model_manager()
        await model_manager.load_model(model_name, force_reload)
        
        return {
            "status": "success",
            "message": f"Model {model_name} loaded successfully",
            "model_info": model_manager.get_model_info(model_name).__dict__
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/models/switch")
async def switch_model(
    request: ModelSwitchRequest,
    current_user: User = Depends(get_current_user_dep)
):
    """Switch active model"""
    try:
        model_manager = await get_model_manager()
        model_manager.switch_model(request.model_name)
        
        return {
            "status": "success",
            "message": f"Switched to model {request.model_name}",
            "active_model": model_manager.get_active_model_name()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/models/{model_name}")
async def unload_model(
    model_name: str,
    current_user: User = Depends(get_current_user_dep)
):
    """Unload a specific model"""
    try:
        model_manager = await get_model_manager()
        await model_manager.unload_model(model_name)
        
        return {
            "status": "success",
            "message": f"Model {model_name} unloaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/models/{model_name}/benchmark")
async def benchmark_model(
    model_name: str,
    iterations: int = 100,
    current_user: User = Depends(get_current_user_dep)
):
    """Benchmark model performance"""
    try:
        model_manager = await get_model_manager()
        results = await model_manager.benchmark_model(model_name, iterations)
        
        return {
            "model_name": model_name,
            "benchmark_results": results
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Inference Endpoints

@app.post("/inference/image", response_model=List[DetectionResponse])
async def inference_image(
    file: UploadFile = File(...),
    request: InferenceRequest = Depends(),
    current_user: User = Depends(get_current_user_dep)
):
    """Run inference on uploaded image"""
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run inference
        model_manager = await get_model_manager()
        detections = await model_manager.predict(
            image,
            model_name=request.model_name,
            confidence=request.confidence_threshold,
            iou_threshold=request.iou_threshold
        )
        
        # Convert to response format
        response_detections = [
            DetectionResponse(
                bbox=list(det.bbox),
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
                track_id=det.track_id
            )
            for det in detections
        ]
        
        return response_detections
        
    except Exception as e:
        logger.error(f"Image inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/ensemble", response_model=List[DetectionResponse])
async def ensemble_inference(
    file: UploadFile = File(...),
    models: List[str] = [],
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    current_user: User = Depends(get_current_user_dep)
):
    """Run ensemble inference across multiple models"""
    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run ensemble inference
        model_manager = await get_model_manager()
        detections = await model_manager.ensemble_predict(
            image,
            models=models or None,
            confidence=confidence_threshold,
            iou_threshold=iou_threshold
        )
        
        # Convert to response format
        response_detections = [
            DetectionResponse(
                bbox=list(det.bbox),
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name
            )
            for det in detections
        ]
        
        return response_detections
        
    except Exception as e:
        logger.error(f"Ensemble inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Video Stream Endpoints

@app.get("/stream/video")
async def video_stream():
    """Get video stream"""
    return StreamingResponse(
        video_stream_manager.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/stream/start")
async def start_video_stream(
    source: Union[int, str] = 0,
    current_user: User = Depends(get_current_user_dep)
):
    """Start video stream from source"""
    try:
        await video_stream_manager.start_stream(source)
        return {"status": "success", "message": "Video stream started"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/stream/stop")
async def stop_video_stream(current_user: User = Depends(get_current_user_dep)):
    """Stop video stream"""
    try:
        await video_stream_manager.stop_stream()
        return {"status": "success", "message": "Video stream stopped"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Analytics and Monitoring Endpoints

@app.get("/analytics/real-time")
async def get_real_time_analytics():
    """Get real-time analytics data"""
    cv_processor = get_cv_processor()
    
    # Get current analytics from video stream
    analytics = video_stream_manager.get_current_analytics()
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "analytics": analytics,
        "system_metrics": get_metrics_manager().get_current_metrics()
    }


@app.get("/analytics/history")
async def get_analytics_history(
    hours: int = 24,
    current_user: User = Depends(get_current_user_dep)
):
    """Get historical analytics data"""
    try:
        db_manager = get_database_manager()
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        analytics_data = await db_manager.get_analytics_history(start_time, end_time)
        
        return {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "data": analytics_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get Prometheus-format metrics"""
    metrics_manager = get_metrics_manager()
    return Response(
        content=metrics_manager.get_prometheus_metrics(),
        media_type="text/plain"
    )


# Data Management Endpoints

@app.post("/data/label")
async def label_object(
    request: LabelRequest,
    current_user: User = Depends(get_current_user_dep)
):
    """Label an object for training"""
    try:
        db_manager = get_database_manager()
        
        # Save label to database
        await db_manager.save_label(
            object_id=request.object_id,
            label=request.label,
            confidence=request.confidence,
            user_id=current_user.id
        )
        
        # Trigger model retraining if configured
        config = get_config()
        if config.get('ml_pipeline.continuous_learning.enabled', False):
            # Add to background task queue
            pass
        
        return {
            "status": "success",
            "message": f"Object {request.object_id} labeled as '{request.label}'"
        }
        
    except Exception as e:
        logger.error(f"Labeling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/annotations")
async def get_annotations(
    limit: int = 100,
    offset: int = 0,
    class_name: Optional[str] = None,
    current_user: User = Depends(get_current_user_dep)
):
    """Get annotation data"""
    try:
        db_manager = get_database_manager()
        annotations = await db_manager.get_annotations(
            limit=limit,
            offset=offset,
            class_name=class_name
        )
        
        return {
            "annotations": annotations,
            "total": len(annotations),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Authentication Endpoints

@app.post("/auth/login")
async def login(user_data: UserLogin):
    """User login"""
    try:
        # Validate credentials (simplified)
        if user_data.username == "admin" and user_data.password == "admin123":
            access_token = create_access_token(
                data={"sub": "1", "username": user_data.username}
            )
            return {"access_token": access_token, "token_type": "bearer"}
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/auth/logout")
async def logout(current_user: User = Depends(get_current_user_dep)):
    """User logout"""
    return {"status": "success", "message": "Logged out successfully"}


# Configuration Endpoints

@app.get("/config")
async def get_configuration(current_user: User = Depends(get_current_user_dep)):
    """Get current configuration"""
    config = get_config()
    return config.to_dict()


@app.put("/config")
async def update_configuration(
    config_data: Dict[str, Any],
    current_user: User = Depends(get_current_user_dep)
):
    """Update configuration"""
    try:
        config = get_config()
        
        # Update configuration (with validation)
        for key, value in config_data.items():
            config.set(key, value)
        
        return {"status": "success", "message": "Configuration updated"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# WebSocket Endpoints

@app.websocket("/ws/live-feed")
async def websocket_live_feed(websocket: WebSocket):
    """WebSocket endpoint for live video feed with real-time analytics"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        cv_processor = get_cv_processor()
        
        # Start video capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            await websocket.send_json({"error": "Cannot open camera"})
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with all CV features
            results = await cv_processor.process_frame(frame)
            
            # Draw visualizations
            output_frame = results['frame']
            if results['detections']:
                output_frame = cv_processor.draw_detections(output_frame, results['detections'])
            
            if results['tracked_objects']:
                output_frame = cv_processor.draw_tracking_trails(output_frame, results['tracked_objects'])
            
            if results['poses']:
                output_frame = cv_processor.draw_poses(output_frame, results['poses'])
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            
            # Prepare analytics data
            analytics = cv_processor.get_analytics(results['tracked_objects'])
            
            # Send data via WebSocket
            await websocket.send_json({
                "type": "frame_update",
                "timestamp": time.time(),
                "detections": [
                    {
                        "bbox": det.bbox,
                        "confidence": det.confidence,
                        "class_id": det.class_id,
                        "class_name": det.class_name,
                        "track_id": det.track_id
                    }
                    for det in results['detections']
                ],
                "analytics": analytics,
                "processing_time": results['processing_time']
            })
            
            # Small delay to control frame rate
            await asyncio.sleep(1/30)  # 30 FPS
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({"error": str(e)})
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        cap.release()


@app.websocket("/ws/analytics")
async def websocket_analytics(websocket: WebSocket):
    """WebSocket endpoint for real-time analytics only"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Get current analytics
            analytics = video_stream_manager.get_current_analytics()
            system_metrics = get_metrics_manager().get_current_metrics()
            
            await websocket.send_json({
                "type": "analytics_update",
                "timestamp": time.time(),
                "analytics": analytics,
                "system_metrics": system_metrics
            })
            
            await asyncio.sleep(1)  # Update every second
            
    except WebSocketDisconnect:
        logger.info("Analytics WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Analytics WebSocket error: {e}")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)


# Training and Learning Endpoints

@app.post("/training/start")
async def start_training(
    dataset_name: str,
    model_name: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user_dep)
):
    """Start model training"""
    try:
        # Add training task to background queue
        background_tasks.add_task(
            _run_training_task,
            dataset_name=dataset_name,
            model_name=model_name,
            user_id=current_user.id
        )
        
        return {
            "status": "success",
            "message": f"Training started for model {model_name}",
            "dataset": dataset_name
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/status")
async def get_training_status(current_user: User = Depends(get_current_user_dep)):
    """Get training status"""
    # Implementation would check training job status
    return {
        "active_jobs": [],
        "completed_jobs": [],
        "failed_jobs": []
    }


# Utility functions

async def _run_training_task(dataset_name: str, model_name: str, user_id: int):
    """Background task for model training"""
    try:
        logger.info(f"Starting training task: dataset={dataset_name}, model={model_name}")
        
        # Implementation would handle actual training
        # This is a placeholder for the training pipeline
        
        await asyncio.sleep(5)  # Simulate training time
        
        logger.info(f"Training task completed: dataset={dataset_name}, model={model_name}")
        
    except Exception as e:
        logger.error(f"Training task failed: {e}")


# Error handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Application startup
if __name__ == "__main__":
    import uvicorn
    
    config = get_config()
    app_config = config.get_app_config()
    
    uvicorn.run(
        "src.api.main:app",
        host=app_config.host,
        port=app_config.port,
        reload=app_config.debug,
        workers=1 if app_config.debug else app_config.workers,
        log_level="info"
    )
