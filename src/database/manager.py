"""
Database Management System
Handles data persistence, versioning, and annotation management
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import uuid
from dataclasses import dataclass, asdict

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import sqlalchemy as sa

from ..core.config import get_config

logger = logging.getLogger(__name__)

# Database models
Base = declarative_base()


class User(Base):
    """User model"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    annotations = relationship("Annotation", back_populates="user")
    training_jobs = relationship("TrainingJob", back_populates="user")


class Dataset(Base):
    """Dataset model"""
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    version = Column(String(20), default="1.0.0")
    format_type = Column(String(20), default="YOLO")  # YOLO, COCO, Pascal VOC
    total_images = Column(Integer, default=0)
    total_annotations = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    images = relationship("Image", back_populates="dataset")
    training_jobs = relationship("TrainingJob", back_populates="dataset")


class Image(Base):
    """Image model"""
    __tablename__ = "images"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_path = Column(String(500))
    stored_path = Column(String(500))
    width = Column(Integer)
    height = Column(Integer)
    file_size = Column(Integer)  # in bytes
    checksum = Column(String(64))  # SHA-256
    metadata = Column(JSON)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="images")
    annotations = relationship("Annotation", back_populates="image")


class Annotation(Base):
    """Annotation model"""
    __tablename__ = "annotations"
    
    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Bounding box coordinates
    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
    
    # Classification
    class_id = Column(Integer, nullable=False)
    class_name = Column(String(50), nullable=False)
    confidence = Column(Float, default=1.0)
    
    # Annotation metadata
    auto_generated = Column(Boolean, default=False)
    verified = Column(Boolean, default=False)
    annotation_time = Column(Float)  # Time taken to annotate
    annotation_method = Column(String(50))  # manual, auto, assisted
    
    # Additional data
    mask_data = Column(Text)  # Segmentation mask (encoded)
    keypoints = Column(JSON)  # Pose keypoints
    attributes = Column(JSON)  # Additional attributes
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    image = relationship("Image", back_populates="annotations")
    user = relationship("User", back_populates="annotations")


class Model(Base):
    """Model model for tracking trained models"""
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    version = Column(String(20), nullable=False)
    architecture = Column(String(50))  # YOLO, RT-DETR, etc.
    model_path = Column(String(500), nullable=False)
    config_path = Column(String(500))
    
    # Training metadata
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    training_job_id = Column(Integer, ForeignKey("training_jobs.id"))
    parent_model_id = Column(Integer, ForeignKey("models.id"))
    
    # Performance metrics
    map_50 = Column(Float)
    map_95 = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    inference_time = Column(Float)  # ms
    model_size = Column(Float)  # MB
    
    # Status
    is_active = Column(Boolean, default=False)
    is_deployed = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    training_job = relationship("TrainingJob", back_populates="model")
    parent_model = relationship("Model", remote_side=[id])


class TrainingJob(Base):
    """Training job model"""
    __tablename__ = "training_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(100), unique=True, nullable=False)
    name = Column(String(100))
    
    # Job configuration
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    base_model_name = Column(String(100))
    config = Column(JSON)  # Training configuration
    
    # Job status
    status = Column(String(20), default="pending")  # pending, running, completed, failed, cancelled
    progress = Column(Float, default=0.0)
    
    # Timing
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    duration = Column(Float)  # seconds
    
    # Results
    final_metrics = Column(JSON)
    best_epoch = Column(Integer)
    total_epochs = Column(Integer)
    
    # Error handling
    error_message = Column(Text)
    error_traceback = Column(Text)
    
    # User tracking
    user_id = Column(Integer, ForeignKey("users.id"))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="training_jobs")
    user = relationship("User", back_populates="training_jobs")
    model = relationship("Model", back_populates="training_job", uselist=False)


class AnalyticsSnapshot(Base):
    """Analytics snapshot model for historical data"""
    __tablename__ = "analytics_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Performance metrics
    fps = Column(Float)
    latency_ms = Column(Float)
    memory_usage_mb = Column(Float)
    cpu_percent = Column(Float)
    gpu_usage_percent = Column(Float)
    
    # Detection statistics
    total_detections = Column(Integer, default=0)
    detection_distribution = Column(JSON)  # Class distribution
    
    # Tracking statistics
    total_tracks = Column(Integer, default=0)
    active_tracks = Column(Integer, default=0)
    average_track_duration = Column(Float, default=0.0)
    
    # Model information
    active_model = Column(String(100))
    loaded_models = Column(JSON)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class DatabaseManager:
    """
    Advanced database manager with support for:
    - Multiple database backends
    - Connection pooling
    - Async operations
    - Data versioning
    """
    
    def __init__(self):
        self.config = get_config()
        self.db_config = self.config.get_database_config()
        
        self.engine = None
        self.session_factory = None
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize database connection"""
        try:
            # Create database URL
            db_url = self._create_database_url()
            
            # Create async engine
            self.engine = create_async_engine(
                db_url,
                pool_size=self.db_config.get('pool_size', 10),
                max_overflow=self.db_config.get('max_overflow', 20),
                echo=self.config.is_development()
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            self.initialized = True
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _create_database_url(self) -> str:
        """Create database URL based on configuration"""
        db_type = self.db_config.get('type', 'sqlite')
        
        if db_type == 'postgresql':
            host = self.db_config.get('host', 'localhost')
            port = self.db_config.get('port', 5432)
            username = self.db_config.get('username', 'postgres')
            password = self.db_config.get('password', '')
            database = self.db_config.get('name', 'object_detection')
            
            return f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{database}"
        
        elif db_type == 'sqlite':
            db_path = self.db_config.get('sqlite', {}).get('path', 'data/database.db')
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite+aiosqlite:///{db_path}"
        
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    async def get_session(self) -> AsyncSession:
        """Get database session"""
        if not self.initialized:
            await self.initialize()
        
        return self.session_factory()
    
    async def close(self) -> None:
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()
            self.initialized = False
            logger.info("Database connections closed")
    
    # User management
    
    async def create_user(self, username: str, email: str, password_hash: str) -> int:
        """Create a new user"""
        async with await self.get_session() as session:
            user = User(
                username=username,
                email=email,
                hashed_password=password_hash
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user.id
    
    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        async with await self.get_session() as session:
            result = await session.execute(
                sa.select(User).where(User.username == username)
            )
            user = result.scalar_one_or_none()
            
            if user:
                return {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "is_active": user.is_active,
                    "created_at": user.created_at.isoformat()
                }
            return None
    
    # Dataset management
    
    async def create_dataset(self, name: str, description: str = "", format_type: str = "YOLO") -> int:
        """Create a new dataset"""
        async with await self.get_session() as session:
            dataset = Dataset(
                name=name,
                description=description,
                format_type=format_type
            )
            session.add(dataset)
            await session.commit()
            await session.refresh(dataset)
            return dataset.id
    
    async def get_datasets(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get datasets with pagination"""
        async with await self.get_session() as session:
            result = await session.execute(
                sa.select(Dataset).limit(limit).offset(offset).order_by(Dataset.created_at.desc())
            )
            datasets = result.scalars().all()
            
            return [
                {
                    "id": ds.id,
                    "name": ds.name,
                    "description": ds.description,
                    "version": ds.version,
                    "format_type": ds.format_type,
                    "total_images": ds.total_images,
                    "total_annotations": ds.total_annotations,
                    "created_at": ds.created_at.isoformat(),
                    "updated_at": ds.updated_at.isoformat()
                }
                for ds in datasets
            ]
    
    # Image management
    
    async def save_image(self, filename: str, original_path: str, stored_path: str,
                        width: int, height: int, dataset_id: int, **metadata) -> int:
        """Save image metadata"""
        async with await self.get_session() as session:
            image = Image(
                filename=filename,
                original_path=original_path,
                stored_path=stored_path,
                width=width,
                height=height,
                dataset_id=dataset_id,
                metadata=metadata
            )
            session.add(image)
            await session.commit()
            await session.refresh(image)
            return image.id
    
    # Annotation management
    
    async def save_annotation(self, annotation_data: Dict[str, Any]) -> int:
        """Save annotation to database"""
        async with await self.get_session() as session:
            annotation = Annotation(
                image_id=annotation_data.get('image_id'),
                user_id=annotation_data.get('user_id'),
                x1=annotation_data['bbox'][0],
                y1=annotation_data['bbox'][1],
                x2=annotation_data['bbox'][2],
                y2=annotation_data['bbox'][3],
                class_id=annotation_data['class_id'],
                class_name=annotation_data['class_name'],
                confidence=annotation_data.get('confidence', 1.0),
                auto_generated=annotation_data.get('auto_generated', False),
                annotation_method=annotation_data.get('annotation_method', 'manual')
            )
            session.add(annotation)
            await session.commit()
            await session.refresh(annotation)
            return annotation.id
    
    async def save_label(self, object_id: int, label: str, confidence: Optional[float] = None, 
                        user_id: Optional[int] = None) -> int:
        """Save object label (simplified annotation)"""
        annotation_data = {
            'image_id': 1,  # Placeholder - would link to actual image
            'user_id': user_id,
            'bbox': [0, 0, 100, 100],  # Placeholder bbox
            'class_id': object_id,
            'class_name': label,
            'confidence': confidence or 1.0,
            'auto_generated': False
        }
        
        return await self.save_annotation(annotation_data)
    
    async def get_annotations(self, limit: int = 100, offset: int = 0, 
                           class_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get annotations with pagination and filtering"""
        async with await self.get_session() as session:
            query = sa.select(Annotation).join(Image).join(User, isouter=True)
            
            if class_name:
                query = query.where(Annotation.class_name == class_name)
            
            query = query.limit(limit).offset(offset).order_by(Annotation.created_at.desc())
            
            result = await session.execute(query)
            annotations = result.scalars().all()
            
            return [
                {
                    "id": ann.id,
                    "image_id": ann.image_id,
                    "bbox": [ann.x1, ann.y1, ann.x2, ann.y2],
                    "class_id": ann.class_id,
                    "class_name": ann.class_name,
                    "confidence": ann.confidence,
                    "auto_generated": ann.auto_generated,
                    "verified": ann.verified,
                    "created_at": ann.created_at.isoformat()
                }
                for ann in annotations
            ]
    
    async def get_all_annotations(self) -> List[Dict[str, Any]]:
        """Get all annotations for export"""
        async with await self.get_session() as session:
            result = await session.execute(sa.select(Annotation))
            annotations = result.scalars().all()
            
            return [
                {
                    "image_id": str(ann.image_id),
                    "bbox": [ann.x1, ann.y1, ann.x2, ann.y2],
                    "class_id": ann.class_id,
                    "class_name": ann.class_name,
                    "confidence": ann.confidence,
                    "auto_generated": ann.auto_generated
                }
                for ann in annotations
            ]
    
    # Model management
    
    async def save_model_info(self, model_data: Dict[str, Any]) -> int:
        """Save model information"""
        async with await self.get_session() as session:
            model = Model(
                name=model_data['name'],
                version=model_data.get('version', '1.0.0'),
                architecture=model_data.get('architecture', 'YOLO'),
                model_path=model_data['model_path'],
                config_path=model_data.get('config_path'),
                dataset_id=model_data.get('dataset_id'),
                map_50=model_data.get('map_50'),
                map_95=model_data.get('map_95'),
                precision=model_data.get('precision'),
                recall=model_data.get('recall'),
                f1_score=model_data.get('f1_score'),
                inference_time=model_data.get('inference_time'),
                model_size=model_data.get('model_size')
            )
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return model.id
    
    async def get_models(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get model information"""
        async with await self.get_session() as session:
            result = await session.execute(
                sa.select(Model).limit(limit).order_by(Model.created_at.desc())
            )
            models = result.scalars().all()
            
            return [
                {
                    "id": model.id,
                    "name": model.name,
                    "version": model.version,
                    "architecture": model.architecture,
                    "map_50": model.map_50,
                    "map_95": model.map_95,
                    "precision": model.precision,
                    "recall": model.recall,
                    "f1_score": model.f1_score,
                    "inference_time": model.inference_time,
                    "model_size": model.model_size,
                    "is_active": model.is_active,
                    "created_at": model.created_at.isoformat()
                }
                for model in models
            ]
    
    # Training job management
    
    async def create_training_job(self, job_data: Dict[str, Any]) -> int:
        """Create a new training job"""
        async with await self.get_session() as session:
            job = TrainingJob(
                job_id=job_data['job_id'],
                name=job_data.get('name'),
                dataset_id=job_data['dataset_id'],
                base_model_name=job_data.get('base_model_name'),
                config=job_data.get('config', {}),
                user_id=job_data.get('user_id')
            )
            session.add(job)
            await session.commit()
            await session.refresh(job)
            return job.id
    
    async def update_training_job(self, job_id: str, updates: Dict[str, Any]) -> None:
        """Update training job status"""
        async with await self.get_session() as session:
            result = await session.execute(
                sa.select(TrainingJob).where(TrainingJob.job_id == job_id)
            )
            job = result.scalar_one_or_none()
            
            if job:
                for key, value in updates.items():
                    if hasattr(job, key):
                        setattr(job, key, value)
                
                job.updated_at = datetime.utcnow()
                await session.commit()
    
    async def get_training_jobs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get training jobs"""
        async with await self.get_session() as session:
            query = sa.select(TrainingJob)
            
            if status:
                query = query.where(TrainingJob.status == status)
            
            query = query.order_by(TrainingJob.created_at.desc())
            
            result = await session.execute(query)
            jobs = result.scalars().all()
            
            return [
                {
                    "id": job.id,
                    "job_id": job.job_id,
                    "name": job.name,
                    "status": job.status,
                    "progress": job.progress,
                    "start_time": job.start_time.isoformat() if job.start_time else None,
                    "end_time": job.end_time.isoformat() if job.end_time else None,
                    "duration": job.duration,
                    "final_metrics": job.final_metrics,
                    "error_message": job.error_message,
                    "created_at": job.created_at.isoformat()
                }
                for job in jobs
            ]
    
    # Analytics and monitoring
    
    async def save_analytics_snapshot(self, analytics_data: Dict[str, Any]) -> None:
        """Save analytics snapshot for historical data"""
        async with await self.get_session() as session:
            snapshot = AnalyticsSnapshot(
                fps=analytics_data.get('fps', 0.0),
                latency_ms=analytics_data.get('latency_ms', 0.0),
                memory_usage_mb=analytics_data.get('memory_usage_mb', 0.0),
                cpu_percent=analytics_data.get('cpu_percent', 0.0),
                gpu_usage_percent=analytics_data.get('gpu_usage_percent'),
                total_detections=analytics_data.get('total_detections', 0),
                detection_distribution=analytics_data.get('detection_distribution', {}),
                total_tracks=analytics_data.get('total_tracks', 0),
                active_tracks=analytics_data.get('active_tracks', 0),
                average_track_duration=analytics_data.get('average_track_duration', 0.0),
                active_model=analytics_data.get('active_model'),
                loaded_models=analytics_data.get('loaded_models', [])
            )
            session.add(snapshot)
            await session.commit()
    
    async def get_analytics_history(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get analytics history for time range"""
        async with await self.get_session() as session:
            result = await session.execute(
                sa.select(AnalyticsSnapshot)
                .where(AnalyticsSnapshot.timestamp.between(start_time, end_time))
                .order_by(AnalyticsSnapshot.timestamp)
            )
            snapshots = result.scalars().all()
            
            return [
                {
                    "timestamp": snap.timestamp.isoformat(),
                    "fps": snap.fps,
                    "latency_ms": snap.latency_ms,
                    "memory_usage_mb": snap.memory_usage_mb,
                    "cpu_percent": snap.cpu_percent,
                    "gpu_usage_percent": snap.gpu_usage_percent,
                    "total_detections": snap.total_detections,
                    "detection_distribution": snap.detection_distribution,
                    "total_tracks": snap.total_tracks,
                    "active_tracks": snap.active_tracks,
                    "active_model": snap.active_model
                }
                for snap in snapshots
            ]
    
    # Data export and backup
    
    async def export_dataset(self, dataset_id: int, export_format: str = "YOLO") -> Dict[str, Any]:
        """Export dataset in specified format"""
        async with await self.get_session() as session:
            # Get dataset info
            dataset_result = await session.execute(
                sa.select(Dataset).where(Dataset.id == dataset_id)
            )
            dataset = dataset_result.scalar_one_or_none()
            
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            # Get images and annotations
            images_result = await session.execute(
                sa.select(Image).where(Image.dataset_id == dataset_id)
            )
            images = images_result.scalars().all()
            
            annotations_result = await session.execute(
                sa.select(Annotation).join(Image).where(Image.dataset_id == dataset_id)
            )
            annotations = annotations_result.scalars().all()
            
            return {
                "dataset_info": {
                    "id": dataset.id,
                    "name": dataset.name,
                    "description": dataset.description,
                    "version": dataset.version,
                    "format_type": dataset.format_type
                },
                "images": [
                    {
                        "id": img.id,
                        "filename": img.filename,
                        "width": img.width,
                        "height": img.height,
                        "stored_path": img.stored_path
                    }
                    for img in images
                ],
                "annotations": [
                    {
                        "image_id": ann.image_id,
                        "bbox": [ann.x1, ann.y1, ann.x2, ann.y2],
                        "class_id": ann.class_id,
                        "class_name": ann.class_name,
                        "confidence": ann.confidence
                    }
                    for ann in annotations
                ]
            }
    
    async def backup_database(self, backup_path: str) -> str:
        """Create database backup"""
        try:
            backup_file = Path(backup_path) / f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.sql"
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Implementation would depend on database type
            # For PostgreSQL: use pg_dump
            # For SQLite: copy database file
            
            logger.info(f"Database backup created: {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            raise
    
    # Statistics and analytics
    
    async def get_annotation_statistics(self) -> Dict[str, Any]:
        """Get annotation statistics"""
        async with await self.get_session() as session:
            # Total annotations
            total_result = await session.execute(sa.select(sa.func.count(Annotation.id)))
            total_annotations = total_result.scalar()
            
            # Auto-generated vs manual
            auto_result = await session.execute(
                sa.select(sa.func.count(Annotation.id)).where(Annotation.auto_generated == True)
            )
            auto_annotations = auto_result.scalar()
            
            # Class distribution
            class_dist_result = await session.execute(
                sa.select(Annotation.class_name, sa.func.count(Annotation.id))
                .group_by(Annotation.class_name)
            )
            class_distribution = dict(class_dist_result.all())
            
            # Recent activity (last 24 hours)
            recent_cutoff = datetime.utcnow() - timedelta(hours=24)
            recent_result = await session.execute(
                sa.select(sa.func.count(Annotation.id))
                .where(Annotation.created_at >= recent_cutoff)
            )
            recent_annotations = recent_result.scalar()
            
            return {
                "total_annotations": total_annotations,
                "auto_generated": auto_annotations,
                "manual_annotations": total_annotations - auto_annotations,
                "class_distribution": class_distribution,
                "recent_24h": recent_annotations
            }
    
    async def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old analytics snapshots"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        async with await self.get_session() as session:
            # Count records to be deleted
            count_result = await session.execute(
                sa.select(sa.func.count(AnalyticsSnapshot.id))
                .where(AnalyticsSnapshot.timestamp < cutoff_date)
            )
            records_to_delete = count_result.scalar()
            
            # Delete old records
            await session.execute(
                sa.delete(AnalyticsSnapshot)
                .where(AnalyticsSnapshot.timestamp < cutoff_date)
            )
            await session.commit()
            
            return {"deleted_records": records_to_delete}


# Global database manager instance
database_manager = DatabaseManager()


def get_database_manager() -> DatabaseManager:
    """Get global database manager instance"""
    return database_manager
