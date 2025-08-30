"""
Advanced Monitoring and Metrics System
Real-time performance monitoring, analytics, and alerting
"""

import time
import psutil
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
import json
import asyncio
from enum import Enum

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

from ..core.config import get_config

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Individual metric data"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class Alert:
    """Alert data"""
    alert_id: str
    level: AlertLevel
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class PerformanceMetrics:
    """System performance metrics"""
    fps: float
    latency_ms: float
    memory_usage_mb: float
    cpu_percent: float
    gpu_usage_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    active_connections: int = 0
    processed_frames: int = 0
    detection_count: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MetricsCollector:
    """Collects various system and application metrics"""
    
    def __init__(self):
        self.config = get_config()
        self.monitoring_config = self.config.get_monitoring_config()
        
        # Initialize NVML for GPU monitoring
        self.gpu_available = False
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.gpu_available = True
                logger.info(f"GPU monitoring initialized: {self.gpu_count} GPUs detected")
            except Exception as e:
                logger.warning(f"GPU monitoring not available: {e}")
        
        # Metrics storage
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.current_metrics: Dict[str, Metric] = {}
        
        # Performance tracking
        self.fps_tracker = deque(maxlen=30)
        self.latency_tracker = deque(maxlen=100)
        self.frame_times = deque(maxlen=30)
        
        self.start_time = time.time()
        self.last_frame_time = time.time()
        
    def record_frame_processed(self) -> None:
        """Record that a frame was processed"""
        current_time = time.time()
        
        # Calculate FPS
        frame_interval = current_time - self.last_frame_time
        if frame_interval > 0:
            fps = 1.0 / frame_interval
            self.fps_tracker.append(fps)
        
        self.frame_times.append(current_time)
        self.last_frame_time = current_time
        
        # Update metrics
        self._update_metric("frames_processed_total", 1, MetricType.COUNTER)
        self._update_metric("current_fps", self.get_current_fps(), MetricType.GAUGE)
    
    def record_inference_time(self, inference_time: float) -> None:
        """Record inference time"""
        self.latency_tracker.append(inference_time * 1000)  # Convert to ms
        self._update_metric("inference_latency_ms", inference_time * 1000, MetricType.HISTOGRAM)
    
    def record_detection(self, count: int) -> None:
        """Record detection count"""
        self._update_metric("detections_total", count, MetricType.COUNTER)
    
    def get_current_fps(self) -> float:
        """Get current FPS"""
        if len(self.fps_tracker) == 0:
            return 0.0
        return float(np.mean(self.fps_tracker))
    
    def get_average_latency(self) -> float:
        """Get average inference latency in ms"""
        if len(self.latency_tracker) == 0:
            return 0.0
        return float(np.mean(self.latency_tracker))
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get system memory usage"""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        return {
            "system_total_mb": memory.total / (1024 * 1024),
            "system_used_mb": memory.used / (1024 * 1024),
            "system_available_mb": memory.available / (1024 * 1024),
            "system_percent": memory.percent,
            "process_mb": process.memory_info().rss / (1024 * 1024),
            "process_percent": process.memory_percent()
        }
    
    def get_cpu_usage(self) -> Dict[str, float]:
        """Get CPU usage information"""
        return {
            "percent": psutil.cpu_percent(interval=None),
            "count": psutil.cpu_count(),
            "count_logical": psutil.cpu_count(logical=True)
        }
    
    def get_gpu_usage(self) -> Optional[Dict[str, Any]]:
        """Get GPU usage information"""
        if not self.gpu_available:
            return None
        
        gpu_info = {}
        
        try:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Memory information
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                # Power
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                except:
                    power = 0.0
                
                gpu_info[f"gpu_{i}"] = {
                    "utilization_percent": util.gpu,
                    "memory_utilization_percent": util.memory,
                    "memory_total_mb": mem_info.total / (1024 * 1024),
                    "memory_used_mb": mem_info.used / (1024 * 1024),
                    "memory_free_mb": mem_info.free / (1024 * 1024),
                    "temperature_c": temp,
                    "power_watts": power
                }
        
        except Exception as e:
            logger.error(f"Failed to get GPU info: {e}")
            return None
        
        return gpu_info
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get comprehensive performance metrics"""
        memory_info = self.get_memory_usage()
        cpu_info = self.get_cpu_usage()
        gpu_info = self.get_gpu_usage()
        
        return PerformanceMetrics(
            fps=self.get_current_fps(),
            latency_ms=self.get_average_latency(),
            memory_usage_mb=memory_info["process_mb"],
            cpu_percent=cpu_info["percent"],
            gpu_usage_percent=gpu_info["gpu_0"]["utilization_percent"] if gpu_info else None,
            gpu_memory_mb=gpu_info["gpu_0"]["memory_used_mb"] if gpu_info else None,
            processed_frames=len(self.frame_times)
        )
    
    def _update_metric(self, name: str, value: float, metric_type: MetricType, labels: Optional[Dict[str, str]] = None) -> None:
        """Update a metric value"""
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            labels=labels or {},
            metric_type=metric_type
        )
        
        self.current_metrics[name] = metric
        self.metrics_history[name].append(metric)


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.config = get_config()
        self.alert_config = self.config.get('monitoring.alerts', {})
        self.enabled = self.alert_config.get('enabled', False)
        
        self.thresholds = self.alert_config.get('alert_thresholds', {})
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Notification settings
        self.email_enabled = self.alert_config.get('email_notifications', False)
        self.slack_webhook = self.alert_config.get('slack_webhook')
    
    def check_thresholds(self, metrics: PerformanceMetrics) -> List[Alert]:
        """Check metrics against thresholds and generate alerts"""
        if not self.enabled:
            return []
        
        new_alerts = []
        
        # Check memory usage
        memory_threshold = self.thresholds.get('high_memory_usage', 85)
        if metrics.memory_usage_mb > 0:
            memory_percent = (metrics.memory_usage_mb / psutil.virtual_memory().total) * 100
            if memory_percent > memory_threshold:
                alert = self._create_alert(
                    "high_memory_usage",
                    AlertLevel.WARNING,
                    f"High memory usage: {memory_percent:.1f}%",
                    "memory_usage_percent",
                    memory_threshold,
                    memory_percent
                )
                new_alerts.append(alert)
        
        # Check FPS
        fps_threshold = self.thresholds.get('low_fps', 15)
        if metrics.fps < fps_threshold:
            alert = self._create_alert(
                "low_fps",
                AlertLevel.WARNING,
                f"Low FPS detected: {metrics.fps:.1f}",
                "fps",
                fps_threshold,
                metrics.fps
            )
            new_alerts.append(alert)
        
        # Check latency
        latency_threshold = self.thresholds.get('high_latency', 200)
        if metrics.latency_ms > latency_threshold:
            alert = self._create_alert(
                "high_latency",
                AlertLevel.WARNING,
                f"High latency detected: {metrics.latency_ms:.1f}ms",
                "latency_ms",
                latency_threshold,
                metrics.latency_ms
            )
            new_alerts.append(alert)
        
        # GPU checks if available
        if metrics.gpu_usage_percent is not None:
            gpu_threshold = self.thresholds.get('high_gpu_usage', 90)
            if metrics.gpu_usage_percent > gpu_threshold:
                alert = self._create_alert(
                    "high_gpu_usage",
                    AlertLevel.WARNING,
                    f"High GPU usage: {metrics.gpu_usage_percent:.1f}%",
                    "gpu_usage_percent",
                    gpu_threshold,
                    metrics.gpu_usage_percent
                )
                new_alerts.append(alert)
        
        # Store alerts
        for alert in new_alerts:
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
        
        return new_alerts
    
    def _create_alert(self, alert_type: str, level: AlertLevel, message: str, 
                     metric_name: str, threshold: float, current_value: float) -> Alert:
        """Create a new alert"""
        alert_id = f"{alert_type}_{int(time.time())}"
        
        return Alert(
            alert_id=alert_id,
            level=level,
            message=message,
            metric_name=metric_name,
            threshold=threshold,
            current_value=current_value
        )
    
    async def send_notifications(self, alerts: List[Alert]) -> None:
        """Send notifications for alerts"""
        for alert in alerts:
            if self.slack_webhook and alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
                await self._send_slack_notification(alert)
    
    async def _send_slack_notification(self, alert: Alert) -> None:
        """Send Slack notification"""
        try:
            import aiohttp
            
            payload = {
                "text": f"🚨 {alert.level.value.upper()} Alert",
                "attachments": [
                    {
                        "color": "danger" if alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL] else "warning",
                        "fields": [
                            {"title": "Message", "value": alert.message, "short": False},
                            {"title": "Metric", "value": alert.metric_name, "short": True},
                            {"title": "Threshold", "value": str(alert.threshold), "short": True},
                            {"title": "Current Value", "value": str(alert.current_value), "short": True},
                            {"title": "Time", "value": alert.timestamp.isoformat(), "short": True}
                        ]
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.slack_webhook, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Slack notification sent for alert {alert.alert_id}")
                    else:
                        logger.error(f"Failed to send Slack notification: {response.status}")
        
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            del self.active_alerts[alert_id]
            return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]


class AnalyticsDashboard:
    """Real-time analytics dashboard data"""
    
    def __init__(self):
        self.detection_stats = defaultdict(int)
        self.class_distribution = defaultdict(int)
        self.hourly_stats = defaultdict(int)
        self.tracking_stats = {
            'total_tracks': 0,
            'active_tracks': 0,
            'average_track_duration': 0.0
        }
        
        # Time series data
        self.fps_history = deque(maxlen=100)
        self.latency_history = deque(maxlen=100)
        self.detection_count_history = deque(maxlen=100)
        
        self.last_update = datetime.utcnow()
    
    def update_detection_stats(self, detections: List[Any], tracked_objects: List[Any] = None) -> None:
        """Update detection statistics"""
        current_hour = datetime.utcnow().hour
        
        # Update detection counts
        for detection in detections:
            self.detection_stats['total_detections'] += 1
            self.class_distribution[detection.class_name] += 1
            self.hourly_stats[current_hour] += 1
        
        # Update tracking stats
        if tracked_objects:
            self.tracking_stats['total_tracks'] = len(tracked_objects)
            self.tracking_stats['active_tracks'] = len([obj for obj in tracked_objects if obj.is_active])
        
        self.last_update = datetime.utcnow()
    
    def update_performance_metrics(self, fps: float, latency: float, detection_count: int) -> None:
        """Update performance metrics"""
        self.fps_history.append(fps)
        self.latency_history.append(latency)
        self.detection_count_history.append(detection_count)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data"""
        return {
            "detection_stats": dict(self.detection_stats),
            "class_distribution": dict(self.class_distribution),
            "hourly_stats": dict(self.hourly_stats),
            "tracking_stats": self.tracking_stats,
            "performance": {
                "current_fps": list(self.fps_history)[-10:] if self.fps_history else [0],
                "current_latency": list(self.latency_history)[-10:] if self.latency_history else [0],
                "detection_trends": list(self.detection_count_history)[-10:] if self.detection_count_history else [0]
            },
            "last_update": self.last_update.isoformat()
        }


class MetricsManager:
    """
    Main metrics manager that coordinates collection, monitoring, and alerting
    """
    
    def __init__(self):
        self.config = get_config()
        self.monitoring_config = self.config.get_monitoring_config()
        self.enabled = self.monitoring_config.get('enabled', True)
        
        self.collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.dashboard = AnalyticsDashboard()
        
        # Monitoring thread
        self.monitoring_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Callbacks for metric updates
        self.metric_callbacks: List[Callable[[str, float], None]] = []
        
        self.start_time = time.time()
    
    def start(self) -> None:
        """Start metrics collection"""
        if not self.enabled:
            logger.info("Metrics monitoring disabled")
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Metrics monitoring started")
    
    def stop(self) -> None:
        """Stop metrics collection"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Metrics monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check for alerts
                performance_metrics = self.collector.get_performance_metrics()
                alerts = self.alert_manager.check_thresholds(performance_metrics)
                
                if alerts:
                    asyncio.create_task(self.alert_manager.send_notifications(alerts))
                
                # Update dashboard
                self.dashboard.update_performance_metrics(
                    performance_metrics.fps,
                    performance_metrics.latency_ms,
                    performance_metrics.detection_count
                )
                
                # Sleep for monitoring interval
                time.sleep(1.0)  # Collect metrics every second
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)  # Wait longer on error
    
    def _collect_system_metrics(self) -> None:
        """Collect system-level metrics"""
        # Memory metrics
        memory_info = self.collector.get_memory_usage()
        for key, value in memory_info.items():
            self.collector._update_metric(f"memory_{key}", value, MetricType.GAUGE)
        
        # CPU metrics
        cpu_info = self.collector.get_cpu_usage()
        for key, value in cpu_info.items():
            self.collector._update_metric(f"cpu_{key}", value, MetricType.GAUGE)
        
        # GPU metrics
        gpu_info = self.collector.get_gpu_usage()
        if gpu_info:
            for gpu_id, gpu_data in gpu_info.items():
                for key, value in gpu_data.items():
                    self.collector._update_metric(f"{gpu_id}_{key}", value, MetricType.GAUGE)
    
    def record_frame_processed(self) -> None:
        """Record frame processing event"""
        self.collector.record_frame_processed()
    
    def record_inference_time(self, inference_time: float) -> None:
        """Record inference timing"""
        self.collector.record_inference_time(inference_time)
    
    def record_detections(self, detections: List[Any], tracked_objects: List[Any] = None) -> None:
        """Record detection results"""
        self.collector.record_detection(len(detections))
        self.dashboard.update_detection_stats(detections, tracked_objects)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        performance = self.collector.get_performance_metrics()
        
        return {
            "performance": performance.__dict__,
            "system": {
                "memory": self.collector.get_memory_usage(),
                "cpu": self.collector.get_cpu_usage(),
                "gpu": self.collector.get_gpu_usage()
            },
            "alerts": {
                "active": len(self.alert_manager.get_active_alerts()),
                "recent": len(self.alert_manager.get_alert_history(hours=1))
            },
            "uptime": time.time() - self.start_time
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get analytics dashboard data"""
        return self.dashboard.get_dashboard_data()
    
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        lines = []
        
        # Add custom metrics
        for name, metric in self.collector.current_metrics.items():
            # Format metric name for Prometheus
            metric_name = name.replace('-', '_').replace(' ', '_').lower()
            
            # Add labels if any
            labels_str = ""
            if metric.labels:
                label_pairs = [f'{k}="{v}"' for k, v in metric.labels.items()]
                labels_str = "{" + ",".join(label_pairs) + "}"
            
            lines.append(f"object_detection_{metric_name}{labels_str} {metric.value}")
        
        # Add system metrics
        memory_info = self.collector.get_memory_usage()
        lines.append(f"system_memory_usage_mb {memory_info['process_mb']}")
        lines.append(f"system_memory_percent {memory_info['process_percent']}")
        
        cpu_info = self.collector.get_cpu_usage()
        lines.append(f"system_cpu_percent {cpu_info['percent']}")
        
        gpu_info = self.collector.get_gpu_usage()
        if gpu_info:
            for gpu_id, gpu_data in gpu_info.items():
                for key, value in gpu_data.items():
                    lines.append(f"gpu_{key}{{gpu=\"{gpu_id}\"}} {value}")
        
        return "\n".join(lines)
    
    def add_metric_callback(self, callback: Callable[[str, float], None]) -> None:
        """Add callback for metric updates"""
        self.metric_callbacks.append(callback)
    
    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[Metric]:
        """Get metric history for specified time period"""
        if metric_name not in self.collector.metrics_history:
            return []
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            metric for metric in self.collector.metrics_history[metric_name]
            if metric.timestamp >= cutoff_time
        ]
    
    def get_alerts(self, active_only: bool = False) -> List[Alert]:
        """Get alerts"""
        if active_only:
            return self.alert_manager.get_active_alerts()
        return self.alert_manager.get_alert_history()
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        return self.alert_manager.resolve_alert(alert_id)


# Global metrics manager instance
metrics_manager = MetricsManager()


def get_metrics_manager() -> MetricsManager:
    """Get global metrics manager instance"""
    return metrics_manager
