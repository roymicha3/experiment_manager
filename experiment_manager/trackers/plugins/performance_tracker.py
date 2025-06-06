"""
Performance Tracker for Experiment Manager

This module provides comprehensive performance monitoring during experiment execution,
tracking CPU, memory, GPU utilization, disk I/O, and other system metrics with
cross-platform support and bottleneck detection.

PERFORMANCE OPTIMIZATIONS:
- Uses non-blocking CPU measurement (psutil.cpu_percent(interval=None)) to avoid 0.1s delays
- Implements lightweight_mode for faster initialization (default in from_config)
- Lazy GPU monitoring initialization to avoid driver detection delays
- Lazy baseline establishment to prevent blocking during tracker creation
- Graceful error handling to prevent hangs when hardware monitoring fails

LIGHTWEIGHT MODE:
- When lightweight_mode=True, GPU monitoring and baseline capture are skipped during init
- Monitoring can be upgraded to full mode with enable_full_monitoring() when needed
- Default mode from YAML configuration for optimal performance in factory creation
"""
import json
import os
import threading
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Cross-platform system monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    warnings.warn("psutil not available - limited performance monitoring", ImportWarning)

# GPU monitoring 
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

# try:
#     import nvidia_ml_py3 as nvml
#     HAS_NVML = True
# except ImportError:
    # HAS_NVML = False
HAS_NVML = False

# Plotting capabilities
try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from experiment_manager.common.common import Level, Metric
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.trackers.tracker import Tracker


@dataclass
class PerformanceSnapshot:
    """Single performance measurement snapshot."""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_available_gb: float = 0.0
    disk_read_mb_s: float = 0.0
    disk_write_mb_s: float = 0.0
    network_sent_mb_s: float = 0.0
    network_recv_mb_s: float = 0.0
    gpu_utilization: List[float] = field(default_factory=list)
    gpu_memory_used: List[float] = field(default_factory=list)
    gpu_memory_total: List[float] = field(default_factory=list)
    gpu_temperature: List[float] = field(default_factory=list)
    process_count: int = 0
    load_average: List[float] = field(default_factory=list)  # Unix only


@dataclass
class PerformanceAlert:
    """Performance alert for resource constraints."""
    timestamp: datetime
    level: str  # 'warning', 'critical'
    resource: str  # 'cpu', 'memory', 'gpu', 'disk'
    message: str
    value: float
    threshold: float


@dataclass
class BottleneckAnalysis:
    """Analysis of performance bottlenecks."""
    timestamp: datetime
    primary_bottleneck: str
    bottleneck_score: float
    recommendations: List[str]
    resource_scores: Dict[str, float]


@YAMLSerializable.register("PerformanceTracker")
class PerformanceTracker(Tracker, YAMLSerializable):
    """
    Comprehensive performance monitoring tracker for experiment execution.
    
    Monitors system performance metrics including CPU, memory, GPU, disk I/O,
    and network usage with real-time alerting and bottleneck detection.
    """
    
    def __init__(self, 
                 workspace: str,
                 monitoring_interval: float = 1.0,
                 enable_alerts: bool = True,
                 enable_bottleneck_detection: bool = True,
                 cpu_threshold: float = 90.0,
                 memory_threshold: float = 90.0,
                 gpu_threshold: float = 95.0,
                 history_size: int = 1000,
                 lightweight_mode: bool = False,
                 test_mode: bool = False):
        """
        Initialize PerformanceTracker.
        
        Args:
            workspace: Workspace directory for storing performance data
            monitoring_interval: Seconds between performance measurements
            enable_alerts: Enable resource constraint alerting
            enable_bottleneck_detection: Enable bottleneck detection
            cpu_threshold: CPU usage alert threshold (%)
            memory_threshold: Memory usage alert threshold (%)
            gpu_threshold: GPU usage alert threshold (%)
            history_size: Number of snapshots to keep in memory
            lightweight_mode: If True, skip expensive initialization for faster startup
            test_mode: If True, disable all monitoring threads for testing
        """
        super().__init__(workspace)
        
        # Ensure workspace directory exists
        os.makedirs(self.workspace, exist_ok=True)
        
        # Configuration
        self.monitoring_interval = monitoring_interval
        self.enable_alerts = enable_alerts
        self.enable_bottleneck_detection = enable_bottleneck_detection
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.gpu_threshold = gpu_threshold
        self.history_size = history_size
        self.lightweight_mode = lightweight_mode
        self.test_mode = test_mode
        
        # State
        self.id = None
        self.parent = None
        self.current_level = None
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Data storage
        self.performance_history = deque(maxlen=history_size)
        self.alerts = []
        self.bottlenecks = []
        self.level_metrics = defaultdict(list)  # Metrics per hierarchy level
        
        # GPU setup - skip in lightweight mode
        self.gpu_count = 0
        if not lightweight_mode:
            self._init_gpu_monitoring()
        
        # Baseline measurements - lazy in all modes
        self.baseline_snapshot = None
        self._establish_baseline()
        
        # File paths - use self.workspace which has been processed by base Tracker class
        self.performance_file = os.path.join(self.workspace, "performance_data.json")
        self.alerts_file = os.path.join(self.workspace, "performance_alerts.json")
        self.bottlenecks_file = os.path.join(self.workspace, "bottleneck_analysis.json")
        
    @classmethod
    def from_config(cls, config, workspace: str) -> "PerformanceTracker":
        """Create tracker from configuration."""
        return cls(
            workspace=workspace,
            monitoring_interval=config.get("monitoring_interval", 1.0),
            enable_alerts=config.get("enable_alerts", True),
            enable_bottleneck_detection=config.get("enable_bottleneck_detection", True),
            cpu_threshold=config.get("cpu_threshold", 90.0),
            memory_threshold=config.get("memory_threshold", 90.0),
            gpu_threshold=config.get("gpu_threshold", 95.0),
            history_size=config.get("history_size", 1000),
            lightweight_mode=config.get("lightweight_mode", True),  # Default to True for performance
            test_mode=config.get("test_mode", False)
        )
    
    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring capabilities."""
        self.gpu_count = 0
        
        # Try NVML first (fastest and most reliable)
        if HAS_NVML:
            try:
                nvml.nvmlInit()
                self.gpu_count = nvml.nvmlDeviceGetCount()
                return  # Success, no need to try GPUtil
            except Exception:
                pass  # Silently fail and try next method
        
        # Fallback to GPUtil
        if self.gpu_count == 0 and HAS_GPUTIL:
            try:
                gpus = GPUtil.getGPUs()
                self.gpu_count = len(gpus)
            except Exception:
                pass  # Silently fail, no GPU monitoring available
    
    def _establish_baseline(self):
        """Establish baseline performance measurements."""
        # Make baseline establishment lazy - only when actually needed
        self.baseline_snapshot = None
        
    def _get_or_establish_baseline(self):
        """Lazy initialization of baseline snapshot."""
        if self.baseline_snapshot is None and HAS_PSUTIL:
            try:
                self.baseline_snapshot = self._capture_snapshot()
            except Exception:
                pass  # Silently fail if baseline can't be established
        return self.baseline_snapshot
    
    def enable_full_monitoring(self):
        """Upgrade from lightweight mode to full monitoring."""
        if self.lightweight_mode:
            self.lightweight_mode = False
            # Initialize GPU monitoring if not already done
            if self.gpu_count == 0:
                self._init_gpu_monitoring()
            # Establish baseline if not already done
            self._get_or_establish_baseline()
    
    def _capture_snapshot(self) -> PerformanceSnapshot:
        """Capture current system performance snapshot."""
        snapshot = PerformanceSnapshot()
        
        if not HAS_PSUTIL:
            return snapshot
        
        try:
            # CPU and memory - use non-blocking CPU measurement
            snapshot.cpu_percent = psutil.cpu_percent(interval=None)  # Non-blocking
            memory = psutil.virtual_memory()
            snapshot.memory_percent = memory.percent
            snapshot.memory_used_gb = memory.used / (1024**3)
            snapshot.memory_available_gb = memory.available / (1024**3)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io and hasattr(self, '_prev_disk_io') and self._prev_disk_io:
                time_delta = time.time() - self._prev_disk_time
                if time_delta > 0:
                    snapshot.disk_read_mb_s = (disk_io.read_bytes - self._prev_disk_io.read_bytes) / (1024**2) / time_delta
                    snapshot.disk_write_mb_s = (disk_io.write_bytes - self._prev_disk_io.write_bytes) / (1024**2) / time_delta
            
            self._prev_disk_io = disk_io
            self._prev_disk_time = time.time()
            
            # Network I/O
            net_io = psutil.net_io_counters()
            if net_io and hasattr(self, '_prev_net_io') and self._prev_net_io:
                time_delta = time.time() - self._prev_net_time
                if time_delta > 0:
                    snapshot.network_sent_mb_s = (net_io.bytes_sent - self._prev_net_io.bytes_sent) / (1024**2) / time_delta
                    snapshot.network_recv_mb_s = (net_io.bytes_recv - self._prev_net_io.bytes_recv) / (1024**2) / time_delta
            
            self._prev_net_io = net_io
            self._prev_net_time = time.time()
            
            # Process count
            snapshot.process_count = len(psutil.pids())
            
            # Load average (Unix only)
            if hasattr(os, 'getloadavg'):
                snapshot.load_average = list(os.getloadavg())
            
        except Exception as e:
            warnings.warn(f"Error capturing system metrics: {e}")
        
        # GPU monitoring - only if successfully initialized
        snapshot.gpu_utilization = []
        snapshot.gpu_memory_used = []
        snapshot.gpu_memory_total = []
        snapshot.gpu_temperature = []
        
        if HAS_NVML and self.gpu_count > 0:
            try:
                for i in range(self.gpu_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # GPU utilization
                    util = nvml.nvmlDeviceGetUtilizationRates(handle)
                    snapshot.gpu_utilization.append(util.gpu)
                    
                    # GPU memory
                    mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    snapshot.gpu_memory_used.append(mem_info.used / (1024**3))  # GB
                    snapshot.gpu_memory_total.append(mem_info.total / (1024**3))  # GB
                    
                    # GPU temperature
                    temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                    snapshot.gpu_temperature.append(temp)
                    
            except Exception as e:
                warnings.warn(f"Error capturing NVIDIA GPU metrics: {e}")
        
        elif HAS_GPUTIL and self.gpu_count > 0:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    snapshot.gpu_utilization.append(gpu.load * 100)
                    snapshot.gpu_memory_used.append(gpu.memoryUsed / 1024)  # GB
                    snapshot.gpu_memory_total.append(gpu.memoryTotal / 1024)  # GB
                    snapshot.gpu_temperature.append(gpu.temperature)
            except Exception as e:
                warnings.warn(f"Error capturing GPU metrics: {e}")
        
        return snapshot
    
    def _check_alerts(self, snapshot: PerformanceSnapshot):
        """Check for performance alerts based on thresholds."""
        if not self.enable_alerts:
            return
        
        alerts = []
        
        # CPU alert
        if snapshot.cpu_percent > self.cpu_threshold:
            alerts.append(PerformanceAlert(
                timestamp=snapshot.timestamp,
                level='critical' if snapshot.cpu_percent > 95 else 'warning',
                resource='cpu',
                message=f"High CPU usage: {snapshot.cpu_percent:.1f}%",
                value=snapshot.cpu_percent,
                threshold=self.cpu_threshold
            ))
        
        # Memory alert
        if snapshot.memory_percent > self.memory_threshold:
            alerts.append(PerformanceAlert(
                timestamp=snapshot.timestamp,
                level='critical' if snapshot.memory_percent > 95 else 'warning',
                resource='memory',
                message=f"High memory usage: {snapshot.memory_percent:.1f}% ({snapshot.memory_used_gb:.1f}GB)",
                value=snapshot.memory_percent,
                threshold=self.memory_threshold
            ))
        
        # GPU alerts
        for i, gpu_util in enumerate(snapshot.gpu_utilization):
            if gpu_util > self.gpu_threshold:
                alerts.append(PerformanceAlert(
                    timestamp=snapshot.timestamp,
                    level='warning',
                    resource=f'gpu_{i}',
                    message=f"High GPU {i} usage: {gpu_util:.1f}%",
                    value=gpu_util,
                    threshold=self.gpu_threshold
                ))
        
        self.alerts.extend(alerts)
        
        # Log critical alerts
        for alert in alerts:
            if alert.level == 'critical':
                print(f"PERFORMANCE ALERT: {alert.message}")
    
    def _analyze_bottlenecks(self, snapshot: PerformanceSnapshot):
        """Analyze performance bottlenecks."""
        if not self.enable_bottleneck_detection or len(self.performance_history) < 10:
            return
        
        # Calculate resource scores (higher = more constrained)
        scores = {
            'cpu': snapshot.cpu_percent / 100.0,
            'memory': snapshot.memory_percent / 100.0,
            'disk_io': min((snapshot.disk_read_mb_s + snapshot.disk_write_mb_s) / 100.0, 1.0),
            'network': min((snapshot.network_sent_mb_s + snapshot.network_recv_mb_s) / 100.0, 1.0)
        }
        
        # Add GPU scores
        if snapshot.gpu_utilization:
            scores['gpu'] = max(snapshot.gpu_utilization) / 100.0
        
        # Find primary bottleneck
        primary_bottleneck = max(scores.items(), key=lambda x: x[1])
        
        # Generate recommendations
        recommendations = []
        if primary_bottleneck[0] == 'cpu' and primary_bottleneck[1] > 0.8:
            recommendations.extend([
                "Consider reducing batch size",
                "Enable multi-processing if available",
                "Check for CPU-intensive operations"
            ])
        elif primary_bottleneck[0] == 'memory' and primary_bottleneck[1] > 0.8:
            recommendations.extend([
                "Reduce batch size to lower memory usage",
                "Enable gradient checkpointing",
                "Consider mixed precision training"
            ])
        elif primary_bottleneck[0] == 'gpu' and primary_bottleneck[1] > 0.9:
            recommendations.extend([
                "Increase batch size to better utilize GPU",
                "Check for GPU memory bottlenecks",
                "Consider using multiple GPUs"
            ])
        elif primary_bottleneck[0] == 'disk_io' and primary_bottleneck[1] > 0.5:
            recommendations.extend([
                "Use faster storage (SSD)",
                "Increase data loading workers",
                "Consider data preprocessing"
            ])
        
        analysis = BottleneckAnalysis(
            timestamp=snapshot.timestamp,
            primary_bottleneck=primary_bottleneck[0],
            bottleneck_score=primary_bottleneck[1],
            recommendations=recommendations,
            resource_scores=scores
        )
        
        self.bottlenecks.append(analysis)
    
    def _monitoring_loop(self):
        """Main monitoring loop running in separate thread."""
        self._prev_disk_io = psutil.disk_io_counters() if HAS_PSUTIL else None
        self._prev_net_io = psutil.net_io_counters() if HAS_PSUTIL else None
        self._prev_disk_time = time.time()
        self._prev_net_time = time.time()
        
        while self.is_monitoring:
            try:
                snapshot = self._capture_snapshot()
                self.performance_history.append(snapshot)
                
                # Store metrics for current level
                if self.current_level:
                    self.level_metrics[self.current_level].append(snapshot)
                
                # Check alerts and bottlenecks
                self._check_alerts(snapshot)
                self._analyze_bottlenecks(snapshot)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                warnings.warn(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.is_monitoring or self.test_mode:
            return
        
        self.is_monitoring = True
        if not self.test_mode:
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
    
    def get_performance_summary(self, level: Optional[Level] = None) -> Dict[str, Any]:
        """Get performance summary for a specific level or overall."""
        snapshots = self.level_metrics.get(level, []) if level else list(self.performance_history)
        
        if not snapshots:
            return {}
        
        # Calculate statistics
        cpu_values = [s.cpu_percent for s in snapshots]
        memory_values = [s.memory_percent for s in snapshots]
        
        summary = {
            'measurement_count': len(snapshots),
            'time_range': {
                'start': snapshots[0].timestamp.isoformat(),
                'end': snapshots[-1].timestamp.isoformat(),
                'duration_minutes': (snapshots[-1].timestamp - snapshots[0].timestamp).total_seconds() / 60
            },
            'cpu': {
                'avg': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory': {
                'avg': sum(memory_values) / len(memory_values),
                'max': max(memory_values),
                'min': min(memory_values),
                'peak_gb': max(s.memory_used_gb for s in snapshots)
            }
        }
        
        # GPU summary
        if any(s.gpu_utilization for s in snapshots):
            gpu_utils = [max(s.gpu_utilization) for s in snapshots if s.gpu_utilization]
            gpu_memory = [max(s.gpu_memory_used) for s in snapshots if s.gpu_memory_used]
            
            summary['gpu'] = {
                'avg_utilization': sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0,
                'max_utilization': max(gpu_utils) if gpu_utils else 0,
                'peak_memory_gb': max(gpu_memory) if gpu_memory else 0,
                'gpu_count': self.gpu_count
            }
        
        # Alert summary
        level_alerts = [a for a in self.alerts if level is None or a.timestamp >= snapshots[0].timestamp]
        summary['alerts'] = {
            'total': len(level_alerts),
            'critical': len([a for a in level_alerts if a.level == 'critical']),
            'warnings': len([a for a in level_alerts if a.level == 'warning'])
        }
        
        # Bottleneck summary
        level_bottlenecks = [b for b in self.bottlenecks if level is None or b.timestamp >= snapshots[0].timestamp]
        if level_bottlenecks:
            bottleneck_counts = defaultdict(int)
            for bottleneck in level_bottlenecks:
                bottleneck_counts[bottleneck.primary_bottleneck] += 1
            
            summary['bottlenecks'] = {
                'primary_bottleneck': max(bottleneck_counts.items(), key=lambda x: x[1])[0],
                'bottleneck_distribution': dict(bottleneck_counts)
            }
        
        return summary
    
    def generate_performance_plot(self, output_path: str, level: Optional[Level] = None):
        """Generate performance visualization plot."""
        if not HAS_MATPLOTLIB:
            warnings.warn("matplotlib not available - cannot generate plots")
            return
        
        snapshots = self.level_metrics.get(level, []) if level else list(self.performance_history)
        if not snapshots:
            return
        
        # Intelligent data decimation to prevent matplotlib tick issues
        max_points = 1000  # Limit to 1000 data points for plotting
        if len(snapshots) > max_points:
            # Decimate data while preserving start, end, and key points
            step = len(snapshots) // max_points
            indices = list(range(0, len(snapshots), step))
            if indices[-1] != len(snapshots) - 1:
                indices.append(len(snapshots) - 1)  # Always include the last point
            snapshots = [snapshots[i] for i in indices]
        
        timestamps = [s.timestamp for s in snapshots]
        cpu_values = [s.cpu_percent for s in snapshots]
        memory_values = [s.memory_percent for s in snapshots]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # CPU usage
        axes[0].plot(timestamps, cpu_values, 'b-', label='CPU Usage %', linewidth=1.5)
        axes[0].axhline(y=self.cpu_threshold, color='r', linestyle='--', alpha=0.7, label='CPU Threshold')
        axes[0].set_ylabel('CPU Usage (%)')
        axes[0].set_title('System Performance During Experiment')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Memory usage
        axes[1].plot(timestamps, memory_values, 'g-', label='Memory Usage %', linewidth=1.5)
        axes[1].axhline(y=self.memory_threshold, color='r', linestyle='--', alpha=0.7, label='Memory Threshold')
        axes[1].set_ylabel('Memory Usage (%)')
        axes[1].set_xlabel('Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Smart x-axis formatting based on data duration
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else timedelta(0)
        
        for ax in axes:
            if duration.total_seconds() < 3600:  # Less than 1 hour
                # Show time in MM:SS format with minute intervals
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=max(1, int(duration.total_seconds() / 300))))
            elif duration.total_seconds() < 86400:  # Less than 1 day
                # Show time in HH:MM format with appropriate intervals
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                interval = max(1, int(duration.total_seconds() / 3600 / 6))  # Aim for ~6 ticks
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
            else:  # More than 1 day
                # Show date and time with day intervals
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            
            # Limit the maximum number of ticks to prevent overflow
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Tracker interface implementation
    
    def track(self, metric: Metric, value, step: int = None, *args, **kwargs):
        """Track a performance metric."""
        # Custom performance metrics
        if metric == Metric.CUSTOM:
            metric_name = value[0] if isinstance(value, (list, tuple)) else str(value)
            metric_value = value[1] if isinstance(value, (list, tuple)) and len(value) > 1 else step
            
            # Store custom performance tracking
            custom_data = {
                'timestamp': datetime.now().isoformat(),
                'metric': metric_name,
                'value': metric_value,
                'step': step,
                'level': self.current_level.name if self.current_level else None
            }
            
            custom_file = os.path.join(self.workspace, "custom_performance_metrics.json")
            
            # Append to custom metrics file
            existing_data = []
            if os.path.exists(custom_file):
                try:
                    with open(custom_file, 'r') as f:
                        existing_data = json.load(f)
                except Exception:
                    pass
            
            existing_data.append(custom_data)
            
            with open(custom_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
    
    def on_checkpoint(self, network, checkpoint_path: str, metrics: Optional[Dict[Metric, Any]] = None, *args, **kwargs):
        """Handle checkpoint events."""
        # Capture performance snapshot at checkpoint
        snapshot = self._capture_snapshot()
        
        checkpoint_perf = {
            'checkpoint_path': checkpoint_path,
            'timestamp': snapshot.timestamp.isoformat(),
            'cpu_percent': snapshot.cpu_percent,
            'memory_percent': snapshot.memory_percent,
            'memory_used_gb': snapshot.memory_used_gb,
            'gpu_utilization': snapshot.gpu_utilization,
            'gpu_memory_used': snapshot.gpu_memory_used
        }
        
        checkpoint_file = os.path.join(self.workspace, "checkpoint_performance.json")
        
        existing_data = []
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    existing_data = json.load(f)
            except Exception:
                pass
        
        existing_data.append(checkpoint_perf)
        
        with open(checkpoint_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
    
    def log_params(self, params: Dict[str, Any]):
        """Log performance configuration parameters."""
        perf_params = {
            'monitoring_interval': self.monitoring_interval,
            'enable_alerts': self.enable_alerts,
            'enable_bottleneck_detection': self.enable_bottleneck_detection,
            'cpu_threshold': self.cpu_threshold,
            'memory_threshold': self.memory_threshold,
            'gpu_threshold': self.gpu_threshold,
            'gpu_count': self.gpu_count,
            'has_psutil': HAS_PSUTIL,
            'has_gputil': HAS_GPUTIL,
            'has_nvml': HAS_NVML,
            'user_params': params
        }
        
        params_file = os.path.join(self.workspace, "performance_params.json")
        with open(params_file, 'w') as f:
            json.dump(perf_params, f, indent=2)
    
    def on_create(self, level: Level, *args, **kwargs):
        """Handle hierarchy level creation."""
        self.current_level = level
        self.id = f"perf_{level.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Clear level-specific metrics
        self.level_metrics[level] = []
        
        # Log level creation
        creation_data = {
            'level': level.name,
            'timestamp': datetime.now().isoformat(),
            'args': args,
            'kwargs': kwargs
        }
        
        level_file = os.path.join(self.workspace, f"level_creation_{level.name.lower()}.json")
        
        existing_data = []
        if os.path.exists(level_file):
            try:
                with open(level_file, 'r') as f:
                    existing_data = json.load(f)
            except Exception:
                pass
        
        existing_data.append(creation_data)
        
        with open(level_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
    
    def on_start(self, level: Level, *args, **kwargs):
        """Handle hierarchy level start."""
        self.current_level = level
        
        # Start monitoring for certain levels
        if level in [Level.EXPERIMENT, Level.TRIAL_RUN, Level.EPOCH]:
            self.start_monitoring()
        
        # Capture baseline for this level - but only if not in lightweight mode
        if HAS_PSUTIL and not self.lightweight_mode:
            baseline = self._capture_snapshot()
            baseline_file = os.path.join(self.workspace, f"baseline_{level.name.lower()}.json")
            
            baseline_data = {
                'level': level.name,
                'timestamp': baseline.timestamp.isoformat(),
                'cpu_percent': baseline.cpu_percent,
                'memory_percent': baseline.memory_percent,
                'memory_used_gb': baseline.memory_used_gb,
                'gpu_utilization': baseline.gpu_utilization,
                'process_count': baseline.process_count
            }
            
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f, indent=2)
    
    def on_end(self, level: Level, *args, **kwargs):
        """Handle hierarchy level end."""
        # Generate summary for this level
        summary = self.get_performance_summary(level)
        
        summary_file = os.path.join(self.workspace, f"summary_{level.name.lower()}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Generate plot for this level - skip in test mode to prevent hanging
        if HAS_MATPLOTLIB and not self.test_mode:
            plot_file = os.path.join(self.workspace, f"performance_plot_{level.name.lower()}.png")
            self.generate_performance_plot(plot_file, level)
        
        # Stop monitoring for certain levels
        if level in [Level.EXPERIMENT, Level.TRIAL_RUN] and self.is_monitoring:
            self.stop_monitoring()
    
    def on_add_artifact(self, level: Level, artifact_path: str, *args, **kwargs):
        """Handle artifact addition."""
        # Log artifact with performance context
        snapshot = self._capture_snapshot() if HAS_PSUTIL else PerformanceSnapshot()
        
        artifact_data = {
            'level': level.name,
            'artifact_path': artifact_path,
            'timestamp': snapshot.timestamp.isoformat(),
            'cpu_percent': snapshot.cpu_percent,
            'memory_percent': snapshot.memory_percent,
            'memory_used_gb': snapshot.memory_used_gb
        }
        
        artifact_file = os.path.join(self.workspace, "artifact_performance.json")
        
        existing_data = []
        if os.path.exists(artifact_file):
            try:
                with open(artifact_file, 'r') as f:
                    existing_data = json.load(f)
            except Exception:
                pass
        
        existing_data.append(artifact_data)
        
        with open(artifact_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
    
    def create_child(self, workspace: str = None) -> "PerformanceTracker":
        """Create child tracker."""
        child_workspace = workspace or os.path.join(self.workspace, "child")
        os.makedirs(child_workspace, exist_ok=True)
        
        child = PerformanceTracker(
            workspace=child_workspace,
            monitoring_interval=self.monitoring_interval,
            enable_alerts=self.enable_alerts,
            enable_bottleneck_detection=self.enable_bottleneck_detection,
            cpu_threshold=self.cpu_threshold,
            memory_threshold=self.memory_threshold,
            gpu_threshold=self.gpu_threshold,
            history_size=self.history_size,
            lightweight_mode=self.lightweight_mode,
            test_mode=self.test_mode
        )
        child.parent = self
        return child
    
    def save(self):
        """Save performance data to files."""
        # Ensure workspace exists
        os.makedirs(self.workspace, exist_ok=True)
        
        # Save performance history
        if self.performance_history:
            history_data = []
            for snapshot in self.performance_history:
                history_data.append({
                    'timestamp': snapshot.timestamp.isoformat(),
                    'cpu_percent': snapshot.cpu_percent,
                    'memory_percent': snapshot.memory_percent,
                    'memory_used_gb': snapshot.memory_used_gb,
                    'memory_available_gb': snapshot.memory_available_gb,
                    'disk_read_mb_s': snapshot.disk_read_mb_s,
                    'disk_write_mb_s': snapshot.disk_write_mb_s,
                    'network_sent_mb_s': snapshot.network_sent_mb_s,
                    'network_recv_mb_s': snapshot.network_recv_mb_s,
                    'gpu_utilization': snapshot.gpu_utilization,
                    'gpu_memory_used': snapshot.gpu_memory_used,
                    'gpu_memory_total': snapshot.gpu_memory_total,
                    'gpu_temperature': snapshot.gpu_temperature,
                    'process_count': snapshot.process_count,
                    'load_average': snapshot.load_average
                })
            
            with open(self.performance_file, 'w') as f:
                json.dump(history_data, f, indent=2)
        
        # Save alerts
        if self.alerts:
            alerts_data = []
            for alert in self.alerts:
                alerts_data.append({
                    'timestamp': alert.timestamp.isoformat(),
                    'level': alert.level,
                    'resource': alert.resource,
                    'message': alert.message,
                    'value': alert.value,
                    'threshold': alert.threshold
                })
            
            with open(self.alerts_file, 'w') as f:
                json.dump(alerts_data, f, indent=2)
        
        # Save bottleneck analysis
        if self.bottlenecks:
            bottlenecks_data = []
            for bottleneck in self.bottlenecks:
                bottlenecks_data.append({
                    'timestamp': bottleneck.timestamp.isoformat(),
                    'primary_bottleneck': bottleneck.primary_bottleneck,
                    'bottleneck_score': bottleneck.bottleneck_score,
                    'recommendations': bottleneck.recommendations,
                    'resource_scores': bottleneck.resource_scores
                })
            
            with open(self.bottlenecks_file, 'w') as f:
                json.dump(bottlenecks_data, f, indent=2)
        
        # Generate final summary
        overall_summary = self.get_performance_summary()
        summary_file = os.path.join(self.workspace, "overall_performance_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(overall_summary, f, indent=2, default=str)
        
        # Generate final plot - skip in test mode to prevent hanging
        if HAS_MATPLOTLIB and not self.test_mode and self.performance_history:
            plot_file = os.path.join(self.workspace, "overall_performance_plot.png")
            self.generate_performance_plot(plot_file)
    
    def __del__(self):
        """Cleanup on destruction."""
        self.stop_monitoring() 