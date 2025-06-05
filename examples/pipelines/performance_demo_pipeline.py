from omegaconf import DictConfig
import time
import numpy as np
from typing import Dict, Any

from experiment_manager.common.common import Level, Metric, RunStatus
from experiment_manager.environment import Environment
from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.common.serializable import YAMLSerializable


@YAMLSerializable.register("PerformanceDemoPipeline")
class PerformanceDemoPipeline(Pipeline, YAMLSerializable):
    """Performance monitoring demo with proper lifecycle management using decorators."""
    
    def __init__(self, env: Environment, id: int = None):
        Pipeline.__init__(self, env)
        YAMLSerializable.__init__(self)
        self.name = "PerformanceDemoPipeline"
        
        # Initialize training components
        self.model = None  # In real implementation, initialize your model here
        self.batches_per_epoch = 8
        self.epochs = 3
    
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment, id: int = None):
        return cls(env, id)
    
    @Pipeline.run_wrapper  # CRITICAL: This decorator is required!
    def run(self, config: DictConfig) -> Dict[str, Any]:
        """Run the pipeline with proper lifecycle management."""
        
        # Get configuration with optimized defaults for faster execution
        self.epochs = config.pipeline.get('epochs', 3)
        self.batches_per_epoch = config.pipeline.get('batches_per_epoch', 8)
        work_duration = config.pipeline.get('work_duration_seconds', 0.2)  # Reduced from 1.2
        cpu_intensity = config.pipeline.get('cpu_intensity', 0.5)  # Reduced from 1.5
        memory_usage_mb = config.pipeline.get('memory_usage_mb', 100)  # Reduced from 300
        
        # Log parameters - tracked by all trackers
        self.env.tracker_manager.log_params({
            "pipeline_type": "PerformanceDemoPipeline",
            "epochs": self.epochs,
            "batches_per_epoch": self.batches_per_epoch,
            "work_duration_seconds": work_duration,
            "cpu_intensity": cpu_intensity,
            "memory_usage_mb": memory_usage_mb,
            "model_type": "ResNet50_Demo",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 64
        })
        
        # Store config for epoch use
        self.work_duration = work_duration
        self.cpu_intensity = cpu_intensity
        self.memory_usage_mb = memory_usage_mb
        
        # Training loop using run_epoch with @epoch_wrapper
        for epoch in range(self.epochs):
            self.env.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            
            # Call run_epoch with @epoch_wrapper - handles lifecycle automatically
            self.run_epoch(epoch, self.model)
        
        # Final test evaluation (after all epochs)
        self.env.logger.info("Running final test evaluation...")
        final_test_loss, final_test_acc = self._simulate_test_evaluation()
        
        # Store final metrics in run_metrics for automatic tracking
        self.run_metrics[Metric.TEST_LOSS] = final_test_loss
        self.run_metrics[Metric.TEST_ACC] = final_test_acc
        
        # Also track directly for immediate availability
        self.env.tracker_manager.track(Metric.TEST_LOSS, final_test_loss)
        self.env.tracker_manager.track(Metric.TEST_ACC, final_test_acc)
        
        self.env.logger.info(f"Final Results - Loss: {final_test_loss:.4f}, Accuracy: {final_test_acc:.4f}")
        
        return {
            "status": "completed",
            "final_test_loss": final_test_loss,
            "final_test_accuracy": final_test_acc,
            "epochs_completed": self.epochs
        }
    
    @Pipeline.epoch_wrapper  # CRITICAL: This manages epoch lifecycle and metrics automatically
    def run_epoch(self, epoch_idx: int, model, *args, **kwargs) -> RunStatus:
        """Run one epoch with proper epoch lifecycle management."""
        
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        
        # Simulate epoch-level work with increasing complexity
        epoch_complexity = 1.0 + (epoch_idx * 0.1)  # Reduced complexity increase
        
        # Batch training simulation
        for batch in range(self.batches_per_epoch):
            self.env.logger.info(f"  Processing batch {batch + 1}/{self.batches_per_epoch}")
            
            # Simulate varying workload types per batch (optimized timing)
            if batch % 3 == 0:
                # CPU-intensive batch (data preprocessing)
                self._simulate_cpu_work(self.work_duration * 0.8, self.cpu_intensity * epoch_complexity)
            elif batch % 3 == 1:
                # Memory-intensive batch (model forward pass)
                self._simulate_memory_work(int(self.memory_usage_mb * epoch_complexity))
            else:
                # Balanced workload (backward pass)
                self._simulate_mixed_work(self.work_duration * 0.6, int(self.memory_usage_mb * 0.5))
            
            # Realistic training metrics with learning progress
            base_loss = 1.5
            base_acc = 0.4
            progress = (epoch_idx * self.batches_per_epoch + batch) / (self.epochs * self.batches_per_epoch)
            
            batch_loss = base_loss * (1 - progress * 0.7) + np.random.normal(0, 0.02)
            batch_acc = base_acc + progress * 0.5 + np.random.normal(0, 0.01)
            
            # Clamp values
            batch_loss = max(0.1, batch_loss)
            batch_acc = min(0.95, max(0.0, batch_acc))
            
            epoch_train_loss += batch_loss
            epoch_train_acc += batch_acc
            
            # Track batch-level metrics with step
            step = epoch_idx * self.batches_per_epoch + batch
            self.env.tracker_manager.track(Metric.TRAIN_LOSS, batch_loss, step=step)
            self.env.tracker_manager.track(Metric.TRAIN_ACC, batch_acc, step=step)
            
            # Custom metrics
            self.env.tracker_manager.track(
                Metric.CUSTOM, 
                ("batch_processing_time", self.work_duration), 
                step=step
            )
            
            # Simulate periodic checkpoints
            if batch == self.batches_per_epoch // 2:  # Middle of epoch
                checkpoint_path = f"checkpoints/epoch_{epoch_idx}_batch_{batch}.pth"
                self.env.tracker_manager.on_checkpoint(
                    network=model,  # Pass the actual model if available
                    checkpoint_path=checkpoint_path,
                    metrics={
                        Metric.TRAIN_LOSS: batch_loss,
                        Metric.TRAIN_ACC: batch_acc,
                        "epoch": epoch_idx,
                        "batch": batch
                    }
                )
                self.env.logger.info(f"    💾 Checkpoint saved: {checkpoint_path}")
        
        # Epoch averages
        avg_train_loss = epoch_train_loss / self.batches_per_epoch
        avg_train_acc = epoch_train_acc / self.batches_per_epoch
        
        # Validation simulation (optimized)
        self.env.logger.info("  Running validation...")
        val_loss, val_acc = self._simulate_validation(avg_train_loss, avg_train_acc)
        
        # CRITICAL: Store in epoch_metrics - these will be automatically tracked by @epoch_wrapper!
        self.epoch_metrics[Metric.TRAIN_LOSS] = avg_train_loss
        self.epoch_metrics[Metric.TRAIN_ACC] = avg_train_acc
        self.epoch_metrics[Metric.VAL_LOSS] = val_loss
        self.epoch_metrics[Metric.VAL_ACC] = val_acc
        
        # Also track validation directly with epoch step
        self.env.tracker_manager.track(Metric.VAL_LOSS, val_loss, step=epoch_idx)
        self.env.tracker_manager.track(Metric.VAL_ACC, val_acc, step=epoch_idx)
        
        self.env.logger.info(f"  Epoch {epoch_idx + 1} - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        self.env.logger.info(f"  Epoch {epoch_idx + 1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        return RunStatus.SUCCESS
    
    def _simulate_cpu_work(self, duration: float, intensity: float = 1.0):
        """Simulate CPU-intensive work (optimized for faster execution)."""
        start_time = time.time()
        base_operations = int(1000 * intensity)  # Reduced from 10000
        
        while time.time() - start_time < duration:
            _ = sum(i * i for i in range(base_operations))
            if time.time() - start_time >= duration:
                break
    
    def _simulate_memory_work(self, mb: int):
        """Simulate memory-intensive work (optimized)."""
        arrays = []
        try:
            for _ in range(max(1, mb // 20)):  # Reduced memory allocation
                arrays.append(np.random.random((100, 100)))  # Smaller arrays
            time.sleep(0.02)  # Reduced sleep time
        finally:
            del arrays
    
    def _simulate_mixed_work(self, duration: float, mb: int):
        """Simulate mixed CPU and memory work (optimized)."""
        self._simulate_cpu_work(duration * 0.6, 0.5)
        self._simulate_memory_work(mb)
    
    def _simulate_validation(self, train_loss: float, train_acc: float) -> tuple:
        """Simulate validation with minimal overhead."""
        self._simulate_cpu_work(self.work_duration * 0.3, self.cpu_intensity * 0.8)
        
        # Validation metrics (slightly worse than training)
        val_loss = train_loss + 0.05 + np.random.normal(0, 0.01)
        val_acc = train_acc - 0.02 + np.random.normal(0, 0.01)
        val_loss = max(0.1, val_loss)
        val_acc = min(0.95, max(0.0, val_acc))
        
        return val_loss, val_acc
    
    def _simulate_test_evaluation(self) -> tuple:
        """Simulate final test evaluation."""
        self._simulate_cpu_work(self.work_duration * 0.5, self.cpu_intensity * 0.6)
        
        # Test metrics (best performance)
        test_loss = 0.15 + np.random.normal(0, 0.01)
        test_acc = 0.92 + np.random.normal(0, 0.01)
        test_loss = max(0.1, test_loss)
        test_acc = min(0.95, max(0.0, test_acc))
        
        return test_loss, test_acc 