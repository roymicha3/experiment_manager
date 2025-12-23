#!/usr/bin/env python3
"""
Factory for Custom Metrics Pipeline

This factory creates the CustomMetricsTestPipeline for the custom metrics example.
"""

from omegaconf import DictConfig
from experiment_manager import Environment, Pipeline
from experiment_manager.common import Factory

from .custom_metrics_pipeline import CustomMetricsTestPipeline


class CustomMetricsPipelineFactory(Factory):
    """Factory for creating CustomMetricsTestPipeline instances."""
    
    @staticmethod
    def create(pipeline_type: str, config: DictConfig, env: Environment, id: int = None) -> Pipeline:
        """
        Create a pipeline instance.
        
        Args:
            pipeline_type: Type of pipeline to create
            config: Configuration for the pipeline
            env: Environment instance
            id: Optional pipeline ID
            
        Returns:
            Pipeline instance
        """
        if pipeline_type == "CustomMetricsTestPipeline":
            return CustomMetricsTestPipeline.from_config(config, env, id)
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
